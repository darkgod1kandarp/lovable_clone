import datetime
import json
import os
import threading
import uuid

from langchain_anthropic             import ChatAnthropic
from langchain_groq                  import ChatGroq
from langchain_ollama                import ChatOllama
from langchain_google_genai          import ChatGoogleGenerativeAI as ChatGemini
from langchain_qwq                   import ChatQwen
from langchain_core.messages         import HumanMessage, trim_messages
from langgraph.graph                 import StateGraph
from langgraph.prebuilt              import create_react_agent
from langgraph.constants             import END
from langgraph.checkpoint.memory     import MemorySaver
from dotenv                          import load_dotenv

from agent.prompts import planner_prompt, architect_prompt, coder_system_prompt, resolver_prompt
from agent.states  import Plan, TaskPlan, CoderState, AgentState
from agent.sandbox import get_sandbox
from agent.crawl_4ai import *
import json   

load_dotenv()

MAX_RESOLVE_RETRIES   = 8     # Max times resolver retries the same error
MAX_RUN_RETRIES       = 10    # Max times runner+resolver loop retries
MAX_TOOL_OUTPUT_CHARS = 2000  # Truncate tool outputs beyond this length
MAX_RECURSION_LIMIT   = 400    # Max tool-call rounds per react_agent invocation
MAX_TOKENS_IN_HISTORY = 4000  # Max tokens kept in message history per agent
API_CALL_LOG_PATH     = os.path.join(os.path.dirname(__file__), "api_call_log.txt")


_phase_local = threading.local()


def log_api_call(api_name: str):
    """Append an API call entry to the log file with timestamp and API name."""
    timestamp = datetime.datetime.now().isoformat()
    with open(API_CALL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {api_name}\n")

def _trim_messages_manually(messages: list, max_chars: int = MAX_TOKENS_IN_HISTORY) -> list:
    """
    Manually trim message history to stay within token budget.
    Always keeps the system message (first message) + most recent messages.
    This replaces the deprecated state_modifier / trim_messages approach.
    """
    if not messages:
        return messages

    # Separate system message from the rest
    system_msgs  = [m for m in messages if getattr(m, "type", None) == "system"
                    or (isinstance(m, dict) and m.get("role") == "system")]
    other_msgs   = [m for m in messages if m not in system_msgs]

    # Walk backwards through non-system messages, keeping as many as fit
    kept   = []
    total  = 0
    for msg in reversed(other_msgs):
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        size    = len(str(content))
        if total + size > max_chars:
            break
        kept.append(msg)
        total += size

    kept.reverse()
    trimmed = system_msgs + kept

    if len(trimmed) < len(messages):
        print(f"  [Trimmer] Trimmed {len(messages) - len(trimmed)} messages "
              f"({total} chars kept, limit={max_chars})")

    return trimmed


def _make_react_agent(llm, tools):
    """
    Creates a react agent.
    Message trimming is handled manually in _invoke_react_agent.
    SandboxManager methods catch all E2B exceptions and return error strings,
    so tool failures are reported to the LLM instead of crashing the agent.
    """
    return create_react_agent(llm, tools)


def _log_tool_calls(result: dict, agent_label: str = "agent"):
    """
    Print every tool call + truncated result from a react_agent result dict.
    Works by scanning messages for AIMessage.tool_calls and ToolMessage responses.
    """
    messages = result.get("messages", [])
    calls = []
    responses = {}

    for msg in messages:
        # AIMessage carries a list of tool_calls: [{id, name, args}, ...]
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                calls.append(tc)

        # ToolMessage carries the output of a single tool call
        if getattr(msg, "type", None) == "tool" or msg.__class__.__name__ == "ToolMessage":
            tid  = getattr(msg, "tool_call_id", "?")
            out  = str(getattr(msg, "content", ""))[:200]
            responses[tid] = out

    if not calls:
        print(f"  [{agent_label}] No tool calls recorded.")
        return

    print(f"  [{agent_label}] Tool calls ({len(calls)}):")
    for tc in calls:
        tid  = tc.get("id", "?")
        name = tc.get("name", "?")
        args = str(tc.get("args", {}))[:120]
        out  = responses.get(tid, "(no response)")
        print(f"    ► {name}({args})")
        print(f"      ↳ {out}")


def _invoke_react_agent(agent, messages: list) -> dict:
    """
    Invoke a react agent with manual message trimming and a recursion cap.
    Tools are already bound to the sandbox instance via SandboxManager.get_tools(),
    so no project_id threading through config is needed.
    """
    trimmed = _trim_messages_manually(messages)
    return agent.invoke(
        {"messages": trimmed},
        config={"recursion_limit": MAX_RECURSION_LIMIT},
    )


def _get_tools(project_id: str, include_process_tools: bool = False) -> list:
    """Return LangChain tools bound to the sandbox for project_id."""
    return get_sandbox(project_id).get_tools(include_process_tools=include_process_tools)


def _extract_json_from_agent_result(result: dict) -> dict:
    """
    Robustly extract a JSON object from a react_agent result.
    Scans messages from last to first so a tool-error explanation
    in the final message doesn't hide valid JSON in an earlier one.
    """

    def _content_to_str(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(content)

    def _try_parse(raw: str) -> dict | None:
        cleaned = raw.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 2:
                inner = parts[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                cleaned = inner.strip()

        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            return None

        try:
            parsed = json.loads(cleaned[start:end])
            if isinstance(parsed, dict) and any(
                k in parsed for k in ("resolved", "success", "error_message")
            ):
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    messages = result.get("messages", [])
    for msg in reversed(messages):
        raw_content = getattr(msg, "content", None)
        if raw_content is None:
            continue
        text   = _content_to_str(raw_content)
        parsed = _try_parse(text)
        if parsed is not None:
            return parsed

    last_content = ""
    if messages:
        last_content = _content_to_str(getattr(messages[-1], "content", ""))
    print(f"[JSON Parse Error] No valid JSON found.\nLast content: {last_content[:300]}")
    return {
        "resolved"      : False,
        "error_message" : "LLM did not return JSON. Possible tool error or unexpected response.",
    }

def planner_agent(state: AgentState) -> AgentState:
    log_api_call("planner_agent_llm")
    """Converts the user prompt into a high-level engineering Plan."""
    user_prompt = state["messages"][0].content
    llm         = state["llm"]

    print("[Planner] Generating plan...")
    resp = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    print(f"[Planner] Plan generated: {resp}")
    return {"plan": resp, "llm": llm}


def architect_agent(state: AgentState) -> AgentState:
    log_api_call("architect_agent_llm")
    """Breaks the Plan into concrete implementation steps (TaskPlan)."""
    plan = state["plan"]
    llm  = state["llm"]

    print("[Architect] Generating task plan...")
    resp = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan))
    if not resp:
        raise ValueError("Architect agent failed to generate a task plan.")

    print(f"[Architect] {len(resp.implementation_steps)} steps created.")
    return {"task_plan": resp, "plan": plan, "llm": llm}


def coder_agent(state: AgentState) -> AgentState:
    log_api_call("coder_agent_llm")
    """
    Executes one implementation step at a time.
    - All steps done  → runner_agent
    - Step fails      → resolver_agent
    - More steps left → loops back to itself
    """
    llm          = state["llm"]
    coder_state  = state.get("coder_state")

    if not coder_state:
        coder_state = CoderState(
            task=state["task_plan"],
            current_step_idx=0,
            current_file_content=""
        )

    steps = coder_state.task.implementation_steps
    print(f"[Coder] Progress: step {coder_state.current_step_idx}/{len(steps)}")

    if coder_state.current_step_idx >= len(steps):
        print("[Coder] All steps done → runner")
        return {
            "coder_state"  : coder_state,
            "llm"          : llm,
            "error_message": None,
            "status"       : "CODING_DONE",
        }

    current_task  = steps[coder_state.current_step_idx]
    print(f"[Coder] Step {coder_state.current_step_idx}: {current_task.task_description}")

    system_prompt = (
        coder_system_prompt()
        + "\n\nAfter completing the task, return ONLY a valid JSON object with:\n"
        "- 'resolved': boolean (true if completed successfully)\n"
        "- 'error_message': string (error details if resolved is false, else null)\n"
        "Do NOT include any explanation outside the JSON. \n" +
        "Always give output in the following structure example:\n" +  
        """  {
                "resolved": true,
                "error_message": null
            }"""
        + "\nIMPORTANT:\n- Use the provided tools to interact with the filesystem and run commands in the sandbox.\n- Do not attempt to implement the next task until the current one is fully resolved.\n"
    )
    user_prompt = (
        f"Implement this task step-by-step:\n\n{current_task}\n\n"
        "Use the provided tools to write files and run commands as needed."
    )

    agent      = _make_react_agent(llm, _get_tools(state.get("project_id", "default_project")))
    raw_result = _invoke_react_agent(agent, [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])
    _log_tool_calls(raw_result, f"coder/step-{coder_state.current_step_idx}")

    result = _extract_json_from_agent_result(raw_result)

    MAX_CODER_STEP_RETRIES = 2
    coder_step_retries = state.get("coder_step_retries", 0)

    if (not result.get("resolved") and
            "LLM did not return JSON" in result.get("error_message", "")):
        had_tool_calls = any(
            getattr(m, "tool_calls", None)
            for m in raw_result.get("messages", [])
        )
        if had_tool_calls:
            # LLM ran tools but forgot the JSON wrapper — work was done, assume success.
            print(f"[Coder] Step {coder_state.current_step_idx}: LLM made tool calls "
                  "but omitted JSON — assuming step succeeded.")
            result = {"resolved": True, "error_message": None}
        elif coder_step_retries < MAX_CODER_STEP_RETRIES:
            # LLM returned nothing at all — retry the same step without advancing.
            print(f"[Coder] Step {coder_state.current_step_idx}: LLM returned nothing "
                  f"(retry {coder_step_retries + 1}/{MAX_CODER_STEP_RETRIES}).")
            return {
                "coder_state"      : coder_state,   # keep same step index
                "llm"              : llm,
                "error_message"    : "CODER_STEP_RETRY",
                "current_task"     : current_task,
                "status"           : None,
                "coder_step_retries": coder_step_retries + 1,
            }
        else:
            # Exhausted per-step retries — give up on this step and move on.
            print(f"[Coder] Step {coder_state.current_step_idx}: LLM returned nothing "
                  "after max retries — skipping step (runner will catch failures).")
            result = {"resolved": True, "error_message": None}

    print(result)
    new_coder_state = CoderState(
        task=coder_state.task,
        current_step_idx=coder_state.current_step_idx + 1,
        current_file_content=coder_state.current_file_content,
    )

    if result.get("resolved") == True:
        print(f"[Coder] Step {coder_state.current_step_idx} completed successfully.")
        return {
            "coder_state"      : new_coder_state,
            "llm"              : llm,
            "error_message"    : None,
            "current_task"     : current_task,
            "status"           : None,
            "coder_step_retries": 0,   # reset per-step retry counter on success
        }
    else:
        error = result.get("error_message", "Unknown error during coding step.")
        print(f"[Coder] Step {coder_state.current_step_idx} {error}")
        return {
            "coder_state"      : new_coder_state,
            "llm"              : llm,
            "error_message"    : error,
            "current_task"     : current_task,
            "status"           : None,
            "coder_step_retries": 0,
        }


def runner_agent(state: AgentState) -> AgentState:
    log_api_call("runner_agent_llm")
    """
    Runs the built application and checks for runtime errors.
    - Clean run  → END
    - Errors     → resolver_agent
    """
    llm         = state["llm"]
    run_retries = state.get("run_retries", 0)
    print(f"[Runner] Attempt {run_retries + 1}/{MAX_RUN_RETRIES} ...")

    system_prompt = """You are a QA engineer verifying a web application works.

The project uses Next.js for the frontend (always) and optionally Node.js/Express for the backend.

Steps to follow IN ORDER:
1. List files in the work directory to understand the project layout.
2. Find and read every package.json (root, frontend/, backend/) to identify start commands.
3. Install dependencies for each package.json found:
   - Run `npm install` in every directory that has a package.json.
4. Start the servers:
   - Next.js frontend: run `npm run dev` in the directory containing the Next.js package.json.
     Confirm the dev script is `next dev -H 0.0.0.0 -p 3000` before running it.
   - Node.js backend (if it exists): run `npm run dev` or `node server.js` in the backend directory.
5. Run `sleep 30` to wait for servers to fully start AND for Next.js to compile all pages.
6. Check ports in this exact priority order:
   - 3000  → Next.js frontend (always check this first)
   - 5000  → Node.js/Express backend
   - 8000  → alternative backend port
   - 8080  → alternative backend port
7. Call get_PID_of_process_running_on_port for each port until you find one with a process.
8. The port where the FRONTEND (Next.js) is running is the one to return.
   - If Next.js is on port 3000 → proceed to step 9.
   - Backend running on 5000 does NOT count as the frontend — keep looking.
9. Sanity curl:
   run_cmd("curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/")
   - 200 → great, return success=true, port=3000.
   - 404 → Next.js is running but returning 404. This is almost ALWAYS caused by a
     COMPILATION ERROR in one of the source files (NOT necessarily pages/index.js).
     Follow the 404 diagnosis procedure below BEFORE giving up.

404 DIAGNOSIS PROCEDURE (run these checks in order):
  A. Check pages/_app.js and styles/globals.css:
       read_file("pages/_app.js")
       It MUST import styles/globals.css. The correct content is:
         import '../styles/globals.css';
         export default function App({ Component, pageProps }) {
           return <Component {...pageProps} />;
         }
       If styles/globals.css does not exist, CREATE it (do NOT remove the import):
         *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
         body { font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.6; }

  B. Check pages/index.js exists and has a default export:
       read_file("pages/index.js")
       If it is missing or has no `export default function`, write:
         export default function Home() { return <div>Loading…</div>; }

  C. Check pages/_document.js for syntax errors:
       read_file("pages/_document.js")
       It must import from 'next/document' (NOT 'next/head').
       Fix any obvious syntax errors.

  D. After fixing any of the above, wait 10 s for Next.js hot-reload:
       run_cmd("sleep 10")
     Then curl again:
       run_cmd("curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/")
     - 200 → fixed! Return success=true, port=3000.
     - Still 404 → continue to step E (do NOT give up here).

  E. Scan pages/index.js for bad import paths FIRST (most common cause of 404):
     run_cmd("grep -n 'src/app\\|app/page\\|src/components\\|src/' pages/index.js")
     If ANY line is returned → those imports are WRONG. This project has NO src/ and NO app/ directory.
     Fix procedure:
       1. Read pages/index.js fully: run_cmd("cat pages/index.js")
       2. For each bad import like `import Hero from '../src/app/page'`:
            - The correct path is ALWAYS `../components/<ComponentName>`
            - Check if components/<ComponentName>.js exists: run_cmd("cat components/Hero.js")
            - If missing → create it with a real implementation (not just a stub)
            - Rewrite the import: import Hero from '../components/Hero'
       3. Rewrite pages/index.js completely with all imports corrected.
       4. Wait 15 s, then curl again. If 200 → done.

     After fixing bad paths, check each remaining import for missing component files:
       - read_file("components/ComponentName.js")
       - If missing → create it.
       - If it exists, scan for syntax errors or undefined exports.
     Repeat until curl returns 200 or all components are verified clean.

CRITICAL:
- Do NOT kill any running process. Servers must stay alive for the preview URL to work.
- `npm run dev` runs in the background automatically — never wait for it to exit.
- If NO process is found after checking all ports, report success=false with a clear error.
- The #1 cause of 404 is bad import paths (src/app/, app/) — ALWAYS grep for these first.

Return ONLY valid JSON:
{"success": boolean, "error_message": string or null, "run_command": string, "port": int}
No text outside the JSON."""

    # Runner can CHECK ports but must NOT kill — server must stay alive for preview
    runner_tools = get_sandbox(state.get("project_id", "default_project")).get_tools(
        include_process_tools=True, include_kill=False
    )

    # ── Token optimisation: trimmed history + recursion cap ────────
    agent  = _make_react_agent(llm, runner_tools)
    result = _invoke_react_agent(agent, [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Install dependencies, start the dev server, and confirm it is running. Leave the server running."},
    ])
    _log_tool_calls(result, "runner")

    parsed = _extract_json_from_agent_result(result)
    print(f"[Runner] Result: {parsed}")

    if parsed.get("success"):
        port = parsed.get("port")
        print(f"[Runner] Application ran successfully on port {port}!")
        return {
            "llm"          : llm,
            "status"       : "ALL_DONE",
            "error_message": None,
            "run_retries"  : run_retries,
            "server_port"  : port,
        }
    else:
        error = parsed.get("error_message", "Unknown runtime error.")
        print(f"[Runner] Runtime error: {error}")
        return {
            "llm"          : llm,
            "status"       : "RUNTIME_ERROR",
            "error_message": error,
            "current_task" : None,
            "run_retries"  : run_retries + 1,
        }


def resolver_agent(state: AgentState) -> AgentState:
    log_api_call("resolver_agent_llm")
    """
    Fixes an error from a coding step or a runtime error.
    - Fixed   → back to coder or runner
    - Failed  → retry up to MAX_RESOLVE_RETRIES, then give up
    """
    llm             = state["llm"]
    error_message   = state.get("error_message", "")
    current_task    = state.get("current_task")
    resolve_retries = state.get("resolve_retries", 0)
    coder_state     = state.get("coder_state")

    if not error_message:
        print("[Resolver] No error to resolve.")
        return {"status": "NO_ERROR", "llm": llm, "resolve_retries": 0}

    print(f"[Resolver] Attempt {resolve_retries + 1}/{MAX_RESOLVE_RETRIES}: {error_message[:200]}")

    task_description = current_task.task_description if current_task else "Runtime / startup error"

    system_prompt = (
        resolver_prompt(task_description, error_message)
        + "\n\nSteps:\n"
        "1. Read relevant source files.\n"
        "2. Identify the root cause.\n"
        "3. Write the corrected files.\n"
        "4. Run the relevant part to verify the fix.\n"
        "5. Kill any started processes.\n\n"
        "Return ONLY valid JSON:\n"
        "- 'resolved': boolean\n"
        "- 'error_message': string or null\n"
        "Do NOT include any explanation outside the JSON."
     +   
    """
        IMPORTANT RULES to minimise token usage:
        - NEVER read or list node_modules/, .git/, __pycache__/, dist/, build/, .next/
        - Read only the specific files relevant to the current task
        - Do NOT explore the entire project structure — read only what you need
        - When running npm/pip commands, ignore verbose install  """
    )
    user_prompt = (
        f"Task context: {task_description}\n\n"
        f"Error:\n{error_message}\n\n"
        "Analyse, fix, and verify."
    )

    resolver_tools = _get_tools(state.get("project_id", "default_project"), include_process_tools=True)

    # ── Token optimisation: trimmed history + recursion cap ────────
    agent  = _make_react_agent(llm, resolver_tools)
    result = _invoke_react_agent(agent, [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])
    _log_tool_calls(result, f"resolver/attempt-{resolve_retries+1}")

    parsed = _extract_json_from_agent_result(result)
    print(f"[Resolver] Output: {parsed}")

    # If the resolver's OWN LLM also failed to return JSON, retrying is pointless —
    # give up immediately rather than burning all 8 retry slots.
    # Use `or ""` because error_message may explicitly be None (key present, value None).
    if "LLM did not return JSON" in (parsed.get("error_message") or ""):
        print("[Resolver] Resolver LLM also failed to return JSON — giving up immediately.")
        return {
            "llm"            : llm,
            "status"         : "RESOLVE_FAILED",
            "error_message"  : error_message,   # restore original error, not the meta-error
            "resolve_retries": MAX_RESOLVE_RETRIES,
        }

    if parsed.get("resolved"):
        print("[Resolver] Error resolved.")
        next_status = (
            "RESOLVED_CODING"
            if coder_state.current_step_idx < len(coder_state.task.implementation_steps)
            else "RESOLVED_RUNTIME"
        )
        return {
            "llm"           : llm,
            "status"        : next_status,
            "error_message" : None,
            "resolve_retries": 0,
            "coder_state"   : coder_state,
        }
    else:
        remaining_error  = parsed.get("error_message", "Unknown error")
        resolve_retries += 1
        print(f"[Resolver] Still unresolved (attempt {resolve_retries}): {remaining_error}")

        if resolve_retries >= MAX_RESOLVE_RETRIES:
            print(f"[Resolver] Max retries reached. Giving up.")
            return {
                "llm"           : llm,
                "status"        : "RESOLVE_FAILED",
                "error_message" : remaining_error,
                "resolve_retries": resolve_retries,
            }

        return {
            "llm"           : llm,
            "status"        : "RETRY_RESOLVE",
            "error_message" : remaining_error,
            "resolve_retries": resolve_retries,
        }



def _coder_router(state: AgentState) -> str:
    status = state.get("status")
    error  = state.get("error_message")
    if status == "CODING_DONE":
        print("[Router:coder] → runner");  return "run"
    if error == "CODER_STEP_RETRY":
        print("[Router:coder] → coder (retry same step — LLM returned nothing)"); return "next"
    if error:
        print(f"[Router:coder] → resolver ({error[:80]})"); return "error"
    print("[Router:coder] → coder (next step)");            return "next"


def _runner_router(state: AgentState) -> str:
    status      = state.get("status")
    run_retries = state.get("run_retries", 0)
    if status == "ALL_DONE":
        print("[Router:runner] → END");                   return "done"
    if run_retries >= MAX_RUN_RETRIES:
        print("[Router:runner] → END (max retries)");     return "done"
    print("[Router:runner] → resolver (runtime error)");     return "error"


def _resolver_router(state: AgentState) -> str:
    status = state.get("status", "")
    if status == "RESOLVED_CODING":
        print("[Router:resolver] → coder");    return "back_to_coder"
    if status == "RESOLVED_RUNTIME":
        print("[Router:resolver] → runner");   return "back_to_runner"
    if status == "NO_ERROR":
        print("[Router:resolver] → coder");    return "back_to_coder"
    if status == "RETRY_RESOLVE":
        print("[Router:resolver] → resolver"); return "retry"
    print("[Router:resolver] → END");       return "give_up"


graph = StateGraph(AgentState)

graph.add_node("plan",     planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder",    coder_agent)
graph.add_node("runner",   runner_agent)
graph.add_node("resolver", resolver_agent)

graph.set_entry_point("plan")
graph.add_edge("plan",      "architect")
graph.add_edge("architect", "coder")

graph.add_conditional_edges("coder",    _coder_router,    {"run": "runner", "error": "resolver", "next": "coder"})
graph.add_conditional_edges("runner",   _runner_router,   {"done": END, "error": "resolver"})
graph.add_conditional_edges("resolver", _resolver_router, {
    "back_to_coder" : "coder",
    "back_to_runner": "runner",
    "retry"         : "resolver",
    "give_up"       : END,
})

memory = MemorySaver()
agent  = graph.compile(checkpointer=memory)


def run_agent(
    user_prompt : str,
    project_id  : str,
    use_ollama  : bool = False,
    use_gemini  : bool = False,
    use_qwen    : bool = False,
    use_groq    : bool = False,
    use_claude    : bool = False,
    on_phase    = None,   # optional callback(node_name: str) called after each graph node
) -> dict:

    if use_ollama:
        llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.2,
                         base_url="http://127.0.0.1:11434")
        print("Using Ollama (qwen2.5-coder:7b)")
    elif use_gemini:
        llm = ChatGemini(model="gemini-2.5-pro", temperature=0.2)
        print("Using Gemini (gemini-2.5-pro)")
    elif use_qwen:
        llm = ChatQwen(model="qwen3-235b-a22b", temperature=0.2)
        print("Using Qwen (qwen3-max)")
    elif use_groq:
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
        print("Using Groq (gpt-oss-120b)")
    elif use_claude:
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", 
                                temperature=0.2, 
                                api_key=os.getenv("CLAUDE_API_KEY"))
        print("Using Claude (claude-haiku-4-5-20251001)")
    

    print(llm)

    crawl_results, has_crawl_results = crawl_website_for_clone(user_prompt)
    if has_crawl_results:
        content = f"Crawl results:\n{crawl_results}\n\n"
    else:
        content = user_prompt

    initial_state = AgentState(
        messages           = [HumanMessage(content=content)],
        project_id         = project_id,
        llm                = llm,
        run_retries        = 0,
        resolve_retries    = 0,
        coder_step_retries = 0,
        error_message      = None,
        status             = None,
    )

    config = {"configurable": {"thread_id": project_id, "project_id": project_id}}

    print(f"\n{'='*60}")
    print(f"Starting agent | Project: {project_id}")
    print(f"Prompt: {user_prompt}")
    print(f"{'='*60}\n")

    # stream() yields {node_name: partial_state} after each node finishes.
    # This lets us report live progress via on_phase without blocking.
    for chunk in agent.stream(initial_state, config=config):
        node_name   = next(iter(chunk))
        node_output = chunk[node_name]
        print(f"[Graph] Node finished: {node_name}")

        if on_phase:
            try:
                # Build a rich info dict so the frontend can show granular progress
                info = {}
                if node_name == "coder":
                    coder_state  = node_output.get("coder_state")
                    current_task = node_output.get("current_task")
                    if coder_state:
                        info["step"]        = coder_state.current_step_idx  # steps done so far
                        info["total_steps"] = len(coder_state.task.implementation_steps)
                    if current_task:
                        info["step_description"] = current_task.task_description[:100]
                on_phase(node_name, info)
            except Exception:
                pass

    # Retrieve the full accumulated state from the checkpointer
    snapshot     = agent.get_state(config=config)
    result       = dict(snapshot.values)

    final_status = result.get("status", "UNKNOWN")
    print(f"\n{'='*60}")
    print(f"Agent finished | Status: {final_status}")
    if result.get("error_message"):
        print(f"Final error: {result['error_message']}")
    print(f"{'='*60}\n")

    return result

def run_edit_agent(
    edit_prompt : str,
    project_id  : str,
    use_ollama  : bool = False,
    use_gemini  : bool = False,
    use_qwen    : bool = False,
    use_groq    : bool = False,
    on_phase    = None,
) -> dict:
    """
    Run a targeted edit on an already-built project without replanning.
    Reuses the existing E2B sandbox (dev server may already be running).
    """
    if use_ollama:
        llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.2,
                         base_url="http://127.0.0.1:11434")
        print("Using Ollama (qwen2.5-coder:7b)")
    elif use_gemini:
        llm = ChatGemini(model="gemini-2.5-pro", temperature=0.2)
        print("Using Gemini (gemini-2.5-pro)")
    elif use_qwen:
        llm = ChatQwen(model="qwen3-235b-a22b", temperature=0.2)
        print("Using Qwen (qwen3-max)")
    elif use_groq:
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
        print("Using Groq (gpt-oss-120b)")
    else:
        llm = ChatAnthropic(model="gemini-2.5-pro", temperature=0.2)
        print("Using default: Gemini (gemini-2.5-pro)")

    if on_phase:
        on_phase("editor", {})

    system_prompt = """You are an expert editor for an existing Next.js web application already running in an E2B sandbox.

The dev server may or may not be running on port 3000. Check first, restart only if needed.

Steps to follow:
1. Check server: run_cmd("curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/")
   - If 200 → server is alive, skip to step 2.
   - If NOT 200 → find the Next.js package.json via list_files(), then run:
       run_cmd("npm run dev &")
     Wait 15s: run_cmd("sleep 15")
     Then curl again to confirm it started.
2. list_files() to understand the current project structure.
3. read_file() ONLY on files that are relevant to the edit request — do NOT read everything.
4. Make targeted, minimal changes with write_file(). Do NOT rewrite unrelated files.
5. Keep the existing Next.js Pages Router structure: pages/, components/, styles/.
6. After writing changed files: run_cmd("sleep 10") to let Next.js hot-reload.
7. Verify: run_cmd("curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/")
   - 200 → return {"resolved": true, "error_message": null, "port": 3000}
   - Not 200 → diagnose, fix, wait 10s, curl again. Retry up to 3 times.
   - If still broken after 3 retries → return {"resolved": false, "error_message": "<what went wrong>", "port": 3000}

IMPORTANT RULES:
- NEVER kill the running dev server (do not use kill_process).
- NEVER modify package.json unless a brand-new npm package is absolutely required for the edit.
  If you do add a package, run `npm install` before using it.
- NEVER import a plain .css file (e.g. globals.css) in any file except pages/_app.js.
  For component-level styles, use CSS Modules: styles/Foo.module.css imported as
  `import styles from '../styles/Foo.module.css'`.
- NEVER read node_modules/, .git/, .next/, dist/, or build/ directories.
- Make only the changes needed for the edit request — do not refactor or improve unrelated code.

Return ONLY valid JSON (no text outside it):
{"resolved": true/false, "error_message": null or "description of problem", "port": 3000}"""

    user_prompt = (
        f"Edit request: {edit_prompt}\n\n"
        "Make this specific change to the existing project. "
        "Do not rebuild the whole app — only modify what is needed."
    )

    tools = get_sandbox(project_id).get_tools(
        include_process_tools=True, include_kill=False
    )
    agent  = _make_react_agent(llm, tools)
    result = _invoke_react_agent(agent, [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])
    log_api_call("edit_agent_llm")
    _log_tool_calls(result, "editor")

    if on_phase:
        on_phase("done", {})

    parsed = _extract_json_from_agent_result(result)
    status = "ALL_DONE" if parsed.get("resolved") else "RESOLVE_FAILED"
    return {
        "status"       : status,
        "error_message": parsed.get("error_message"),
        "server_port"  : parsed.get("port", 3000),
    }


def is_clone_request(user_prompt: str) -> bool:
    indicators = ["clone", "copy", "replicate", "similar to", "like"]
    return any(indicator in user_prompt.lower() for indicator in indicators)


def crawl_website_for_clone(user_prompt: str) -> list:
    if is_clone_request(user_prompt):
        crawl_results = asyncio.run(crawl_website(user_prompt))
        print(f"Crawl results obtained: {crawl_results}")
        return crawl_results, True
    else:
        return user_prompt, False
    
    
if __name__ == "__main__":
    app_1_id = str(uuid.uuid4())   
    prompt = "Can you make clone of https://aceengineeringworks.co.uk/"  
    
    run_agent(
        user_prompt=prompt,
        project_id=app_1_id,
        use_gemini=True,
    )
    
    
    


