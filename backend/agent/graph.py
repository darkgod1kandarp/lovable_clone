import datetime
import json
import os
import uuid

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
from agent.sandbox import get_sandbox, cleanup_sandbox, get_preview_url
from agent.crawl_4ai import *
import json   

load_dotenv()

MAX_RESOLVE_RETRIES   = 3     # Max times resolver retries the same error
MAX_RUN_RETRIES       = 3     # Max times runner+resolver loop retries
MAX_TOOL_OUTPUT_CHARS = 2000  # Truncate tool outputs beyond this length
MAX_RECURSION_LIMIT   = 30    # Max tool-call rounds per react_agent invocation
MAX_TOKENS_IN_HISTORY = 4000  # Max tokens kept in message history per agent
API_CALL_LOG_PATH     = os.path.join(os.path.dirname(__file__), "api_call_log.txt")


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

    agent  = _make_react_agent(llm, _get_tools(state.get("project_id", "default_project")))
    result = _invoke_react_agent(agent, [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])

    result = _extract_json_from_agent_result(result)
    print(result)
    new_coder_state = CoderState(
        task=coder_state.task,
        current_step_idx=coder_state.current_step_idx + 1,
        current_file_content=coder_state.current_file_content,
    )

    if result.get("resolved") == True:
        print(f"[Coder] Step {coder_state.current_step_idx} completed successfully.")
        return {
            "coder_state"  : new_coder_state,
            "llm"          : llm,
            "error_message": None,
            "current_task" : current_task,
            "status"       : None,
        }
    else:
        error = result.get("error_message", "Unknown error during coding step.")
        print(f"[Coder] Step {coder_state.current_step_idx} {error}")
        return {
            "coder_state"  : new_coder_state,
            "llm"          : llm,
            "error_message": error,
            "current_task" : current_task,
            "status"       : None,
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

Steps to follow IN ORDER:
1. List files to understand the project structure.
2. Read package.json (or requirements.txt) to find the start command.
3. Install dependencies:
   - Node.js: run `npm install`
   - Python:  run `pip install -r requirements.txt`
4. Start the dev server using run_cmd — it runs in the background automatically.
5. Wait a few seconds, then use get_PID_of_process_running_on_port to confirm the server is up.
6. If a process is found on the port → success=true. Leave the server running.

CRITICAL:
- Do NOT kill the server process. It must stay running so users can access the preview URL.
- npm run dev / vite / npm start run in the background — do not wait for them to exit.
- If the port check confirms a running process, report success=true immediately.

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

    
    print(result, "RESULT")
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

    parsed = _extract_json_from_agent_result(result)
    print(f"[Resolver] Output: {parsed}")

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
) -> dict:

    if use_ollama:
        llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.2,
                         base_url="http://127.0.0.1:11434")
        print("Using Ollama (qwen2.5-coder:7b)")
    elif use_gemini:
        llm = ChatGemini(model="gemini-2.5-flash", temperature=0.2)
        print("Using Gemini (gemini-2.5-flash)")
    elif use_qwen:
        llm = ChatQwen(model="qwen3-max", temperature=0.2)
        print("Using Qwen (qwen3-max)")
    else:
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
        print("Using Groq (gpt-oss-120b)")


    crawl_results, has_crawl_results = crawl_website_for_clone(user_prompt) 
    if has_crawl_results:
        content = f"Crawl results:\n{crawl_results}\n\n"
    else:
        content = user_prompt
        
    initial_state = AgentState(
        messages        = [HumanMessage(content=content)],
        project_id      = project_id,
        llm             = llm,
        run_retries     = 0,
        resolve_retries = 0,
        error_message   = None,
        status          = None,
    )

    print(f"\n{'='*60}")
    print(f"Starting agent | Project: {project_id}")
    print(f"Prompt: {user_prompt}")
    print(f"{'='*60}\n")

    result = agent.invoke(
        initial_state,
        config={"configurable": {"thread_id": project_id, "project_id": project_id}},
    )

    final_status = result.get("status", "UNKNOWN")
    print(f"\n{'='*60}")
    print(f"Agent finished | Status: {final_status}")
    if result.get("error_message"):
        print(f"Final error: {result['error_message']}")
    print(f"{'='*60}\n")

    return result

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