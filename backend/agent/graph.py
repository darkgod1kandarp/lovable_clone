import json
import uuid
import os 

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.constants import END
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI as ChatGemini
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Importing the prompt template for the planner agent
from prompts import planner_prompt, architect_prompt, coder_system_prompt, resolver_prompt

# Importing the Plan schema which defines the structure of the engineering plan
from states import Plan, TaskPlan, CoderState, AgentState, ResolverState

# Importing the tools for file operations and command execution
from tools import *

# Loading the Environment Variables
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_RESOLVE_RETRIES = 3   # Max times resolver retries the same error
MAX_RUN_RETRIES = 3       # Max times runner+resolver loop retries

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json_from_agent_result(result: dict) -> dict:
    """
    Robustly extract a JSON object from a react_agent result.

    Handles:
    - Plain string content:  "```json\\n{...}\\n```"
    - Gemini list content:   [{'type': 'text', 'text': '```json\\n{...}```', ...}]
    - Trailing text after JSON
    - Missing JSON entirely (tool errors, apologies, etc.)

    Scans all messages from last to first so a tool-error explanation
    in the final message doesn't hide valid JSON in an earlier one.
    """

    def _content_to_str(content) -> str:
        """Normalize any content format to a plain string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Gemini returns [{'type': 'text', 'text': '...'}, ...]
            parts = []
            for block in content:
                if isinstance(block, dict):
                    # grab 'text' key if present, else skip
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        # Fallback — stringify whatever it is
        return str(content)

    def _try_parse(raw: str) -> dict | None:
        """Try to extract and parse a JSON object from a raw string."""
        cleaned = raw.strip()

        # Strip markdown fences  ```json ... ``` or ``` ... ```
        if "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 2:
                inner = parts[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                cleaned = inner.strip()

        # Find outermost { ... }
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            return None

        try:
            parsed = json.loads(cleaned[start:end])
            # Only accept dicts that contain at least one expected key
            if isinstance(parsed, dict) and any(
                k in parsed for k in ("resolved", "success", "error_message")
            ):
                return parsed
        except json.JSONDecodeError:
            pass

        return None

    # ── Scan messages newest → oldest ────────────────────────────────────────
    messages = result.get("messages", [])
    for msg in reversed(messages):
        raw_content = getattr(msg, "content", None)
        if raw_content is None:
            continue

        text = _content_to_str(raw_content)
        parsed = _try_parse(text)
        if parsed is not None:
            return parsed

    # Nothing found anywhere
    last_content = ""
    if messages:
        last_content = _content_to_str(getattr(messages[-1], "content", ""))
    print(f"[JSON Parse Error] No valid JSON found in any message.\nLast content: {last_content[:300]}")
    return {"resolved": False, "error_message": "LLM did not return JSON. Possible tool error or unexpected response."}


def _make_coder_tools(extra=None):
    """Return the standard set of tools for coding/resolving agents."""
    base = [write_file, read_file, get_current_directory, list_files, run_cmd]
    if extra:
        base += extra
    return base


# ── Agent Nodes ───────────────────────────────────────────────────────────────

def planner_agent(state: AgentState) -> AgentState:
    """Converts the user prompt into a high-level engineering Plan."""
    user_prompt = state["messages"][0].content
    llm = state["llm"]

    print("[Planner] Generating plan...")
    plan_prompt_text = planner_prompt(user_prompt)
    resp = llm.with_structured_output(Plan).invoke(plan_prompt_text)

    print(f"[Planner] Plan generated: {resp}")
    return {"plan": resp, "llm": llm}


def architect_agent(state: AgentState) -> AgentState:
    """Breaks the Plan into concrete implementation steps (TaskPlan)."""
    plan: Plan = state["plan"]
    llm = state["llm"]

    print("[Architect] Generating task plan...")
    arch_prompt_text = architect_prompt(plan)
    resp = llm.with_structured_output(TaskPlan).invoke(arch_prompt_text)

    if not resp:
        raise ValueError("Architect agent failed to generate a task plan.")

    print(f"[Architect] {len(resp.implementation_steps)} steps created.")
    return {"task_plan": resp, "plan": plan, "llm": llm}


def coder_agent(state: AgentState) -> AgentState:
    """
    Executes one implementation step at a time.
    - If all steps are done → transitions to runner_agent.
    - If step fails → transitions to resolver_agent.
    - Otherwise → loops back to itself for the next step.
    """
    llm = state["llm"]
    coder_state: CoderState = state.get("coder_state")

    # ── Initialize coder state on first entry ──────────────────────────────
    if not coder_state:
        coder_state = CoderState(
            task=state["task_plan"],
            current_step_idx=0,
            current_file_content=""
        )

    steps = coder_state.task.implementation_steps
    print(f"[Coder] Progress: step {coder_state.current_step_idx}/{len(steps)}")

    # ── All coding steps are complete → hand off to runner ─────────────────
    if coder_state.current_step_idx >= len(steps):
        print("[Coder] All implementation steps done. Handing off to runner.")
        print(f"[Coder] Final state: {coder_state}")
        return_data =  {
            "coder_state": coder_state,
            "llm": llm,
            "error_message": None,
            "status": "CODING_DONE",
        }
        print(f"[Coder] Returning: {return_data}")
        return return_data

    # ── Execute current step ───────────────────────────────────────────────
    current_task = steps[coder_state.current_step_idx]
    print(f"[Coder] Step {coder_state.current_step_idx}: {current_task.task_description}")

    system_prompt = (
        coder_system_prompt()
        + "\n\nAfter completing the task, return ONLY a valid JSON object with:\n"
        "- 'resolved': boolean (true if completed successfully)\n"
        "- 'error_message': string (error details if resolved is false, else null)\n"
        "Do NOT include any explanation outside the JSON."
    )
    user_prompt = (
        f"Implement this task step-by-step:\n\n{current_task}\n\n"
        "Use the provided tools to write files and run commands as needed."
    )

    react_agent = create_react_agent(llm, _make_coder_tools())
    result = react_agent.invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    })

    parsed = _extract_json_from_agent_result(result)

    # Advance step index regardless of success/failure
    new_coder_state = CoderState(
        task=coder_state.task,
        current_step_idx=coder_state.current_step_idx + 1,
        current_file_content=coder_state.current_file_content,
    )

    if parsed.get("resolved"):
        print(f"[Coder] Step {coder_state.current_step_idx} completed successfully.")
        return {
            "coder_state": new_coder_state,
            "llm": llm,
            "error_message": None,
            "current_task": current_task,
            "status": None,
        }
    else:
        error = parsed.get("error_message", "Unknown error during coding step.")
        print(f"[Coder] Step {coder_state.current_step_idx} FAILED: {error}")
        return {
            "coder_state": new_coder_state,
            "llm": llm,
            "error_message": error,
            "current_task": current_task,
            "status": None,
        }


def runner_agent(state: AgentState) -> AgentState:
    """
    Runs the built application and checks for runtime errors.
    - Installs dependencies, starts the app, waits, then inspects output/logs.
    - If runtime errors detected → transitions to resolver_agent.
    - If clean → transitions to END.
    """
    llm = state["llm"]
    run_retries = state.get("run_retries", 0)

    print(f"[Runner] Attempting to run the application (attempt {run_retries + 1}/{MAX_RUN_RETRIES})...")

    system_prompt =  """You are a QA engineer verifying a web application works.

            Steps to follow IN ORDER:

            1. List files to understand the project structure.
            2. Read package.json (or requirements.txt) to find the start command.
            3. Install dependencies:
            - For Node.js: run `npm install` (this may take 2-5 minutes, be patient)
            - For Python: run `pip install -r requirements.txt`
            4. Start the app using run_cmd — dev servers (npm run dev / npm start) run in the
            background automatically and return immediately with a PID. Do NOT wait for them.
            5. Wait 3-5 seconds after starting, then check if the process is still running
            using get_PID_of_process_running_on_port on port 3000 (or the app's port).
            6. If the port has an active process → the app is running successfully.
            7. Kill the process using kill_process before finishing.

            IMPORTANT:
            - npm run dev / npm start WILL NOT EXIT on their own — they are background servers.
            The tool handles this automatically. Just call run_cmd and move on.
            - If the port check confirms a running process, report success=true.
            - Do not try to read stdout from the dev server — it runs forever.

            Return ONLY a valid JSON object:
            {
            "success": boolean,
            "error_message": string or null,
            "run_command": string
            }
        No text outside the JSON."""

    runner_tools = [
        write_file, read_file, get_current_directory, list_files, run_cmd,
        kill_process, get_PID_of_process_running_on_port
    ]
    run_agent = create_react_agent(llm, runner_tools)

    result = run_agent.invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Install dependencies, start the application, verify it runs, then kill the process."},
        ]
    })

    parsed = _extract_json_from_agent_result(result)
    print(f"[Runner] Result: {parsed}")

    if parsed.get("success"):
        
        # Still task is remaining to complete do it 
    
        print("[Runner] ✅ Application ran successfully!")
        return {
            "llm": llm,
            "status": "ALL_DONE",
            "error_message": None,
            "run_retries": run_retries,
        }
    else:
        error = parsed.get("error_message", "Unknown runtime error.")
        print(f"[Runner] ❌ Runtime error detected: {error}")
        return {
            "llm": llm,
            "status": "RUNTIME_ERROR",
            "error_message": error,
            "current_task": None,   # No specific coding task — it's a runtime issue
            "run_retries": run_retries + 1,
        }


def resolver_agent(state: AgentState) -> AgentState:
    """
    Analyzes an error (either from a coding step or a runtime error),
    implements a fix, and verifies the fix works.
    - On success → routes back to coder (coding error) or runner (runtime error).
    - On failure → retries up to MAX_RESOLVE_RETRIES times, then gives up.
    """
    llm = state["llm"]
    error_message = state.get("error_message", "")
    current_task = state.get("current_task")
    resolve_retries = state.get("resolve_retries", 0)
    coder_state = state.get("coder_state")

    if not error_message:
        print("[Resolver] No error to resolve.")
        return {"status": "NO_ERROR", "llm": llm, "resolve_retries": 0}

    print(f"[Resolver] Resolving error (attempt {resolve_retries + 1}/{MAX_RESOLVE_RETRIES}): {error_message[:200]}")

    task_description = current_task.task_description if current_task else "Runtime / startup error"

    system_prompt = (
        resolver_prompt(task_description, error_message)
        + "\n\nSteps to follow:\n"
        "1. Read the relevant source files to understand the codebase.\n"
        "2. Identify the root cause of the error.\n"
        "3. Implement the fix by writing the corrected files.\n"
        "4. Run the relevant part of the code to verify the fix.\n"
        "5. Kill any started processes.\n\n"
        "Return ONLY a valid JSON object:\n"
        "- 'resolved': boolean (true if the error is fixed and verified)\n"
        "- 'error_message': string (remaining error details if resolved is false, else null)\n"
        "Do NOT include any explanation outside the JSON."
    )
    user_prompt = (
        f"Task context: {task_description}\n\n"
        f"Error:\n{error_message}\n\n"
        "Analyze, fix, and verify."
    )

    resolver_tools = _make_coder_tools(extra=[kill_process, get_PID_of_process_running_on_port])
    res_agent = create_react_agent(llm, resolver_tools)
    result = res_agent.invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    })

    parsed = _extract_json_from_agent_result(result)
    print(f"[Resolver] Output: {parsed}")

    if parsed.get("resolved"):
        print("[Resolver] ✅ Error resolved successfully.")
        # Determine where to go back: if current_task exists, return to coder; else to runner
        if coder_state.current_step_idx < len(coder_state.task.implementation_steps):
            next_status = "RESOLVED_CODING"
        else:
            next_status = "RESOLVED_RUNTIME"
            
        return {
            "llm": llm,
            "status": next_status,
            "error_message": None,
            "resolve_retries": 0,
            "coder_state": coder_state,
        }
    else:
        remaining_error = parsed.get("error_message", "Unknown error")
        resolve_retries += 1
        print(f"[Resolver] ❌ Still unresolved after attempt {resolve_retries}. Error: {remaining_error}")

        if resolve_retries >= MAX_RESOLVE_RETRIES:
            print(f"[Resolver] 🚨 Max retries ({MAX_RESOLVE_RETRIES}) reached. Giving up.")
            return {
                "llm": llm,
                "status": "RESOLVE_FAILED",
                "error_message": remaining_error,
                "resolve_retries": resolve_retries,
            }

        return {
            "llm": llm,
            "status": "RETRY_RESOLVE",
            "error_message": remaining_error,
            "resolve_retries": resolve_retries,
        }


# ── Routing Functions ─────────────────────────────────────────────────────────

def _coder_router(state: AgentState) -> str:
    """
    After coder_agent runs:
    - CODING_DONE   → all steps finished, go run the app
    - error_message → a step failed, go to resolver
    - otherwise     → more steps remain, loop back to coder
    """
    status = state.get("status")
    error = state.get("error_message")
    

    if status == "CODING_DONE":
        print("[Router:coder] → runner")
        return "run"

    if error:
        print(f"[Router:coder] → resolver (error: {error[:80]})")
        return "error"

    print("[Router:coder] → coder (next step)")
    return "next"


def _runner_router(state: AgentState) -> str:
    """
    After runner_agent runs:
    - ALL_DONE      → everything works, finish
    - RUNTIME_ERROR → runtime issues found, go to resolver
    - Too many retries → give up
    """
    status = state.get("status")
    run_retries = state.get("run_retries", 0)

    if status == "ALL_DONE":
        print("[Router:runner] → END ✅")
        return "done"

    if run_retries >= MAX_RUN_RETRIES:
        print(f"[Router:runner] → END (max run retries reached) 🚨")
        return "done"

    print("[Router:runner] → resolver (runtime error)")
    return "error"


def _resolver_router(state: AgentState) -> str:
    """
    After resolver_agent runs:
    - RESOLVED_CODING  → fixed a coding-step error, back to coder
    - RESOLVED_RUNTIME → fixed a runtime error, back to runner to verify
    - RETRY_RESOLVE    → resolver needs another attempt
    - RESOLVE_FAILED   → give up
    - NO_ERROR         → nothing to fix, back to coder
    """
    status = state.get("status", "")

    if status == "RESOLVED_CODING":
        print("[Router:resolver] → coder (coding error fixed)")
        return "back_to_coder"

    if status == "RESOLVED_RUNTIME":
        print("[Router:resolver] → runner (runtime error fixed, re-run to verify)")
        return "back_to_runner"

    if status == "NO_ERROR":
        print("[Router:resolver] → coder (no error)")
        return "back_to_coder"

    if status == "RETRY_RESOLVE":
        print("[Router:resolver] → resolver (retry)")
        return "retry"

    # RESOLVE_FAILED or anything else → terminate
    print("[Router:resolver] → END (unresolvable error) 🚨")
    return "give_up"


# ── Graph Definition ──────────────────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("plan", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)
graph.add_node("runner", runner_agent)
graph.add_node("resolver", resolver_agent)

graph.set_entry_point("plan")
graph.add_edge("plan", "architect")
graph.add_edge("architect", "coder")

graph.add_conditional_edges(
    "coder",
    _coder_router,
    {
        "run":   "runner",   # All steps done → run the app
        "error": "resolver", # Step failed → fix it
        "next":  "coder",    # More steps → continue
    },
)

graph.add_conditional_edges(
    "runner",
    _runner_router,
    {
        "done":  END,        # App works or max retries hit → finish
        "error": "resolver", # Runtime error → fix it
    },
)

graph.add_conditional_edges(
    "resolver",
    _resolver_router,
    {
        "back_to_coder":  "coder",    # Coding error fixed → resume coding
        "back_to_runner": "runner",   # Runtime error fixed → re-run to verify
        "retry":          "resolver", # Still failing → retry resolver
        "give_up":        END,        # Max retries → give up
    },
)

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_agent(
    user_prompt: str,
    project_id: str,
    use_ollama: bool = False,
    use_gemini: bool = False,
) -> dict:
    """
    Main entry point. Creates the LLM and kicks off the agent graph.

    Args:
        user_prompt: The user's natural language request.
        project_id:  A unique ID for this project (used for memory/checkpointing).
        use_ollama:  Use local Ollama model instead of Groq.
        use_gemini:  Use Google Gemini instead of Groq.

    Returns:
        Final AgentState dict.
    """
    if use_ollama:
        llm = ChatOllama(model="qwen2.5-coder:32b", temperature=0.2, num_ctx=8192)
        print("Using Ollama (qwen2.5-coder:32b)")
    elif use_gemini:
        llm = ChatGemini(model="gemini-2.5-pro", temperature=0.2)
        print("Using Gemini (gemini-2.5-pro)")
    else:
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
        print("Using Groq (gpt-oss-120b)")

    initial_state = AgentState(
        messages=[HumanMessage(content=user_prompt)],
        project_id=project_id,
        llm=llm,
        run_retries=0,
        resolve_retries=0,
        error_message=None,
        status=None,
    )

    print(f"\n{'='*60}")
    print(f"🚀 Starting agent | Project: {project_id}")
    print(f"📝 Prompt: {user_prompt}")
    print(f"{'='*60}\n")

    result = agent.invoke(
        initial_state,
        config={
            "configurable": {
                "thread_id": project_id,
                "project_id": project_id,
            }
        },
        
    )

    final_status = result.get("status", "UNKNOWN")
    print(f"\n{'='*60}")
    print(f"🏁 Agent finished | Status: {final_status}")
    if result.get("error_message"):
        print(f"⚠️  Final error: {result['error_message']}")
    print(f"{'='*60}\n")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app_1_id = str(uuid.uuid4())
    run_agent(
        "Build a React app that should exactly look like this https://aceengineeringworks.co.uk/.",
        app_1_id,
        use_gemini=True,
    )

    # ── Examples of follow-up iterations (same project_id = memory preserved) ──

    # Iterate on App 1 (like Lovable's iteration feature):
    # run_agent("Make the background of the app dark black.", app_1_id, use_gemini=True)

    # Create a second, independent app:
    # app_2_id = str(uuid.uuid4())
    # run_agent("Build a Flask calculator API.", app_2_id)