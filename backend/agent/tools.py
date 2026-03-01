"""
tools.py — LangChain tools backed by the E2B sandbox.

All file I/O and command execution now happens inside the secure E2B cloud
sandbox instead of on the host machine.  The public API (tool names, signatures)
is unchanged so graph.py needs no edits.
"""

from typing import Tuple, Optional

from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from agent.sandbox import get_sandbox, SandboxManager

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_project_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("project_id", "default_project")


def _sb(config: RunnableConfig) -> SandboxManager:
    """Look up (or lazily create) the sandbox for the current project."""
    return get_sandbox(_get_project_id(config))


# Commands that run forever — must be backgrounded so the agent doesn't hang
_BACKGROUND_COMMANDS = (
    "npm run dev",
    "npm start",
    "npm run start",
    "yarn dev",
    "yarn start",
    "pnpm dev",
    "pnpm start",
    "vite",
    "next dev",
    "python manage.py runserver",
    "flask run",
    "uvicorn",
    "fastapi dev",
    "nodemon",
)


def _is_background_command(cmd: str) -> bool:
    return any(cmd.strip().lower().startswith(c) for c in _BACKGROUND_COMMANDS)


def _smart_timeout(cmd: str, user_timeout: int) -> int:
    """Give slow commands (npm install, builds) an automatically extended timeout."""
    _SLOW = {
        "npm install": 300, "npm ci": 300,
        "yarn install": 300, "yarn": 300,
        "pnpm install": 300,
        "pip install": 180, "pip3 install": 180,
        "npm run build": 180, "vite build": 180,
        "next build": 300, "tsc": 120,
    }
    cmd_lower = cmd.strip().lower()
    for prefix, t in _SLOW.items():
        if cmd_lower.startswith(prefix):
            return max(user_timeout, t)
    return max(user_timeout, 60)


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def run_cmd(
    cmd: str,
    cwd: str = None,
    timeout: int = 60,
    config: RunnableConfig = None,
) -> Tuple[int, str, str]:
    """
    Run a shell command inside the secure E2B sandbox.

    Dev-server commands (npm run dev, npm start, vite, etc.) are automatically
    run in the background so the agent is not blocked.
    Install commands get extended timeouts automatically.

    Returns: (return_code, stdout, stderr)
    """
    sb = _sb(config)

    if _is_background_command(cmd):
        print(f"[run_cmd] Backgrounding in sandbox: {cmd}")
        return sb.run_background_cmd(cmd, cwd=cwd)

    effective_timeout = _smart_timeout(cmd, timeout)
    print(f"[run_cmd] Running in sandbox: {cmd}")
    return sb.run_cmd(cmd, cwd=cwd, timeout=effective_timeout)


@tool
def write_file(path: str, content: str, config: RunnableConfig) -> str:
    """Write content to a file inside the E2B sandbox project directory."""
    return _sb(config).write_file(path, content)


@tool
def read_file(path: str, config: RunnableConfig) -> str:
    """Read content from a file inside the E2B sandbox project directory."""
    return _sb(config).read_file(path)


@tool
def get_current_directory(config: RunnableConfig) -> str:
    """Return the current working directory path inside the sandbox."""
    sb = _sb(config)
    return sb.work_dir


@tool
def list_files(directory: str = ".", config: RunnableConfig = None) -> str:
    """
    List all files in the specified directory inside the E2B sandbox.
    node_modules, .git, __pycache__, dist, and build are excluded.
    """
    return _sb(config).list_files(directory)


@tool
def kill_process(procees_id: int, config: RunnableConfig = None) -> str:
    """Kill a process (by PID) running inside the E2B sandbox."""
    return _sb(config).kill_process(procees_id)


@tool
def get_PID_of_process_running_on_port(
    port: int,
    config: RunnableConfig = None,
) -> str:
    """
    Check whether a process is listening on *port* inside the E2B sandbox.
    Returns a non-empty string (truthy) if a process is found, empty string if not.
    """
    return _sb(config).get_pid_on_port(port)
