import pathlib
import subprocess
from typing import Tuple


from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig


BASE_DIR = pathlib.Path.cwd() / "agent_workspace"

import subprocess
import time
from pathlib import Path

# Commands that run forever and must be backgrounded
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
    cmd_lower = cmd.strip().lower()
    return any(cmd_lower.startswith(c) for c in _BACKGROUND_COMMANDS)

def _smart_timeout(cmd: str, user_timeout: int) -> int:
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


@tool
def run_cmd(
    cmd: str,
    cwd: str = None,
    timeout: int = 60,
    config: RunnableConfig = None,
) -> Tuple[int, str, str]:
    """
    Run a shell command. Dev server commands (npm run dev, npm start, etc.)
    are automatically run in the background and their PID is returned.
    Install commands automatically get extended timeouts.

    Returns: (return_code, stdout, stderr)
    """
    cwd_dir = safe_path_for_project(cwd, config) if cwd else get_project_root(config)

    # ── Auto-background dev servers ───────────────────────────────────────
    if _is_background_command(cmd):
        print(f"[run_cmd] 🔄 Backgrounding dev server: {cmd}")
        try:
            # Start process detached, don't wait for it
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(cwd_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Wait 3 seconds to capture any immediate startup errors
            time.sleep(3)
            poll = proc.poll()

            if poll is not None:
                # Process already exited — something went wrong
                out, err = proc.communicate(timeout=5)
                return poll, out, f"Process exited early.\n{err}"
            else:
                # Still running — success, return the PID
                return 0, f"Dev server started in background. PID={proc.pid}", ""
        except Exception as e:
            return -1, "", f"{type(e).__name__}: {e}"

    # ── Normal command with smart timeout ─────────────────────────────────
    effective_timeout = _smart_timeout(cmd, timeout)
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd_dir),
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
        return res.returncode, res.stdout, res.stderr

    except subprocess.TimeoutExpired:
        msg = (
            f"Command timed out after {effective_timeout}s: {cmd}\n"
            "The process may still be running. Use get_PID_of_process_running_on_port or kill_process."
        )
        return -1, "", msg

    except Exception as e:
        return -1, "", f"{type(e).__name__}: {e}"

def get_project_root(config: RunnableConfig) -> pathlib.Path:
    # Extract project_id from LangGraph config
    project_id = config.get("configurable", {}).get("project_id", "default_project")
    project_root = BASE_DIR / project_id
    project_root.mkdir(parents=True, exist_ok=True)
    return project_root

def safe_path_for_project(path: str, config: RunnableConfig) -> pathlib.Path:
    project_root = get_project_root(config)
    p = (project_root / path).resolve()
    if project_root.resolve() not in p.parents and project_root.resolve() != p.parent and project_root.resolve() != p:
        raise ValueError("Attempt to write outside project root")
    return p


@tool
def write_file(path: str, content: str, config: RunnableConfig) -> str:
    """Writes content to a file at the specified path within the project root."""
    p = safe_path_for_project(path, config)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return f"WROTE:{p}"


@tool
def read_file(path: str, config: RunnableConfig) -> str:
    """Reads content from a file at the specified path within the project root."""
    p = safe_path_for_project(path, config)
    if not p.exists():
        return ""
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


@tool
def get_current_directory(config: RunnableConfig) -> str:
    """Returns the current working directory."""
    return str(get_project_root(config))

@tool
def list_files(directory: str = ".", config: RunnableConfig = None) -> str:
    """Lists all files in the specified directory within the project root."""
    p = safe_path_for_project(directory, config)
    if not p.is_dir():
        return f"ERROR: {p} is not a directory"
    files = [str(f.relative_to(get_project_root(config))) for f in p.glob("**/*") if f.is_file()]
    return "\n".join(files) if files else "No files found."


@tool
def kill_process(procees_id: int) -> str: 
    """Kills a process by its ID."""
    try: 
        subprocess.run(f"kill {procees_id}", shell=True, check=True)
        return f"Process {procees_id} killed successfully."
    except subprocess.CalledProcessError as e: 
        return f"Failed to kill process {procees_id}: {e}"

@tool
def get_PID_of_process_running_on_port(port: int) -> str: 
    """Gets the process ID of the process running on the specified port."""
    try: 
        res = subprocess.run(f"lsof -i :{port} -t", shell=True, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e: 
        return f"Failed to get process ID for port {port}: {e}"