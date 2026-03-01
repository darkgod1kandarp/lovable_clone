"""
sandbox.py — E2B sandbox manager.

A single SandboxManager wraps one E2B Sandbox instance per project.
get_sandbox(project_id) returns the cached instance, creating it on first call.
"""

from e2b_code_interpreter import Sandbox
import os
import threading

os.environ["E2B_API_KEY"] = "e2b_711d9614c47701c4690fa0ab78137c2e44dff94f"

_sandboxes: dict[str, "SandboxManager"] = {}
_lock = threading.Lock()


class SandboxManager:
    def __init__(self, project_id: str):
        self.project_id = project_id
        print(f"[Sandbox] Spinning up E2B sandbox for project: {project_id}")
        self._sbx = Sandbox.create(timeout=3600)  # 1 hour — keeps server alive after agent finishes
        self.work_dir = f"/home/user/{project_id}"
        self._sbx.commands.run(f"mkdir -p {self.work_dir}")
        print(f"[Sandbox] Ready. work_dir={self.work_dir}")

    # ── Commands ────────────────────────────────────────────────────────────────

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative path to an absolute path inside work_dir."""
        if not path.startswith("/"):
            return f"{self.work_dir}/{path}"
        return path

    def run_cmd(self, cmd: str, cwd: str = None, timeout: int = 60):
        """Run a foreground command. Returns (exit_code, stdout, stderr)."""
        cwd = cwd or self.work_dir
        try:
            result = self._sbx.commands.run(cmd, cwd=cwd, timeout=timeout)
            return result.exit_code, result.stdout, result.stderr
        except Exception as e:
            return 1, "", f"[Sandbox error] {e}"

    def run_background_cmd(self, cmd: str, cwd: str = None):
        """
        Fire-and-forget a long-running command (dev server, etc.).
        Returns immediately with (0, message, "").
        """
        cwd = cwd or self.work_dir
        try:
            bg_cmd = f"nohup {cmd} > /tmp/bg_stdout.log 2>/tmp/bg_stderr.log &"
            self._sbx.commands.run(bg_cmd, cwd=cwd, timeout=15)
            return 0, f"Started in background: {cmd}", ""
        except Exception as e:
            return 1, "", f"[Sandbox error] {e}"

    # ── File I/O ────────────────────────────────────────────────────────────────

    def write_file(self, path: str, content: str) -> str:
        """Write content to path inside the sandbox (relative paths resolved to work_dir)."""
        path = self._resolve_path(path)
        try:
            # Ensure parent directory exists
            parent = "/".join(path.split("/")[:-1])
            if parent:
                self._sbx.commands.run(f"mkdir -p {parent}")
            self._sbx.files.write(path, content)
            return f"Written: {path}"
        except Exception as e:
            return f"[Error writing {path}] {e}"

    def read_file(self, path: str) -> str:
        """Read a file from inside the sandbox. Returns an error string if missing."""
        path = self._resolve_path(path)
        try:
            return self._sbx.files.read(path)
        except Exception as e:
            return f"[Error] File not found or unreadable: {path} — {e}"

    def list_files(self, directory: str = ".") -> str:
        """List files, excluding noisy dirs like node_modules, .git, dist, build."""
        if directory in (".", ""):
            directory = self.work_dir
        elif not directory.startswith("/"):
            directory = f"{self.work_dir}/{directory}"

        try:
            result = self._sbx.commands.run(
                f"find {directory} "
                r"-not \( -name node_modules -prune \) "
                r"-not \( -name .git -prune \) "
                r"-not \( -name __pycache__ -prune \) "
                r"-not \( -name dist -prune \) "
                r"-not \( -name build -prune \) "
                "-type f",
                timeout=30,
            )
            return result.stdout or "(empty directory)"
        except Exception as e:
            return f"[Error listing {directory}] {e}"

    # ── Process management ──────────────────────────────────────────────────────

    def kill_process(self, pid: int) -> str:
        try:
            self._sbx.commands.run(f"kill {pid} 2>/dev/null || true")
            return f"Killed PID {pid}"
        except Exception as e:
            return f"[Error killing PID {pid}] {e}"

    def get_pid_on_port(self, port: int) -> str:
        """Returns a non-empty string if a process is listening on port, else empty."""
        try:
            result = self._sbx.commands.run(
                f"ss -tlnp | grep :{port} || true"
            )
            return result.stdout.strip()
        except Exception as e:
            return f"[Error checking port {port}] {e}"

    # ── Error tracker injection ─────────────────────────────────────────────────

    def inject_error_tracker(self, api_base_url: str = "") -> str:
        """
        Find the project's index.html and inject a postMessage error-tracking
        script before </head>.

        The script catches window.onerror + unhandledrejection and:
          1. Sends a postMessage to the parent window (picked up by the iframe host).
          2. Optionally POSTs to /api/preview-error/<project_id> on the Flask server.

        api_base_url — e.g. "http://localhost:5000" — if empty, only postMessage is used.
        """
        ERROR_SCRIPT = f"""<script>
(function() {{
  var _pid = "{self.project_id}";
  var _api = "{api_base_url}";

  function _report(data) {{
    var payload = Object.assign({{ type: "preview-error", project_id: _pid }}, data);
    // 1. postMessage to parent frame
    try {{ window.parent.postMessage(payload, "*"); }} catch(e) {{}}
    // 2. POST back to Flask so the resolver can pick it up
    if (_api) {{
      try {{
        fetch(_api + "/api/preview-error/" + _pid, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});
      }} catch(e) {{}}
    }}
  }}

  window.addEventListener("error", function(e) {{
    _report({{
      message : e.message,
      source  : e.filename,
      line    : e.lineno,
      col     : e.colno,
      stack   : e.error ? e.error.stack : null
    }});
  }});

  window.addEventListener("unhandledrejection", function(e) {{
    _report({{
      message : String(e.reason),
      stack   : e.reason && e.reason.stack ? e.reason.stack : null
    }});
  }});
}})();
</script>"""

        # Find the project's index.html (exclude node_modules, dist, build)
        result = self._sbx.commands.run(
            f'find {self.work_dir} -name "index.html" '
            r'-not \( -path "*/node_modules/*" \) '
            r'-not \( -path "*/dist/*" \) '
            r'-not \( -path "*/build/*" \) '
            '-print -quit',
            timeout=10,
        )
        html_path = result.stdout.strip()

        if not html_path:
            return "[ErrorTracker] No index.html found — skipping injection"

        try:
            content = self._sbx.files.read(html_path)
            if "preview-error" in content:
                return f"[ErrorTracker] Already injected in {html_path}"

            if "</head>" in content:
                content = content.replace("</head>", f"{ERROR_SCRIPT}\n</head>", 1)
            else:
                content = ERROR_SCRIPT + "\n" + content

            self._sbx.files.write(html_path, content)
            return f"[ErrorTracker] Injected into {html_path}"
        except Exception as e:
            return f"[ErrorTracker] Failed: {e}"

    # ── LangChain tool factory ───────────────────────────────────────────────────

    def get_tools(self, include_process_tools: bool = False, include_kill: bool = True) -> list:
        """
        Return a list of LangChain tools bound to this sandbox instance.

        Pass include_process_tools=True to also include kill_process and
        get_PID_of_process_running_on_port (needed by runner / resolver agents).

        Usage:
            sbx = get_sandbox(project_id)
            agent = create_react_agent(llm, sbx.get_tools())
        """
        from langchain_core.tools import tool

        _sbx = self  # capture self in closure

        _BACKGROUND_PREFIXES = (
            "npm run dev", "npm start", "npm run start",
            "yarn dev", "yarn start", "pnpm dev", "pnpm start",
            "vite", "next dev", "python manage.py runserver",
            "flask run", "uvicorn", "fastapi dev", "nodemon",
        )
        _SLOW_TIMEOUTS = {
            "npm install": 300, "npm ci": 300,
            "yarn install": 300, "yarn": 300,
            "pnpm install": 300,
            "pip install": 180, "pip3 install": 180,
            "npm run build": 180, "vite build": 180,
            "next build": 300, "tsc": 120,
        }

        def _is_background(cmd: str) -> bool:
            return any(cmd.strip().lower().startswith(p) for p in _BACKGROUND_PREFIXES)

        def _smart_timeout(cmd: str, base: int) -> int:
            lower = cmd.strip().lower()
            for prefix, t in _SLOW_TIMEOUTS.items():
                if lower.startswith(prefix):
                    return max(base, t)
            return max(base, 60)

        @tool
        def run_cmd(cmd: str, cwd: str = None, timeout: int = 60) -> str:
            """
            Run a shell command inside the E2B sandbox.
            Dev-server commands (npm run dev, vite, flask run, etc.) are
            automatically backgrounded and return immediately.
            Install commands (npm install, pip install) get extended timeouts.
            Returns exit_code, stdout, and stderr.
            """
            if _is_background(cmd):
                exit_code, stdout, stderr = _sbx.run_background_cmd(cmd, cwd)
            else:
                exit_code, stdout, stderr = _sbx.run_cmd(cmd, cwd, _smart_timeout(cmd, timeout))

            parts = [f"exit_code: {exit_code}"]
            if stdout.strip():
                parts.append(f"stdout:\n{stdout.strip()}")
            if stderr.strip():
                parts.append(f"stderr:\n{stderr.strip()}")
            return "\n".join(parts)

        @tool
        def write_file(path: str, content: str) -> str:
            """Write content to a file inside the E2B sandbox. Parent directories are created automatically."""
            return _sbx.write_file(path, content)

        @tool
        def read_file(path: str) -> str:
            """Read a file from inside the E2B sandbox. Returns an error message if the file does not exist."""
            return _sbx.read_file(path)

        @tool
        def list_files(directory: str = ".") -> str:
            """List all files in a directory inside the E2B sandbox. Excludes node_modules, .git, dist, build."""
            return _sbx.list_files(directory)

        @tool
        def get_current_directory() -> str:
            """Return the working directory path inside the sandbox."""
            return _sbx.work_dir

        base_tools = [run_cmd, write_file, read_file, list_files, get_current_directory]

        if include_process_tools:
            @tool
            def get_PID_of_process_running_on_port(port: int) -> str:
                """
                Check whether a process is listening on a port inside the E2B sandbox.
                Returns a non-empty string if a process is found, empty string if not.
                """
                return _sbx.get_pid_on_port(port)

            base_tools.append(get_PID_of_process_running_on_port)

            if include_kill:
                @tool
                def kill_process(pid: int) -> str:
                    """Kill a process by PID inside the E2B sandbox."""
                    return _sbx.kill_process(pid)

                base_tools.append(kill_process)

        return base_tools
    

def get_preview_url(project_id: str, port: int = None) -> str | None:
    """
    Return the public preview URL for the sandbox.

    Uses sbx.get_host(port) which returns the correct E2B-assigned hostname
    for the given port (e.g. '{sandbox_id}-3000.e2b.dev').
    """
    sbx_manager = get_sandbox(project_id)
    sbx = sbx_manager._sbx

    ports_to_try = [port] if port else [3000, 5173, 5000, 8000, 8080]

    for p in ports_to_try:
        pid = sbx_manager.get_pid_on_port(p)
        if not pid:
            print(f"[Preview] Port {p} not open for project {project_id}")
            continue
        try:
            host = sbx.get_host(p)
            url = f"https://{host}"
            print(f"[Preview] URL for project {project_id} port {p}: {url}")
            return url
        except Exception as e:
            print(f"[Preview] get_host({p}) failed: {e}")

    return None

def cleanup_sandbox(project_id: str):   
    """Delete the sandbox for project_id, if it exists."""
    with _lock:
        sbx = _sandboxes.pop(project_id, None)
    if sbx:
        print(f"[Sandbox] Cleaning up sandbox for project {project_id}…")
        sbx._sbx.kill()  # force-kill the sandbox container
    else:
        print(f"[Sandbox] No sandbox found to clean for project {project_id}")
# ── Registry ────────────────────────────────────────────────────────────────────

def get_sandbox(project_id: str) -> SandboxManager:
    """Return the SandboxManager for project_id, creating it if needed."""
    with _lock:
        if project_id not in _sandboxes:
            _sandboxes[project_id] = SandboxManager(project_id)
        return _sandboxes[project_id]
