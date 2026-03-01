import uuid
from collections import deque

from flask import Flask, request, jsonify
from flask_cors import CORS

from agent.graph import run_agent
from agent.sandbox import get_sandbox, cleanup_sandbox, get_preview_url

# In-memory store: project_id → deque of last 50 error payloads
_preview_errors: dict[str, deque] = {}

app = Flask(__name__)
CORS(app)



@app.route("/api/test")
def test_api():
    return jsonify({"message": "Hello, this is a test API!"})


@app.route("/api/prompt", methods=["POST"])
def prompt_api():
    """
    Body (JSON):
      {
        "prompt":      "Build a todo app",
        "use_ollama":  false,   // optional
        "use_gemini":  false,   // optional
        "use_qwen":    false    // optional
      }

    Response (JSON):
      {
        "project_id":   "uuid",
        "status":       "ALL_DONE" | "RESOLVE_FAILED" | …,
        "preview_url":  "https://…",    // live app URL (E2B)
        "error":        null | "…"
      }
    """
    data = request.get_json(force=True)
    prompt      = data.get("prompt", "").strip()
    use_ollama  = bool(data.get("use_ollama", False))
    use_gemini  = bool(data.get("use_gemini", False))
    use_qwen    = bool(data.get("use_qwen", False))

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    project_id = str(uuid.uuid4())

    # ── 1. Pre-warm the sandbox so the container is ready before the agent starts
    print(f"[API] Creating E2B sandbox for project {project_id[:8]}…")
    get_sandbox(project_id)     # registers sandbox in the global registry

    # ── 2. Run the full agent pipeline
    try:
        result = run_agent(
            user_prompt=prompt,
            project_id=project_id,
            use_ollama=use_ollama,
            use_gemini=use_gemini,
            use_qwen=use_qwen,
        )
    except Exception as exc:
        # Don't leak the sandbox on crash — clean it up
        cleanup_sandbox(project_id)
        return jsonify({"error": str(exc), "project_id": project_id}), 500

    final_status = result.get("status", "UNKNOWN")
    server_port  = result.get("server_port")   # port reported by runner_agent

    # ── 3. Build preview URL using the exact port the runner reported
    preview_url = get_preview_url(project_id, port=server_port)
    print(f"[API] server_port={server_port}  preview_url={preview_url}")

    return jsonify({
        "project_id":  project_id,
        "status":      final_status,
        "server_port": server_port,
        "preview_url": preview_url,
        "error":       result.get("error_message"),
    })


@app.route("/api/preview/<project_id>")
def preview_api(project_id: str):
    """
    Return the live preview URL(s) for a project that has already been generated.
    Useful if the frontend wants to refresh or retrieve the URL without
    re-running the agent.
    """
    url = get_preview_url(project_id)
    if url is None:
        return jsonify({"error": "No active sandbox for this project_id"}), 404
    return jsonify({"project_id": project_id, "preview_url": url})


# ── postMessage error bridge ───────────────────────────────────────────────────

@app.route("/api/preview-error/<project_id>", methods=["POST"])
def preview_error_api(project_id: str):
    """
    Receive a runtime JS error forwarded from the preview iframe via postMessage.

    The frontend does:
        window.addEventListener('message', (e) => {
            if (e.data.type === 'preview-error') {
                fetch('/api/preview-error/' + e.data.project_id, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(e.data)
                });
            }
        });

    Or the injected script inside the sandbox POSTs directly.
    """
    error = request.get_json(force=True) or {}
    print(f"[PreviewError] project={project_id} | {error.get('message', '?')}")

    if project_id not in _preview_errors:
        _preview_errors[project_id] = deque(maxlen=50)
    _preview_errors[project_id].appendleft(error)

    return jsonify({"received": True})


@app.route("/api/preview-errors/<project_id>")
def get_preview_errors_api(project_id: str):
    """Return the last runtime errors collected for a project."""
    errors = list(_preview_errors.get(project_id, []))
    return jsonify({"project_id": project_id, "errors": errors})


@app.route("/api/fix-error/<project_id>", methods=["POST"])
def fix_error_api(project_id: str):
    """
    Trigger the resolver agent to fix the most recent runtime error.
    The frontend calls this when the user clicks 'Auto-fix'.
    """
    errors = list(_preview_errors.get(project_id, []))
    if not errors:
        return jsonify({"error": "No errors recorded for this project"}), 400

    latest = errors[0]
    error_msg = latest.get("message", "Unknown error")
    stack     = latest.get("stack") or ""

    # Re-run the agent in resolver mode by injecting the error as the prompt
    fix_prompt = (
        f"Runtime error in the preview:\n\n"
        f"Message: {error_msg}\n"
        f"Stack:\n{stack}\n\n"
        "Identify the root cause in the source files and fix it. "
        "Leave the dev server running after the fix."
    )

    try:
        get_sandbox(project_id)   # ensure sandbox is alive before fix attempt
        result = run_agent(
            user_prompt=fix_prompt,
            project_id=project_id,
        )
        # Clear the error after a fix attempt
        _preview_errors.pop(project_id, None)
        return jsonify({
            "project_id": project_id,
            "status": result.get("status"),
            "error": result.get("error_message"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Sandbox cleanup ────────────────────────────────────────────────────────────

@app.route("/api/sandbox/<project_id>", methods=["DELETE"])
def cleanup_api(project_id: str):
    """
    Kill the E2B sandbox for a project to free cloud resources.
    Call this when the user closes their preview or navigates away.
    """
    cleanup_sandbox(project_id)
    return jsonify({"message": f"Sandbox {project_id[:8]}… cleaned up."})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
