import threading
import uuid
from collections import deque
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS

from agent.graph import run_agent
from agent.sandbox import get_sandbox, cleanup_sandbox, get_preview_url

app = Flask(__name__)
CORS(app)

# ── in-memory stores ───────────────────────────────────────────────────────────

# job_id → job dict
_jobs: dict[str, dict] = {}

# project_id → deque of last 50 runtime error payloads
_preview_errors: dict[str, deque] = {}


# ── background worker ──────────────────────────────────────────────────────────

def _run_agent_job(job_id: str, project_id: str, prompt: str, **model_kwargs):
    """Runs the full agent pipeline in a background thread and updates _jobs."""

    def on_phase(node_name: str, info: dict):
        _jobs[job_id]["phase"]      = node_name
        _jobs[job_id]["phase_info"] = info   # e.g. {"step": 2, "total_steps": 5, "step_description": "..."}

    try:
        result = run_agent(
            user_prompt=prompt,
            project_id=project_id,
            on_phase=on_phase,
            **model_kwargs,
        )
        server_port = result.get("server_port")
        preview_url = get_preview_url(project_id, port=server_port)
        _jobs[job_id].update({
            "status"      : "done",
            "phase"       : "done",
            "agent_status": result.get("status", "UNKNOWN"),
            "preview_url" : preview_url,
            "server_port" : server_port,
            "error"       : result.get("error_message"),
        })
        print(f"[Job {job_id[:8]}] done — agent_status={result.get('status')}")
    except Exception as exc:
        cleanup_sandbox(project_id)
        _jobs[job_id].update({
            "status": "error",
            "phase" : None,
            "error" : str(exc),
        })
        print(f"[Job {job_id[:8]}] error — {exc}")


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/test")
def test_api():
    return jsonify({"message": "Hello, this is a test API!"})


@app.route("/api/prompt", methods=["POST"])
def prompt_api():
    """
    Submit a build prompt.  Returns immediately with a job_id for polling.

    Request JSON:
      { "prompt": "...", "use_gemini": true, "use_groq": false, ... }

    Response JSON:
      { "job_id": "uuid", "project_id": "uuid" }
    """
    data       = request.get_json(force=True)
    prompt     = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    model_kwargs = {
        "use_ollama": bool(data.get("use_ollama", False)),
        "use_gemini": bool(data.get("use_gemini", False)),
        "use_qwen"  : bool(data.get("use_qwen",   False)),
        "use_groq"  : bool(data.get("use_groq",   False)),
    }

    job_id     = str(uuid.uuid4())
    project_id = str(uuid.uuid4())

    # Register job immediately so the status endpoint can respond right away
    _jobs[job_id] = {
        "status"      : "starting",
        "phase"       : None,
        "phase_info"  : {},
        "project_id"  : project_id,
        "preview_url" : None,
        "agent_status": None,
        "server_port" : None,
        "error"       : None,
        "created_at"  : datetime.now(timezone.utc).isoformat(),
    }

    # Pre-warm E2B sandbox (fast; ~1 s) before handing off to the thread
    print(f"[API] Pre-warming sandbox for project {project_id[:8]}…")
    get_sandbox(project_id)
    _jobs[job_id]["status"] = "running"

    # Start agent pipeline in a daemon thread
    thread = threading.Thread(
        target=_run_agent_job,
        args=(job_id, project_id, prompt),
        kwargs=model_kwargs,
        daemon=True,
        name=f"agent-{job_id[:8]}",
    )
    thread.start()

    return jsonify({"job_id": job_id, "project_id": project_id})


@app.route("/api/status/<job_id>")
def status_api(job_id: str):
    """
    Poll this endpoint every few seconds to track build progress.

    Response JSON:
      {
        "job_id":       "uuid",
        "project_id":   "uuid",
        "status":       "starting" | "running" | "done" | "error",
        "phase":        "plan" | "architect" | "coder" | "runner" | "resolver" | "done" | null,
        "agent_status": "ALL_DONE" | "RESOLVE_FAILED" | … | null,
        "preview_url":  "https://…" | null,
        "error":        "…" | null
      }
    """
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "job_id"      : job_id,
        "project_id"  : job["project_id"],
        "status"      : job["status"],
        "phase"       : job.get("phase"),
        "phase_info"  : job.get("phase_info", {}),   # step / total_steps / step_description
        "agent_status": job.get("agent_status"),
        "preview_url" : job.get("preview_url"),
        "server_port" : job.get("server_port"),
        "error"       : job.get("error"),
    })


@app.route("/api/preview/<project_id>")
def preview_api(project_id: str):
    url = get_preview_url(project_id)
    if url is None:
        return jsonify({"error": "No active sandbox for this project_id"}), 404
    return jsonify({"project_id": project_id, "preview_url": url})


# ── postMessage error bridge ───────────────────────────────────────────────────

@app.route("/api/preview-error/<project_id>", methods=["POST"])
def preview_error_api(project_id: str):
    """Receive a runtime JS error forwarded from the preview iframe."""
    error = request.get_json(force=True) or {}
    print(f"[PreviewError] project={project_id[:8]} | {error.get('message', '?')[:120]}")

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
    Blocks until the fix attempt completes (typically < 60 s).
    """
    errors = list(_preview_errors.get(project_id, []))
    if not errors:
        return jsonify({"error": "No errors recorded for this project"}), 400

    latest    = errors[0]
    error_msg = latest.get("message", "Unknown error")
    stack     = latest.get("stack") or ""

    fix_prompt = (
        f"Runtime error in the preview:\n\n"
        f"Message: {error_msg}\n"
        f"Stack:\n{stack}\n\n"
        "Identify the root cause in the source files and fix it. "
        "Leave the dev server running after the fix."
    )

    try:
        get_sandbox(project_id)
        result = run_agent(user_prompt=fix_prompt, project_id=project_id)
        _preview_errors.pop(project_id, None)
        return jsonify({
            "project_id": project_id,
            "status"    : result.get("status"),
            "error"     : result.get("error_message"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── sandbox cleanup ────────────────────────────────────────────────────────────

@app.route("/api/sandbox/<project_id>", methods=["DELETE"])
def cleanup_api(project_id: str):
    cleanup_sandbox(project_id)
    return jsonify({"message": f"Sandbox {project_id[:8]}… cleaned up."})


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # threaded=True is required for background jobs + simultaneous poll requests
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
