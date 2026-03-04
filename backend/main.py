import threading
import uuid
from collections import deque
from datetime import datetime, timezone

import io
import requests as _requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from agent.graph import run_agent, run_edit_agent
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


def _run_edit_job(job_id: str, project_id: str, edit_prompt: str, **model_kwargs):
    """Runs the edit agent in a background thread and updates _jobs."""

    def on_phase(node_name: str, info: dict):
        _jobs[job_id]["phase"]      = node_name
        _jobs[job_id]["phase_info"] = info

    try:
        result      = run_edit_agent(
            edit_prompt=edit_prompt,
            project_id=project_id,
            on_phase=on_phase,
            **model_kwargs,
        )
        server_port = result.get("server_port", 3000)
        preview_url = get_preview_url(project_id, port=server_port)
        _jobs[job_id].update({
            "status"      : "done",
            "phase"       : "done",
            "agent_status": result.get("status", "UNKNOWN"),
            "preview_url" : preview_url,
            "server_port" : server_port,
            "error"       : result.get("error_message"),
        })
        print(f"[EditJob {job_id[:8]}] done — agent_status={result.get('status')}")
    except Exception as exc:
        _jobs[job_id].update({
            "status": "error",
            "phase" : None,
            "error" : str(exc),
        })
        print(f"[EditJob {job_id[:8]}] error — {exc}")


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
        "use_claude": bool(data.get("use_claude", False)),
    }

    job_id     = str(uuid.uuid4())
    project_id = str(uuid.uuid4())

    # Register job immediately so the status endpoint can respond right away
    _jobs[job_id] = {
        "status"       : "starting",
        "phase"        : None,
        "phase_info"   : {},
        "project_id"   : project_id,
        "preview_url"  : None,
        "agent_status" : None,
        "server_port"  : None,
        "error"        : None,
        "model_kwargs" : model_kwargs,   # stored so fix-error can reuse the same model
        "created_at"   : datetime.now(timezone.utc).isoformat(),
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


@app.route("/api/preview-html/<project_id>")
def preview_html_api(project_id: str):
    """Proxy-fetch the preview HTML server-side to avoid browser CORS blocks."""
    url = get_preview_url(project_id)
    if url is None:
        return jsonify({"error": "No active sandbox for this project_id"}), 404
    try:
        resp = _requests.get(url, timeout=10, headers={"Cache-Control": "no-store"})
        return jsonify({"html": resp.text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


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

@app.route("/api/download/<project_id>")
def download_api(project_id: str):
    """Download the project source as a ZIP file."""
    try:
        sb = get_sandbox(project_id)
        zip_bytes = sb.download_project_zip()   
        print(len(zip_bytes), "bytes")  # log size for debugging
        return send_file(
            io.BytesIO(zip_bytes),
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"project-{project_id[:8]}.zip",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/api/preview-errors/<project_id>")
def get_preview_errors_api(project_id: str):
    """Return the last runtime errors collected for a project."""
    errors = list(_preview_errors.get(project_id, []))
    return jsonify({"project_id": project_id, "errors": errors})

@app.route("/api/continue-prompt/<project_id>", methods=["POST"])   
def continue_prompt_api(project_id: str):
    """
    Continue the agent with a custom user prompt (e.g. "Please fix the error and continue").
    This is for advanced users who want to interact with the agent beyond the automatic fix.
    """
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Look up the model used for the original build job so we reuse the same LLM
    model_kwargs = {}
    for job in _jobs.values():
        if job.get("project_id") == project_id:
            model_kwargs = job.get("model_kwargs", {})
            break

    try:
        get_sandbox(project_id)  # ensure sandbox exists before continuing
        result = run_agent(user_prompt=prompt, project_id=project_id, **model_kwargs)
        return jsonify({
            "project_id": project_id,
            "status"    : result.get("status"),
            "error"     : result.get("error_message"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

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

    # Look up the model used for the original build job so we reuse the same LLM
    model_kwargs = {}
    for job in _jobs.values():
        if job.get("project_id") == project_id:
            model_kwargs = job.get("model_kwargs", {})
            break

    try:
        get_sandbox(project_id)
        result = run_agent(user_prompt=fix_prompt, project_id=project_id, **model_kwargs)
        _preview_errors.pop(project_id, None)
        return jsonify({
            "project_id": project_id,
            "status"    : result.get("status"),
            "error"     : result.get("error_message"),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/edit/<project_id>", methods=["POST"])
def edit_api(project_id: str):
    """
    Submit a follow-up edit prompt for an already-built project.
    Returns immediately with a job_id for polling via /api/status/<job_id>.

    Request JSON:
      { "prompt": "Add a sticky navbar with a dark background" }

    Response JSON:
      { "job_id": "uuid", "project_id": "uuid" }
    """
    data        = request.get_json(force=True)
    edit_prompt = data.get("prompt", "").strip()
    if not edit_prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Reuse the same model that was used for the original build
    model_kwargs = {}
    for job in _jobs.values():
        if job.get("project_id") == project_id:
            model_kwargs = job.get("model_kwargs", {})
            break

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status"      : "running",
        "phase"       : None,
        "phase_info"  : {},
        "project_id"  : project_id,
        "preview_url" : get_preview_url(project_id),
        "agent_status": None,
        "server_port" : None,
        "error"       : None,
        "model_kwargs": model_kwargs,
        "created_at"  : datetime.now(timezone.utc).isoformat(),
    }

    thread = threading.Thread(
        target=_run_edit_job,
        args=(job_id, project_id, edit_prompt),
        kwargs=model_kwargs,
        daemon=True,
        name=f"edit-{job_id[:8]}",
    )
    thread.start()

    return jsonify({"job_id": job_id, "project_id": project_id})


# ── sandbox cleanup ────────────────────────────────────────────────────────────

@app.route("/api/sandbox/<project_id>", methods=["DELETE"])
def cleanup_api(project_id: str):
    cleanup_sandbox(project_id)
    return jsonify({"message": f"Sandbox {project_id[:8]}… cleaned up."})


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # threaded=True is required for background jobs + simultaneous poll requests
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
