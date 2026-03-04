import { useState, useEffect, useRef, useCallback } from 'react';

const API_BASE      = 'http://localhost:5000';
const MAX_AUTO_FIXES = 3;

// ── constants ─────────────────────────────────────────────────────────────────

const PHASE_LABELS = {
  plan      : '📋 Planning your app…',
  architect : '🏗  Architecting structure…',
  coder     : '💻 Writing code…',
  runner    : '🚀 Starting dev server…',
  resolver  : '🔧 Fixing errors…',
  done      : '✅ Done!',
};

const MODELS = [
  { value: 'use_gemini', label: 'Gemini 2.5 Pro' },
  { value: 'use_groq',   label: 'Groq GPT-OSS'   },
  { value: 'use_qwen',   label: 'Qwen 3'          },
  { value: 'use_ollama', label: 'Ollama (local)'  },
  { value: 'use_claude', label: 'Claude'  },
];

// ── helpers ───────────────────────────────────────────────────────────────────

function agentStatusColor(s) {
  if (!s) return '#888';
  if (s === 'ALL_DONE') return '#22c55e';
  if (s.includes('FAIL') || s.includes('ERROR')) return '#ef4444';
  return '#f59e0b';
}

// ── component ─────────────────────────────────────────────────────────────────

export default function PromptInput() {
  const [prompt, setPrompt]       = useState('');
  const [model, setModel]         = useState('use_gemini');

  // job tracking
  const [, setJobId]               = useState(null);
  const [jobState, setJobState]   = useState(null);  // raw /api/status response

  // runtime errors captured via postMessage / backend polling
  const [errors, setErrors]       = useState([]);
  const [isFixing, setIsFixing]   = useState(false);
  const [fixStatus, setFixStatus] = useState(null);  // 'ok' | 'fail' | null
  const [autoFixLog, setAutoFixLog] = useState([]);  // history of auto-fix attempts

  // iterative editing
  const [editPrompt, setEditPrompt]   = useState('');
  const [isEditing, setIsEditing]     = useState(false);
  const [editHistory, setEditHistory] = useState([]);  // [{prompt, status}]

  const iframeRef     = useRef(null);
  const jobPollRef    = useRef(null);   // interval for job status
  const errPollRef    = useRef(null);   // interval for preview errors
  const autoFixRef    = useRef(null);   // timer for auto-fix debounce
  const autoFixCount  = useRef(0);      // how many auto-fixes have been attempted

  // derived
  const isBuilding  = jobState && (jobState.status === 'starting' || jobState.status === 'running');
  const isDone      = jobState?.status === 'done';
  const isError     = jobState?.status === 'error';
  const previewUrl  = jobState?.preview_url ?? null;
  const projectId   = jobState?.project_id  ?? null;
  const hasErrors   = errors.length > 0;

  // ── postMessage listener ────────────────────────────────────────────────────
  useEffect(() => {
    function onMessage(event) {
      const data = event.data;
      if (!data || data.type !== 'preview-error' || !projectId) return;

      const payload = {
        message : data.message || 'Unknown error',
        source  : data.source  || null,
        lineno  : data.lineno  || null,
        colno   : data.colno   || null,
        stack   : data.stack   || null,
        kind    : data.kind    || 'runtime',
        ts      : Date.now(),
      };

      setErrors(prev => [payload, ...prev].slice(0, 20));

      fetch(`${API_BASE}/api/preview-error/${projectId}`, {
        method  : 'POST',
        headers : { 'Content-Type': 'application/json' },
        body    : JSON.stringify(payload),
      }).catch(() => {});
    }

    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [projectId]);

  // ── job status polling ──────────────────────────────────────────────────────
  const startJobPoll = useCallback((jid) => {
    if (jobPollRef.current) clearInterval(jobPollRef.current);

    jobPollRef.current = setInterval(async () => {
      try {
        const res  = await fetch(`${API_BASE}/api/status/${jid}`);
        const data = await res.json();
        setJobState(data);

        // Stop polling once the job has finished
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(jobPollRef.current);
          jobPollRef.current = null;
          // Switch to error polling once preview is live
          if (data.project_id) startErrPoll(data.project_id);
          // Also poll the preview URL itself for Next.js build errors
          if (data.preview_url && data.project_id) startBuildErrPoll(data.preview_url, data.project_id);
          // Reload iframe so edits are reflected.
          // Wait 3 s for Next.js to finish hot-reloading, then force a hard reload
          // using a cache-busting query param (src += '' is a no-op in many browsers).
          if (data.status === 'done' && iframeRef.current) {
            const currentSrc = iframeRef.current.src.split('?')[0];
            setTimeout(() => {
              if (iframeRef.current) {
                iframeRef.current.src = currentSrc + '?t=' + Date.now();
              }
            }, 3000);
          }
        }
      } catch (_) {}
    }, 2500);
  }, []); // eslint-disable-line

  // ── preview error polling ───────────────────────────────────────────────────
  const startErrPoll = useCallback((pid) => {
    if (errPollRef.current) clearInterval(errPollRef.current);

    errPollRef.current = setInterval(async () => {
      try {
        const res  = await fetch(`${API_BASE}/api/preview-errors/${pid}`);
        const data = await res.json();
        if (data.errors?.length) {
          // Only update state when errors actually change to avoid spurious re-renders
          // that would keep resetting the auto-fix debounce timer.
          setErrors(prev => {
            if (JSON.stringify(prev) === JSON.stringify(data.errors)) return prev;
            return data.errors;
          });
        }
      } catch (_) {}
    }, 4000);
  }, []);

  // ── build error detection via HTTP poll ─────────────────────────────────────
  // Next.js build errors (webpack compile failures) never fire window.onerror.
  // Instead, we fetch the preview URL from the parent and look for the
  // "Build Error" / "Failed to compile" text that Next.js injects into the HTML.
  const buildErrPollRef = useRef(null);
  const startBuildErrPoll = useCallback((_previewUrl, pid) => {
    if (buildErrPollRef.current) clearInterval(buildErrPollRef.current);
    buildErrPollRef.current = setInterval(async () => {
      try {
        const res  = await fetch(`${API_BASE}/api/preview-html/${pid}`, { cache: 'no-store' });
        const data = await res.json();
        if (!data.html) return;
        const html = data.html;
        // Next.js error overlay injects these strings into the page HTML
        if (html.includes('Failed to compile') || html.includes('Module not found')) {
          // Extract the first error message we can find
          const match = html.match(/Module not found[^<"]{0,200}|Can't resolve '[^']+'/);
          const msg   = match ? match[0].replace(/<[^>]+>/g, '').trim() : 'Build error: Failed to compile';
          const payload = { message: msg, kind: 'build-error', ts: Date.now() };
          setErrors(prev => {
            if (prev.some(e => e.message === msg)) return prev;
            return [payload, ...prev].slice(0, 20);
          });
          // Also report to backend so fix-error endpoint can use it
          fetch(`${API_BASE}/api/preview-error/${pid}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          }).catch(() => {});
        }
      } catch (_) {}
    }, 6000);
  }, []);

  // ── auto-fix on runtime error ───────────────────────────────────────────────
  // When the build is done and new errors arrive, automatically trigger the
  // resolver after a short delay. Retries up to MAX_AUTO_FIXES times.
  useEffect(() => {
    if (!isDone || !projectId || isFixing || errors.length === 0) return;
    if (autoFixCount.current >= MAX_AUTO_FIXES) return;

    // Debounce: wait 4 s in case more errors arrive before triggering
    clearTimeout(autoFixRef.current);
    autoFixRef.current = setTimeout(async () => {
      if (isFixing) return;
      autoFixCount.current += 1;
      setIsFixing(true);
      setFixStatus(null);

      const attempt = autoFixCount.current;
      setAutoFixLog(prev => [...prev, `Auto-fix #${attempt} triggered…`]);

      try {
        const res  = await fetch(`${API_BASE}/api/fix-error/${projectId}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
        });
        const data = await res.json();
        const ok   = data.status === 'ALL_DONE' || data.status === 'RESOLVED_RUNTIME';
        setFixStatus(ok ? 'ok' : 'fail');
        setAutoFixLog(prev => [...prev, `Auto-fix #${attempt}: ${ok ? 'resolved ✓' : 'failed ✗'}`]);
        if (ok) {
          setErrors([]);
          autoFixCount.current = 0;  // reset on success so future errors get fixed too
          if (iframeRef.current) iframeRef.current.src += '';
        }
      } catch (_) {
        setFixStatus('fail');
        setAutoFixLog(prev => [...prev, `Auto-fix #${attempt}: network error`]);
      } finally {
        setIsFixing(false);
      }
    }, 4000);
  }, [errors, isDone, projectId]); // eslint-disable-line

  // Clean up intervals on unmount
  useEffect(() => () => {
    clearInterval(jobPollRef.current);
    clearInterval(errPollRef.current);
    clearInterval(buildErrPollRef.current);
    clearTimeout(autoFixRef.current);
  }, []);

  // ── submit ──────────────────────────────────────────────────────────────────
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isBuilding) return;

    // Reset state
    setJobId(null);
    setJobState(null);
    setErrors([]);
    setFixStatus(null);
    clearInterval(jobPollRef.current);
    clearInterval(errPollRef.current);
    clearInterval(buildErrPollRef.current);

    // POST prompt — backend returns immediately with job_id
    try {
      const res  = await fetch(`${API_BASE}/api/prompt`, {
        method  : 'POST',
        headers : { 'Content-Type': 'application/json' },
        body    : JSON.stringify({ prompt, [model]: true }),
      });
      const data = await res.json();
      if (data.error) { setJobState({ status: 'error', error: data.error }); return; }
      setJobId(data.job_id);
      setJobState({ status: 'starting', phase: null, project_id: data.project_id });
      startJobPoll(data.job_id);
    } catch (err) {
      setJobState({ status: 'error', error: err.message });
    }
  };

  // ── auto-fix ────────────────────────────────────────────────────────────────
  const fixInFlightRef = useRef(false);   // synchronous guard against concurrent calls
  const handleAutoFix = async () => {
    if (!projectId || isFixing || fixInFlightRef.current) return;
    fixInFlightRef.current = true;
    setIsFixing(true);
    setFixStatus(null);

    try {
      const res  = await fetch(`${API_BASE}/api/fix-error/${projectId}`, {
        method  : 'POST',
        headers : { 'Content-Type': 'application/json' },
      });
      const data = await res.json();
      const ok   = data.status === 'ALL_DONE' || data.status === 'RESOLVED_RUNTIME';
      setFixStatus(ok ? 'ok' : 'fail');
      if (ok) {
        setErrors([]);
        if (iframeRef.current) iframeRef.current.src += '';  // force reload
      }
    } catch (_) {
      setFixStatus('fail');
    } finally {
      setIsFixing(false);
      fixInFlightRef.current = false;
    }
  };

  // ── iterative edit ──────────────────────────────────────────────────────────
  const handleEdit = async (e) => {
    e.preventDefault();
    if (!editPrompt.trim() || isEditing || !projectId) return;

    const thisPrompt = editPrompt.trim();
    setEditPrompt('');
    setIsEditing(true);
    setEditHistory(prev => [...prev, { prompt: thisPrompt, status: 'running' }]);

    try {
      const res  = await fetch(`${API_BASE}/api/edit/${projectId}`, {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({ prompt: thisPrompt }),
      });
      const data = await res.json();
      if (data.error) {
        setEditHistory(prev => prev.map((h, i) =>
          i === prev.length - 1 ? { ...h, status: 'error' } : h));
        return;
      }
      // Start polling — iframe reload + history update happen when the job finishes
      startJobPoll(data.job_id);
      // Keep status as 'running' (⏳) until the poll sees the job is done.
      // We use a separate interval here so the edit history icon updates correctly.
      const checkDone = setInterval(async () => {
        try {
          const sr = await fetch(`${API_BASE}/api/status/${data.job_id}`);
          const sd = await sr.json();
          if (sd.status === 'done' || sd.status === 'error') {
            clearInterval(checkDone);
            const nextStatus = sd.status === 'done' ? 'done' : 'error';
            setEditHistory(prev => prev.map((h, i) =>
              i === prev.length - 1 ? { ...h, status: nextStatus } : h));
          }
        } catch (_) { clearInterval(checkDone); }
      }, 2500);
    } catch (_) {
      setEditHistory(prev => prev.map((h, i) =>
        i === prev.length - 1 ? { ...h, status: 'error' } : h));
    } finally {
      setIsEditing(false);
    }
  };

  // ── styles ──────────────────────────────────────────────────────────────────
  const s = {
    wrapper: {
      display: 'flex', flexDirection: 'column', height: '100vh',
      fontFamily: '"Inter", system-ui, -apple-system, sans-serif',
      background: '#070707', color: '#e2e8f0',
      overflow: 'hidden',
    },
    topBar: {
      display: 'flex', alignItems: 'center', gap: '0.875rem',
      padding: '0.7rem 1.5rem',
      borderBottom: '1px solid rgba(255,255,255,0.05)',
      background: 'rgba(9,9,9,0.98)',
      backdropFilter: 'blur(20px)',
      flexShrink: 0,
      zIndex: 10,
    },
    promptInput: {
      flex: 1, padding: '0.6rem 1rem', borderRadius: '10px',
      border: '1px solid rgba(255,255,255,0.08)',
      background: 'rgba(255,255,255,0.04)',
      color: '#e2e8f0',
      fontSize: '0.875rem', outline: 'none',
      transition: 'border-color 0.2s, box-shadow 0.2s',
      fontFamily: 'inherit',
    },
    select: {
      padding: '0.55rem 0.8rem', borderRadius: '10px',
      border: '1px solid rgba(255,255,255,0.08)',
      background: 'rgba(255,255,255,0.04)',
      color: '#94a3b8',
      fontSize: '0.8rem', cursor: 'pointer', outline: 'none',
      fontFamily: 'inherit',
    },
    btn: (disabled, color = '#6366f1') => ({
      padding: '0.6rem 1.3rem', borderRadius: '10px', border: 'none',
      background: disabled ? 'rgba(255,255,255,0.04)' : color,
      color: disabled ? '#374151' : '#fff',
      cursor: disabled ? 'not-allowed' : 'pointer',
      fontWeight: 600, fontSize: '0.85rem', whiteSpace: 'nowrap',
      transition: 'all 0.18s',
      boxShadow: disabled ? 'none' : `0 0 20px ${color}55`,
      fontFamily: 'inherit',
      letterSpacing: '0.01em',
    }),
    body: { display: 'flex', flex: 1, overflow: 'hidden' },
    sidebar: {
      width: '288px', flexShrink: 0, display: 'flex', flexDirection: 'column',
      borderRight: '1px solid rgba(255,255,255,0.05)',
      overflowY: 'auto', padding: '1rem', gap: '0.7rem',
      background: '#090909',
    },
    card: {
      background: 'rgba(255,255,255,0.025)',
      border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: '12px', padding: '0.9rem',
      animation: 'fadeIn 0.2s ease-out',
    },
    label: {
      fontSize: '0.63rem', fontWeight: 700, color: '#374151',
      textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: '0.5rem',
    },
    statusBadge: (color) => ({
      display: 'inline-block', padding: '0.22rem 0.7rem', borderRadius: '99px',
      fontSize: '0.71rem', fontWeight: 600,
      background: color + '18', color, border: `1px solid ${color}32`,
      letterSpacing: '0.03em',
    }),
    phaseBadge: {
      display: 'flex', alignItems: 'center', gap: '0.5rem',
      padding: '0.6rem 0.8rem', borderRadius: '10px',
      background: 'rgba(99,102,241,0.07)', border: '1px solid rgba(99,102,241,0.18)',
      fontSize: '0.82rem', color: '#a5b4fc',
    },
    spinner: {
      width: '13px', height: '13px',
      border: '2px solid rgba(99,102,241,0.15)',
      borderTop: '2px solid #818cf8', borderRadius: '50%',
      animation: 'spin 0.7s linear infinite', flexShrink: 0,
    },
    errorItem: {
      background: 'rgba(239,68,68,0.04)',
      border: '1px solid rgba(239,68,68,0.12)',
      borderRadius: '10px', padding: '0.65rem 0.75rem',
      fontSize: '0.75rem', lineHeight: 1.6, marginBottom: '0.4rem',
      wordBreak: 'break-word',
    },
    previewPane: {
      flex: 1, display: 'flex', flexDirection: 'column',
      position: 'relative', background: '#0a0a0a',
    },
    iframe: { flex: 1, border: 'none', width: '100%' },
    emptyPreview: {
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      height: '100%', gap: '1.25rem',
    },
  };

  const phaseLabel = jobState?.phase ? (PHASE_LABELS[jobState.phase] ?? `⚙️ ${jobState.phase}…`) : null;
  const phaseInfo  = jobState?.phase_info ?? {};
  const isCoding   = jobState?.phase === 'coder' && phaseInfo.total_steps > 0;

  return (
    <>
      {/* Global keyframes & interactive class styles */}
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }
        @keyframes pulse { 0%,100%{opacity:.5;transform:scale(1)} 50%{opacity:1;transform:scale(1.06)} }
        @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)} }
        .pi-input:focus { border-color: rgba(99,102,241,0.55) !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important; }
        .pi-select:focus { border-color: rgba(99,102,241,0.45) !important; }
        .pi-btn { transition: all 0.18s !important; }
        .pi-btn:hover:not(:disabled) { filter: brightness(1.14); transform: translateY(-1px); }
        .pi-btn:active:not(:disabled) { transform: translateY(0); filter: brightness(0.93); }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.07); border-radius: 99px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }
      `}</style>

      <div style={s.wrapper}>

        {/* ── top bar ── */}
        <div style={s.topBar}>
          {/* Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
            <div style={{
              width: '28px', height: '28px', borderRadius: '8px',
              background: 'linear-gradient(135deg, #6366f1 0%, #a78bfa 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.8rem', boxShadow: '0 0 14px rgba(99,102,241,0.5)',
            }}>✦</div>
            <span style={{
              fontSize: '0.92rem', fontWeight: 700, letterSpacing: '-0.02em',
              background: 'linear-gradient(135deg, #a78bfa, #60a5fa)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            }}>Lovable</span>
          </div>

          <form onSubmit={handleSubmit}
                style={{ display: 'flex', flex: 1, gap: '0.6rem', alignItems: 'center' }}>
            <input
              className="pi-input"
              style={s.promptInput}
              placeholder="Describe the app you want to build…"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={isBuilding}
            />
            <select
              className="pi-select"
              style={s.select}
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isBuilding}
            >
              {MODELS.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            <button
              type="submit"
              className="pi-btn"
              style={s.btn(isBuilding || !prompt.trim())}
              disabled={isBuilding || !prompt.trim()}
            >
              {isBuilding ? '⚙ Building…' : '✦ Build'}
            </button>
          </form>
        </div>

        {/* ── body ── */}
        <div style={s.body}>

          {/* ── sidebar ── */}
          <div style={s.sidebar}>

            {/* live phase indicator */}
            {isBuilding && phaseLabel && (
              <div style={{ ...s.card, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={s.spinner} />
                  <span style={{ color: '#a5b4fc', fontSize: '0.82rem' }}>{phaseLabel}</span>
                </div>

                {/* coder step progress */}
                {isCoding && (
                  <>
                    {/* step counter */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.72rem', color: '#666' }}>
                        Step {phaseInfo.step} / {phaseInfo.total_steps}
                      </span>
                      <span style={{ fontSize: '0.72rem', color: '#4ade80' }}>
                        {Math.round((phaseInfo.step / phaseInfo.total_steps) * 100)}%
                      </span>
                    </div>

                    {/* progress bar */}
                    <div style={{ height: '4px', background: '#2a2a2a', borderRadius: '99px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${(phaseInfo.step / phaseInfo.total_steps) * 100}%`,
                        background: 'linear-gradient(90deg, #6366f1, #a5b4fc)',
                        borderRadius: '99px',
                        transition: 'width 0.4s ease',
                      }} />
                    </div>

                    {/* current step description */}
                    {phaseInfo.step_description && (
                      <div style={{
                        fontSize: '0.72rem', color: '#888', fontStyle: 'italic',
                        overflow: 'hidden', textOverflow: 'ellipsis',
                        display: '-webkit-box', WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                      }}>
                        {phaseInfo.step_description}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {/* idle queued state */}
            {isBuilding && !phaseLabel && (
              <div style={{ ...s.phaseBadge, color: '#888', border: '1px solid #333' }}>
                <div style={s.spinner} />
                <span>Queued…</span>
              </div>
            )}

            {/* agent / final status */}
            {(isDone || isError) && (
              <div style={s.card}>
                <div style={s.label}>Result</div>
                <span style={s.statusBadge(agentStatusColor(jobState.agent_status ?? (isError ? 'ERROR' : null)))}>
                  {jobState.agent_status ?? (isError ? 'ERROR' : 'UNKNOWN')}
                </span>
                {jobState.error && (
                  <p style={{ color: '#f87171', fontSize: '0.78rem', marginTop: '0.5rem', wordBreak: 'break-word' }}>
                    {jobState.error}
                  </p>
                )}
              </div>
            )}

            {/* runtime errors from iframe */}
            {hasErrors && (
              <div style={s.card}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.45rem' }}>
                  <div style={s.label}>
                    Runtime Errors&nbsp;
                    <span style={{ background: '#7f1d1d', color: '#fca5a5', borderRadius: '99px', padding: '0.1rem 0.4rem', fontSize: '0.68rem' }}>
                      {errors.length}
                    </span>
                  </div>
                  <button onClick={() => setErrors([])}
                          style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', fontSize: '0.72rem' }}>
                    clear
                  </button>
                </div>

                {errors.slice(0, 4).map((err, i) => (
                  <div key={i} style={s.errorItem}>
                    <div style={{ color: '#f87171', fontWeight: 600 }}>{err.message}</div>
                    {err.source && (
                      <div style={{ color: '#555', fontSize: '0.7rem' }}>
                        {err.source}{err.lineno ? `:${err.lineno}` : ''}
                      </div>
                    )}
                    {err.stack && (
                      <div style={{ color: '#666', fontFamily: 'monospace', fontSize: '0.7rem', marginTop: '0.25rem', whiteSpace: 'pre-wrap' }}>
                        {err.stack.slice(0, 280)}
                      </div>
                    )}
                  </div>
                ))}

                <button
                  className="pi-btn"
                  onClick={handleAutoFix}
                  disabled={isFixing}
                  style={{ ...s.btn(isFixing, '#dc2626'), width: '100%', marginTop: '0.1rem' }}
                >
                  {isFixing ? '⚡ Fixing…' : '⚡ Auto-fix'}
                </button>

                {/* auto-fix status */}
                {isFixing && (
                  <p style={{ color: '#a5b4fc', fontSize: '0.74rem', marginTop: '0.4rem', textAlign: 'center' }}>
                    ⚡ Auto-fixing…
                  </p>
                )}
                {fixStatus === 'ok' && !isFixing && (
                  <p style={{ color: '#22c55e', fontSize: '0.76rem', marginTop: '0.4rem', textAlign: 'center' }}>
                    ✓ Fix applied — preview reloaded.
                  </p>
                )}
                {fixStatus === 'fail' && !isFixing && (
                  <p style={{ color: '#f87171', fontSize: '0.76rem', marginTop: '0.4rem', textAlign: 'center' }}>
                    ✗ Auto-fix failed. Try rephrasing.
                  </p>
                )}

                {/* auto-fix attempt log */}
                {autoFixLog.length > 0 && (
                  <div style={{ marginTop: '0.4rem', borderTop: '1px solid #2a2a2a', paddingTop: '0.4rem' }}>
                    {autoFixLog.slice(-3).map((line, i) => (
                      <div key={i} style={{ fontSize: '0.68rem', color: '#555', lineHeight: 1.6 }}>{line}</div>
                    ))}
                    {autoFixCount.current >= MAX_AUTO_FIXES && (
                      <div style={{ fontSize: '0.68rem', color: '#7f1d1d', marginTop: '0.2rem' }}>
                        Max auto-fix attempts reached.
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* preview URL + reload */}
            {previewUrl && (
              <div style={s.card}>
                <div style={s.label}>Preview URL</div>
                <a href={previewUrl} target="_blank" rel="noreferrer"
                   style={{ color: '#818cf8', fontSize: '0.78rem', wordBreak: 'break-all', display: 'block', marginBottom: '0.6rem' }}>
                  {previewUrl}
                </a>
                <div style={{ display: 'flex', gap: '0.4rem' }}>
                  <button
                    className="pi-btn"
                    onClick={() => {
                      if (iframeRef.current) {
                        const base = previewUrl.split('?')[0];
                        iframeRef.current.src = base + '?t=' + Date.now();
                      }
                    }}
                    style={{ ...s.btn(false, '#4f46e5'), flex: 1, fontSize: '0.78rem', padding: '0.45rem 0.6rem' }}
                  >
                    ↺ Reload
                  </button>
                  <a href={previewUrl} target="_blank" rel="noreferrer" style={{ flex: 1, textDecoration: 'none' }}>
                    <button className="pi-btn"
                      style={{ ...s.btn(false, '#0f766e'), width: '100%', fontSize: '0.78rem', padding: '0.45rem 0.6rem' }}>
                      ↗ Open tab
                    </button>
                  </a>
                </div>
              </div>
            )}

            {/* download project ZIP */}
            {projectId && isDone && (
              <div style={s.card}>
                <div style={s.label}>Download</div>
                <a
                  href={`${API_BASE}/api/download/${projectId}`}
                  download
                  style={{ textDecoration: 'none' }}
                >
                  <button className="pi-btn" style={{ ...s.btn(false, '#059669'), width: '100%' }}>
                    ⬇ Download ZIP
                  </button>
                </a>
              </div>
            )}

            {/* ── continue editing ── */}
            {isDone && projectId && (
              <div style={s.card}>
                <div style={s.label}>Continue Editing</div>
                <div style={{
                  fontSize: '0.7rem', color: '#4b5563',
                  background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.12)',
                  borderRadius: '6px', padding: '0.5rem 0.6rem', marginBottom: '0.6rem', lineHeight: 1.5,
                }}>
                  💡 Best for <strong style={{ color: '#6366f1' }}>small changes</strong> — color, text, layout tweaks.
                  For major features (new pages, new functions), use <strong style={{ color: '#6366f1' }}>Build</strong> instead with a detailed prompt.
                </div>

                {/* history of past edits */}
                {editHistory.length > 0 && (
                  <div style={{ marginBottom: '0.5rem' }}>
                    {editHistory.slice(-4).map((h, i) => (
                      <div key={i} style={{
                        fontSize: '0.72rem',
                        color: h.status === 'error' ? '#f87171' : '#555',
                        padding: '0.2rem 0',
                        borderBottom: '1px solid #1e1e1e',
                        wordBreak: 'break-word',
                      }}>
                        {h.status === 'running' ? '⏳' : h.status === 'done' ? '✓' : '✗'}&nbsp;{h.prompt}
                      </div>
                    ))}
                  </div>
                )}

                <form onSubmit={handleEdit}
                      style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  <textarea
                    value={editPrompt}
                    onChange={(e) => setEditPrompt(e.target.value)}
                    placeholder="Add a contact form, change the color scheme…"
                    disabled={isEditing}
                    rows={3}
                    style={{
                      width: '100%', padding: '0.6rem 0.75rem', borderRadius: '8px',
                      border: '1px solid rgba(255,255,255,0.08)',
                      background: 'rgba(255,255,255,0.03)', color: '#e2e8f0',
                      fontSize: '0.8rem', resize: 'vertical', outline: 'none',
                      fontFamily: 'inherit', boxSizing: 'border-box',
                      transition: 'border-color 0.2s',
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleEdit(e); }
                    }}
                  />
                  <button
                    type="submit"
                    className="pi-btn"
                    disabled={isEditing || !editPrompt.trim()}
                    style={{ ...s.btn(isEditing || !editPrompt.trim(), '#7c3aed'), width: '100%' }}
                  >
                    {isEditing ? '✏️ Editing…' : '✏️ Apply Edit'}
                  </button>
                </form>
              </div>
            )}

          </div>

          {/* ── preview pane ── */}
          <div style={s.previewPane}>
            {previewUrl ? (
              <>
                {/* red error strip above iframe */}
                {hasErrors && (
                  <div style={{
                    background: 'rgba(127,29,29,0.85)', color: '#fca5a5',
                    backdropFilter: 'blur(8px)',
                    borderBottom: '1px solid rgba(239,68,68,0.2)',
                    padding: '0.4rem 1rem', fontSize: '0.78rem',
                    display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0,
                  }}>
                    <span>⚠️</span>
                    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {errors[0]?.message}
                    </span>
                  </div>
                )}
                <iframe
                  ref={iframeRef}
                  src={previewUrl}
                  style={s.iframe}
                  title="App Preview"
                  sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
                />
              </>
            ) : (
              <div style={s.emptyPreview}>
                {/* Ambient glow orbs */}
                <div style={{
                  position: 'absolute', width: '320px', height: '320px', borderRadius: '50%',
                  background: 'radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%)',
                  pointerEvents: 'none',
                }} />
                <div style={{
                  width: '80px', height: '80px', borderRadius: '22px',
                  background: 'linear-gradient(135deg, rgba(99,102,241,0.12), rgba(167,139,250,0.12))',
                  border: '1px solid rgba(99,102,241,0.2)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '2.2rem',
                  animation: isBuilding ? 'pulse 2s ease-in-out infinite' : 'float 4s ease-in-out infinite',
                  boxShadow: '0 0 40px rgba(99,102,241,0.08)',
                  zIndex: 1,
                }}>
                  {isBuilding ? '⚙️' : '🚀'}
                </div>
                <div style={{ textAlign: 'center', zIndex: 1 }}>
                  <div style={{ fontSize: '1rem', fontWeight: 600, color: '#334155', marginBottom: '0.4rem' }}>
                    {isBuilding ? (phaseLabel ?? 'Starting…') : 'Your preview will appear here'}
                  </div>
                  {!isBuilding && (
                    <div style={{ fontSize: '0.8rem', color: '#1e293b' }}>
                      Describe an app above and click Build to get started
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

        </div>
      </div>
    </>
  );
}
