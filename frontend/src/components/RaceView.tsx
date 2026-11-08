import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { RolloutResult } from '../api/client';
import { useGridStore } from '../store/gridStore';
import { useSessionStore } from '../store/sessionStore';

const AGENT_COLORS: Record<string, string> = { q: '#3b82f6', s: '#ef4444', e: '#22c55e' };
const AGENT_NAMES: Record<string, string> = { q: 'Q-Learning', s: 'SARSA', e: 'Expected SARSA' };
const CELL_COLORS: Record<string, string> = {
  '.': '#f1f5f9', '#': '#1e293b', S: '#93c5fd', G: '#86efac', H: '#fca5a5', M: '#d97706',
};
const CELL_COLORS_DARK: Record<string, string> = {
  '.': '#1e293b', '#': '#475569', S: '#3b82f6', G: '#4ade80', H: '#f87171', M: '#ca8a04',
};

function RaceMiniGrid({ rollout, nRows, nCols, rows, step, color, dark }: {
  rollout: RolloutResult | null; nRows: number; nCols: number; rows: string[][]; step: number; color: string; dark: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = Math.min(canvas.width / nCols, canvas.height / nRows);
    const colors = dark ? CELL_COLORS_DARK : CELL_COLORS;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const cell = rows[r]?.[c] ?? '.';
        ctx.fillStyle = colors[cell] || colors['.'];
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        ctx.strokeStyle = dark ? '#334155' : '#cbd5e1';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }

    if (rollout && rollout.steps.length > 0) {
      const trail = rollout.steps.slice(0, step + 1);
      for (let i = 0; i < trail.length; i++) {
        const s = trail[i].s;
        const tr = Math.floor(s / nCols);
        const tc = s % nCols;
        const opacity = 0.1 + 0.4 * (i / Math.max(trail.length - 1, 1));
        ctx.fillStyle = color;
        ctx.globalAlpha = opacity;
        ctx.fillRect(tc * cellSize + 1, tr * cellSize + 1, cellSize - 2, cellSize - 2);
      }
      ctx.globalAlpha = 1;

      if (step >= 0 && step < rollout.steps.length) {
        const s = rollout.steps[step].s;
        const ar = Math.floor(s / nCols);
        const ac = s % nCols;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(ac * cellSize + cellSize / 2, ar * cellSize + cellSize / 2, cellSize * 0.35, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.font = `bold ${Math.max(8, cellSize * 0.3)}px system-ui`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(step + 1), ac * cellSize + cellSize / 2, ar * cellSize + cellSize / 2);
      }

      if (trail.length >= 2) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        for (let i = 0; i < trail.length; i++) {
          const s = trail[i].s;
          const tr = Math.floor(s / nCols);
          const tc = s % nCols;
          const cx = tc * cellSize + cellSize / 2;
          const cy = tr * cellSize + cellSize / 2;
          if (i === 0) ctx.moveTo(cx, cy);
          else ctx.lineTo(cx, cy);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }
  }, [rollout, nRows, nCols, rows, step, color, dark]);

  return <canvas ref={canvasRef} width={300} height={300} className="rounded-lg" />;
}

export default function RaceView() {
  const sessionId = useSessionStore((s) => s.sessionId);
  const agentConfigs = useSessionStore((s) => s.agentConfigs);
  const episode = useSessionStore((s) => s.episode);
  const nRows = useGridStore((s) => s.nRows);
  const nCols = useGridStore((s) => s.nCols);
  const rows = useGridStore((s) => s.rows);
  const dark = false;

  const [rollouts, setRollouts] = useState<Record<string, RolloutResult>>({});
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [greedy, setGreedy] = useState(true);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef<number>();
  const prevEpisodeRef = useRef(0);

  const maxSteps = Math.max(...Object.values(rollouts).map((r) => r.steps.length), 0);

  const fetchRollouts = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    const results: Record<string, RolloutResult> = {};
    for (const cfg of agentConfigs) {
      try {
        results[cfg.id] = await api.rollout(sessionId, cfg.id, greedy, 0);
      } catch { /* skip */ }
    }
    setRollouts(results);
    setStep(0);
    setPlaying(false);
    setLoading(false);
  }, [sessionId, agentConfigs, greedy]);

  useEffect(() => {
    if (sessionId && episode > 0 && episode - prevEpisodeRef.current >= 50) {
      prevEpisodeRef.current = episode;
      fetchRollouts();
    }
  }, [episode, sessionId, fetchRollouts]);

  useEffect(() => {
    if (playing && step < maxSteps - 1) {
      timerRef.current = window.setTimeout(() => setStep((s) => s + 1), 150);
    } else if (step >= maxSteps - 1 && playing) {
      setPlaying(false);
    }
    return () => clearTimeout(timerRef.current);
  }, [playing, step, maxSteps]);

  const winner = (() => {
    const finished = Object.entries(rollouts)
      .filter(([, r]) => r.success)
      .sort((a, b) => a[1].steps.length - b[1].steps.length);
    return finished.length > 0 ? finished[0][0] : null;
  })();

  const hasRollouts = Object.keys(rollouts).length > 0;

  if (!sessionId) {
    return (
      <div className="flex items-center justify-center py-16 text-slate-400 dark:text-slate-500">
        Start training to see agent races.
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={fetchRollouts}
          disabled={loading}
          className="px-4 py-2 text-sm font-medium bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Loading...' : 'Fetch Rollouts'}
        </button>

        {hasRollouts && (
          <>
            <button
              onClick={() => { if (step >= maxSteps - 1) setStep(0); setPlaying(!playing); }}
              className="px-4 py-2 text-sm font-medium bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
            >
              {playing ? 'Pause' : 'Play'}
            </button>
            <button
              onClick={() => { setStep(0); setPlaying(false); }}
              className="px-3 py-2 text-sm bg-slate-200 dark:bg-slate-700 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
            >
              Reset
            </button>
          </>
        )}

        <label className="flex items-center gap-1.5 text-sm text-slate-600 dark:text-slate-400 ml-2">
          <input type="checkbox" checked={greedy} onChange={(e) => setGreedy(e.target.checked)} className="rounded" />
          Greedy policy
        </label>

        {hasRollouts && (
          <div className="flex-1 flex items-center gap-2">
            <input
              type="range" min={0} max={Math.max(maxSteps - 1, 0)} value={step}
              onChange={(e) => { setStep(Number(e.target.value)); setPlaying(false); }}
              className="flex-1 accent-blue-500"
            />
            <span className="text-sm font-mono text-slate-500 w-16 text-right">
              {step + 1} / {maxSteps}
            </span>
          </div>
        )}
      </div>

      {hasRollouts && (
        <div className="flex gap-6 justify-center">
          {agentConfigs.map((cfg) => {
            const r = rollouts[cfg.id];
            const isWinner = cfg.id === winner;
            const color = AGENT_COLORS[cfg.id] || '#666';
            const isDone = r && step >= r.steps.length - 1;

            return (
              <div key={cfg.id} className="flex flex-col items-center">
                <div
                  className={`rounded-xl p-1 transition-all ${
                    isWinner ? 'ring-3 ring-yellow-400 shadow-lg shadow-yellow-400/20' : 'ring-1 ring-slate-200 dark:ring-slate-700'
                  }`}
                  style={isWinner ? { borderColor: color } : {}}
                >
                  <RaceMiniGrid
                    rollout={r} nRows={nRows} nCols={nCols} rows={rows}
                    step={step} color={color} dark={dark}
                  />
                </div>

                <div className="mt-2 text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
                    <span className="text-sm font-semibold" style={{ color }}>
                      {AGENT_NAMES[cfg.id]}
                    </span>
                    {isWinner && <span className="text-xs bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300 px-1.5 py-0.5 rounded-full font-medium">Fastest</span>}
                  </div>

                  {r && (
                    <div className="mt-1 flex gap-3 justify-center text-xs text-slate-500 dark:text-slate-400 tabular-nums">
                      <span>{r.steps.length} steps</span>
                      <span>R = {r.total_return.toFixed(1)}</span>
                      <span className={`font-medium ${r.success ? 'text-green-600 dark:text-green-400' : 'text-red-500 dark:text-red-400'}`}>
                        {isDone ? (r.success ? 'Goal!' : 'Timeout') : `Step ${Math.min(step + 1, r.steps.length)}`}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {hasRollouts && rollouts[Object.keys(rollouts)[0]]?.optimal_steps != null && (
        <div className="text-center text-xs text-slate-400 dark:text-slate-500 mt-3">
          Optimal path: {rollouts[Object.keys(rollouts)[0]]?.optimal_steps} steps
        </div>
      )}
    </div>
  );
}
