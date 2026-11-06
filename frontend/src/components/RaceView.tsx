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

function MiniGrid({ rollout, nRows, nCols, rows, step, color }: {
  rollout: RolloutResult | null; nRows: number; nCols: number; rows: string[][]; step: number; color: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = Math.min(canvas.width / nCols, canvas.height / nRows);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const cell = rows[r]?.[c] ?? '.';
        ctx.fillStyle = CELL_COLORS[cell] || '#f1f5f9';
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }

    if (rollout && step >= 0 && step < rollout.steps.length) {
      const s = rollout.steps[step].s;
      const r = Math.floor(s / nCols);
      const c = s % nCols;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(c * cellSize + cellSize / 2, r * cellSize + cellSize / 2, cellSize * 0.35, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [rollout, nRows, nCols, rows, step, color]);

  return <canvas ref={canvasRef} width={200} height={200} className="border border-slate-200 dark:border-slate-700 rounded" />;
}

export default function RaceView() {
  const sessionId = useSessionStore((s) => s.sessionId);
  const agentConfigs = useSessionStore((s) => s.agentConfigs);
  const nRows = useGridStore((s) => s.nRows);
  const nCols = useGridStore((s) => s.nCols);
  const rows = useGridStore((s) => s.rows);

  const [rollouts, setRollouts] = useState<Record<string, RolloutResult>>({});
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [greedy, setGreedy] = useState(true);
  const timerRef = useRef<number>();

  const maxSteps = Math.max(...Object.values(rollouts).map((r) => r.steps.length), 0);

  const fetchRollouts = useCallback(async () => {
    if (!sessionId) return;
    const results: Record<string, RolloutResult> = {};
    for (const cfg of agentConfigs) {
      try {
        results[cfg.id] = await api.rollout(sessionId, cfg.id, greedy, 0);
      } catch { /* skip */ }
    }
    setRollouts(results);
    setStep(0);
  }, [sessionId, agentConfigs, greedy]);

  useEffect(() => {
    if (playing && step < maxSteps - 1) {
      timerRef.current = window.setTimeout(() => setStep((s) => s + 1), 200);
    } else if (step >= maxSteps - 1) {
      setPlaying(false);
    }
    return () => clearTimeout(timerRef.current);
  }, [playing, step, maxSteps]);

  return (
    <div className="p-3">
      <div className="flex items-center gap-3 mb-3">
        <button onClick={fetchRollouts} className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600">
          Fetch Rollouts
        </button>
        <button onClick={() => { setPlaying(!playing); }} className="px-3 py-1 text-sm bg-slate-200 dark:bg-slate-700 rounded">
          {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={() => setStep(0)} className="px-3 py-1 text-sm bg-slate-200 dark:bg-slate-700 rounded">
          Reset
        </button>
        <label className="flex items-center gap-1 text-sm text-slate-600 dark:text-slate-400">
          <input type="checkbox" checked={greedy} onChange={(e) => setGreedy(e.target.checked)} />
          Greedy
        </label>
        <input
          type="range" min={0} max={Math.max(maxSteps - 1, 0)} value={step}
          onChange={(e) => setStep(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-xs tabular-nums text-slate-500">{step}/{maxSteps}</span>
      </div>
      <div className="flex gap-4 justify-center">
        {agentConfigs.map((cfg) => {
          const r = rollouts[cfg.id];
          return (
            <div key={cfg.id} className="text-center">
              <MiniGrid rollout={r} nRows={nRows} nCols={nCols} rows={rows} step={step} color={AGENT_COLORS[cfg.id] || '#666'} />
              <div className="mt-1 text-xs font-medium" style={{ color: AGENT_COLORS[cfg.id] }}>{AGENT_NAMES[cfg.id]}</div>
              {r && (
                <div className="text-xs text-slate-500 tabular-nums">
                  Steps: {r.steps.length} | Return: {r.total_return.toFixed(1)} | {r.success ? 'Goal' : 'No goal'}
                </div>
              )}
            </div>
          );
        })}
      </div>
      {Object.values(rollouts).some(r => r.optimal_steps != null) && (
        <div className="text-center text-xs text-slate-400 mt-2">
          Optimal: {Object.values(rollouts)[0]?.optimal_steps ?? '?'} steps
        </div>
      )}
    </div>
  );
}
