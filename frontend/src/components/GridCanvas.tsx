import { useRef, useEffect, useCallback, useState } from 'react';
import { useGridStore } from '../store/gridStore';
import { useSessionStore } from '../store/sessionStore';
import { useUIStore } from '../store/uiStore';
import type { CellType } from '../store/gridStore';
import type { OverlayType } from '../store/uiStore';

const CELL_COLORS: Record<string, string> = {
  '.': '#ffffff',
  '#': '#1e293b',
  S: '#3b82f6',
  G: '#22c55e',
  H: '#ef4444',
  M: '#a16207',
};

const CELL_COLORS_DARK: Record<string, string> = {
  '.': '#0f172a',
  '#': '#475569',
  S: '#60a5fa',
  G: '#4ade80',
  H: '#f87171',
  M: '#ca8a04',
};

const AGENT_COLORS: Record<string, string> = {
  q: '#3b82f6',
  s: '#ef4444',
  e: '#22c55e',
};

const ACTION_DX = [0, 1, 0, -1]; // up, right, down, left
const ACTION_DY = [-1, 0, 1, 0];

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function divergingColor(value: number, min: number, max: number): string {
  if (max === min) return 'rgba(200,200,200,0.5)';
  const t = (value - min) / (max - min);
  const r = Math.round(lerp(59, 239, t));
  const g = Math.round(lerp(130, 68, t < 0.5 ? t * 2 : (1 - t) * 2));
  const b = Math.round(lerp(246, 68, t));
  return `rgba(${r},${g},${b},0.45)`;
}

function heatmapColor(value: number, max: number): string {
  if (max === 0) return 'rgba(200,200,200,0.3)';
  const t = Math.min(value / max, 1);
  const r = Math.round(255 * t);
  const g = Math.round(255 * (1 - t) * 0.5);
  const b = Math.round(60 * (1 - t));
  return `rgba(${r},${g},${b},0.5)`;
}

export function GridCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverCell, setHoverCell] = useState<{ r: number; c: number } | null>(null);
  const [canvasSize, setCanvasSize] = useState({ w: 600, h: 600 });

  const rows = useGridStore((s) => s.rows);
  const nRows = useGridStore((s) => s.nRows);
  const nCols = useGridStore((s) => s.nCols);
  const stagedEdits = useGridStore((s) => s.stagedEdits);
  const setCell = useGridStore((s) => s.setCell);

  const activeTool = useUIStore((s) => s.activeTool);
  const isPainting = useUIStore((s) => s.isPainting);
  const setIsPainting = useUIStore((s) => s.setIsPainting);
  const selectedOverlay = useUIStore((s) => s.selectedOverlay);
  const selectedAgent = useUIStore((s) => s.selectedAgent);
  const darkMode = useUIStore((s) => s.darkMode);

  const agentSnapshots = useSessionStore((s) => s.agentSnapshots);
  const optimalSnapshot = useSessionStore((s) => s.optimalSnapshot);
  const agentConfigs = useSessionStore((s) => s.agentConfigs);
  const trainingState = useSessionStore((s) => s.state);

  const cellSize = Math.floor(Math.min(canvasSize.w / nCols, canvasSize.h / nRows));

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setCanvasSize({ w: width, h: height });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const getCellAt = useCallback(
    (clientX: number, clientY: number): { r: number; c: number } | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      const c = Math.floor(x / cellSize);
      const r = Math.floor(y / cellSize);
      if (r < 0 || r >= nRows || c < 0 || c >= nCols) return null;
      return { r, c };
    },
    [cellSize, nRows, nCols]
  );

  const paint = useCallback(
    (r: number, c: number) => {
      if (trainingState === 'running' || trainingState === 'paused') {
        const store = useGridStore.getState();
        store.stageEdit(r, c, activeTool);
      } else {
        setCell(r, c, activeTool);
      }
    },
    [activeTool, setCell, trainingState]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const cell = getCellAt(e.clientX, e.clientY);
      if (!cell) return;
      setIsPainting(true);
      paint(cell.r, cell.c);
    },
    [getCellAt, paint, setIsPainting]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const cell = getCellAt(e.clientX, e.clientY);
      setHoverCell(cell);
      if (isPainting && cell) {
        paint(cell.r, cell.c);
      }
    },
    [getCellAt, isPainting, paint]
  );

  const handleMouseUp = useCallback(() => {
    setIsPainting(false);
  }, [setIsPainting]);

  const handleMouseLeave = useCallback(() => {
    setIsPainting(false);
    setHoverCell(null);
  }, [setIsPainting]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = cellSize * nCols;
    const H = cellSize * nRows;
    canvas.width = W;
    canvas.height = H;

    const colors = darkMode ? CELL_COLORS_DARK : CELL_COLORS;

    // Clear
    ctx.fillStyle = darkMode ? '#0f172a' : '#f8fafc';
    ctx.fillRect(0, 0, W, H);

    // Cells
    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const cell = rows[r]?.[c] ?? '.';
        const x = c * cellSize;
        const y = r * cellSize;
        ctx.fillStyle = colors[cell] ?? colors['.'];
        ctx.fillRect(x, y, cellSize, cellSize);
      }
    }

    // Staged edits (ghost)
    for (const edit of stagedEdits) {
      const x = edit.c * cellSize;
      const y = edit.r * cellSize;
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = colors[edit.to] ?? colors['.'];
      ctx.fillRect(x, y, cellSize, cellSize);
      ctx.globalAlpha = 1;
      // Diagonal stripes to indicate staged
      ctx.strokeStyle = darkMode ? '#94a3b8' : '#64748b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + cellSize, y + cellSize);
      ctx.moveTo(x + cellSize / 2, y);
      ctx.lineTo(x + cellSize, y + cellSize / 2);
      ctx.moveTo(x, y + cellSize / 2);
      ctx.lineTo(x + cellSize / 2, y + cellSize);
      ctx.stroke();
    }

    // Overlays
    drawOverlay(ctx, selectedOverlay, selectedAgent, agentSnapshots, optimalSnapshot, agentConfigs, nRows, nCols, cellSize, darkMode);

    // Grid lines
    ctx.strokeStyle = darkMode ? '#334155' : '#e2e8f0';
    ctx.lineWidth = 1;
    for (let r = 0; r <= nRows; r++) {
      ctx.beginPath();
      ctx.moveTo(0, r * cellSize);
      ctx.lineTo(W, r * cellSize);
      ctx.stroke();
    }
    for (let c = 0; c <= nCols; c++) {
      ctx.beginPath();
      ctx.moveTo(c * cellSize, 0);
      ctx.lineTo(c * cellSize, H);
      ctx.stroke();
    }

    // Cell labels for S, G
    ctx.font = `bold ${Math.max(10, cellSize * 0.4)}px system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const cell = rows[r]?.[c] ?? '.';
        if (cell === 'S' || cell === 'G' || cell === 'H' || cell === 'M') {
          ctx.fillStyle = '#fff';
          ctx.fillText(cell, c * cellSize + cellSize / 2, r * cellSize + cellSize / 2);
        }
      }
    }

    // Hover highlight
    if (hoverCell) {
      ctx.strokeStyle = darkMode ? '#fbbf24' : '#f59e0b';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        hoverCell.c * cellSize + 1,
        hoverCell.r * cellSize + 1,
        cellSize - 2,
        cellSize - 2
      );
    }
  }, [rows, nRows, nCols, cellSize, stagedEdits, hoverCell, selectedOverlay, selectedAgent, agentSnapshots, optimalSnapshot, agentConfigs, darkMode]);

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[300px] flex items-center justify-center">
      <canvas
        ref={canvasRef}
        className="cursor-crosshair"
        style={{ maxWidth: '100%', maxHeight: '100%' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      {hoverCell && (
        <div
          className="absolute top-2 right-2 bg-white/95 dark:bg-slate-800/95 text-xs px-3 py-2 rounded shadow-lg pointer-events-none space-y-1 min-w-[180px]"
        >
          <div className="font-medium text-slate-700 dark:text-slate-200">
            ({hoverCell.r}, {hoverCell.c}) {rows[hoverCell.r]?.[hoverCell.c] ?? '?'}
          </div>
          <QPopover state={hoverCell.r * nCols + hoverCell.c} agentSnapshots={agentSnapshots} optimalSnapshot={optimalSnapshot} />
        </div>
      )}
    </div>
  );
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  overlay: OverlayType,
  selectedAgent: string,
  agentSnapshots: Record<string, any>,
  optimalSnapshot: any,
  agentConfigs: any[],
  nRows: number,
  nCols: number,
  cellSize: number,
  darkMode: boolean
) {
  if (overlay === 'none') return;

  const snap = agentSnapshots[selectedAgent];
  const nStates = nRows * nCols;

  if (overlay === 'q_heatmap' && snap?.q) {
    const maxQ = snap.q.map((row: number[]) => Math.max(...row));
    const minV = Math.min(...maxQ);
    const maxV = Math.max(...maxQ);
    for (let s = 0; s < nStates; s++) {
      const r = Math.floor(s / nCols);
      const c = s % nCols;
      ctx.fillStyle = divergingColor(maxQ[s], minV, maxV);
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
    }
  }

  if (overlay === 'policy' && snap?.greedy_policy) {
    const optPolicy = optimalSnapshot?.policy;
    drawArrows(ctx, snap.greedy_policy, optPolicy, nRows, nCols, cellSize, AGENT_COLORS[selectedAgent] ?? '#3b82f6');
  }

  if (overlay === 'visits' && snap?.visits) {
    const maxV = Math.max(...snap.visits, 1);
    for (let s = 0; s < nStates; s++) {
      const r = Math.floor(s / nCols);
      const c = s % nCols;
      ctx.fillStyle = heatmapColor(snap.visits[s], maxV);
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
    }
  }

  if (overlay === 'td_error' && snap?.td_ema) {
    const maxV = Math.max(...snap.td_ema, 0.01);
    for (let s = 0; s < nStates; s++) {
      const r = Math.floor(s / nCols);
      const c = s % nCols;
      ctx.fillStyle = heatmapColor(snap.td_ema[s], maxV);
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
    }
  }

  if (overlay === 'optimal' && optimalSnapshot?.v) {
    const V = optimalSnapshot.v;
    const minV = Math.min(...V);
    const maxV = Math.max(...V);
    for (let s = 0; s < nStates; s++) {
      const r = Math.floor(s / nCols);
      const c = s % nCols;
      ctx.fillStyle = divergingColor(V[s], minV, maxV);
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
    }
    if (optimalSnapshot.policy) {
      const flatPolicy = optimalSnapshot.policy.map((acts: number[]) => acts[0] ?? 0);
      drawArrows(ctx, flatPolicy, null, nRows, nCols, cellSize, '#a855f7');
    }
  }

  if (overlay === 'disagreement') {
    const allPolicies = agentConfigs
      .map((cfg: any) => agentSnapshots[cfg.id]?.greedy_policy)
      .filter(Boolean);
    if (allPolicies.length >= 2) {
      for (let s = 0; s < nStates; s++) {
        const actions = new Set(allPolicies.map((p: number[]) => p[s]));
        if (actions.size > 1) {
          const r = Math.floor(s / nCols);
          const c = s % nCols;
          ctx.fillStyle = 'rgba(249, 115, 22, 0.4)';
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
          ctx.strokeStyle = '#f97316';
          ctx.lineWidth = 2;
          ctx.strokeRect(c * cellSize + 2, r * cellSize + 2, cellSize - 4, cellSize - 4);
        }
      }
    }
  }
}

function drawArrows(
  ctx: CanvasRenderingContext2D,
  policy: number[],
  optimalPolicy: number[][] | null,
  nRows: number,
  nCols: number,
  cellSize: number,
  defaultColor: string
) {
  const arrowLen = cellSize * 0.3;
  const headLen = cellSize * 0.1;
  const nStates = nRows * nCols;

  for (let s = 0; s < Math.min(policy.length, nStates); s++) {
    const r = Math.floor(s / nCols);
    const c = s % nCols;
    const a = policy[s];
    if (a === undefined || a === null) continue;

    const cx = c * cellSize + cellSize / 2;
    const cy = r * cellSize + cellSize / 2;
    const dx = ACTION_DX[a] * arrowLen;
    const dy = ACTION_DY[a] * arrowLen;

    let color = defaultColor;
    if (optimalPolicy && optimalPolicy[s]) {
      color = optimalPolicy[s].includes(a) ? '#22c55e' : '#ef4444';
    }

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;

    // Shaft
    ctx.beginPath();
    ctx.moveTo(cx - dx * 0.5, cy - dy * 0.5);
    ctx.lineTo(cx + dx * 0.5, cy + dy * 0.5);
    ctx.stroke();

    // Head
    const tipX = cx + dx * 0.5;
    const tipY = cy + dy * 0.5;
    const angle = Math.atan2(dy, dx);
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(
      tipX - headLen * Math.cos(angle - Math.PI / 6),
      tipY - headLen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      tipX - headLen * Math.cos(angle + Math.PI / 6),
      tipY - headLen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
  }
}

const ACTIONS = ['Up', 'Right', 'Down', 'Left'];
const Q_AGENT_NAMES: Record<string, string> = { q: 'Q-Lrn', s: 'SARSA', e: 'E-SARSA' };
const Q_AGENT_COLORS: Record<string, string> = { q: '#3b82f6', s: '#ef4444', e: '#22c55e' };

function QPopover({ state, agentSnapshots, optimalSnapshot }: { state: number; agentSnapshots: Record<string, any>; optimalSnapshot: any }) {
  const agents = Object.entries(agentSnapshots);
  if (agents.length === 0) return null;

  const optActions = optimalSnapshot?.policy?.[state] as number[] | undefined;

  return (
    <div className="space-y-1">
      {agents.map(([aid, snap]) => {
        const q = snap?.q?.[state] as number[] | undefined;
        if (!q) return null;
        const maxA = q.indexOf(Math.max(...q));
        return (
          <div key={aid}>
            <div className="font-medium" style={{ color: Q_AGENT_COLORS[aid] }}>{Q_AGENT_NAMES[aid] || aid}</div>
            <div className="grid grid-cols-4 gap-x-2 tabular-nums text-[10px] text-slate-600 dark:text-slate-400">
              {ACTIONS.map((name, i) => (
                <span key={i} className={`${i === maxA ? 'font-bold text-slate-800 dark:text-slate-100' : ''} ${optActions?.includes(i) ? 'underline' : ''}`}>
                  {name}: {q[i].toFixed(1)}
                </span>
              ))}
            </div>
          </div>
        );
      })}
      {optActions && (
        <div className="text-[10px] text-purple-500 dark:text-purple-400">
          Optimal: {optActions.map((a: number) => ACTIONS[a]).join(', ')}
        </div>
      )}
    </div>
  );
}
