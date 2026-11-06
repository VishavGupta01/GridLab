import { api } from '../api/client';
import { useGridStore } from '../store/gridStore';
import { useSessionStore } from '../store/sessionStore';

const AGENT_NAMES: Record<string, string> = { q: 'Q-Learning', s: 'SARSA', e: 'Expected SARSA' };

interface Disruption {
  label: string;
  description: string;
  edits: { r: number; c: number; to: string }[];
}

function getDisruptions(nRows: number, nCols: number): Disruption[] {
  return [
    {
      label: 'Open Shortcut',
      description: 'Remove a wall to create a shortcut',
      edits: [{ r: Math.floor(nRows / 2), c: Math.floor(nCols / 2), to: '.' }],
    },
    {
      label: 'Block Corridor',
      description: 'Place a wall to block a corridor',
      edits: [{ r: 1, c: Math.floor(nCols / 2), to: '#' }],
    },
    {
      label: 'Move Goal',
      description: 'Move goal to a different position',
      edits: [
        { r: nRows - 1, c: nCols - 1, to: '.' },
        { r: 0, c: nCols - 1, to: 'G' },
      ],
    },
    {
      label: 'Add Hazard',
      description: 'Place a hazard near the goal',
      edits: [{ r: Math.max(0, nRows - 2), c: nCols - 1, to: 'H' }],
    },
  ];
}

export default function AdaptationPanel() {
  const events = useSessionStore((s) => s.events);
  const evalHistory = useSessionStore((s) => s.evalHistory);
  const agentConfigs = useSessionStore((s) => s.agentConfigs);
  const sessionId = useSessionStore((s) => s.sessionId);
  const trainingState = useSessionStore((s) => s.state);
  const nRows = useGridStore((s) => s.nRows);
  const nCols = useGridStore((s) => s.nCols);

  const disruptions = getDisruptions(nRows, nCols);

  const applyDisruption = async (d: Disruption) => {
    if (!sessionId) return;
    await api.edit(sessionId, d.edits);
  };

  if (events.length === 0) {
    return (
      <div className="p-3">
        {(trainingState === 'running' || trainingState === 'paused') && (
          <div className="mb-4">
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Scripted Disruptions</h3>
            <div className="flex flex-wrap gap-2">
              {disruptions.map((d) => (
                <button
                  key={d.label}
                  onClick={() => applyDisruption(d)}
                  className="px-3 py-1.5 text-xs rounded bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-800/50"
                  title={d.description}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="text-center text-slate-400 dark:text-slate-500 py-4">
          No grid changes yet. Edit the grid during training to see adaptation metrics.
        </div>
      </div>
    );
  }

  return (
    <div className="p-3 overflow-x-auto">
      {(trainingState === 'running' || trainingState === 'paused') && (
        <div className="mb-3 flex flex-wrap gap-2">
          {disruptions.map((d) => (
            <button
              key={d.label}
              onClick={() => applyDisruption(d)}
              className="px-3 py-1 text-xs rounded bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-800/50"
              title={d.description}
            >
              {d.label}
            </button>
          ))}
        </div>
      )}
      <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Grid Change Events</h3>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-slate-200 dark:border-slate-700">
            <th className="text-left py-2 px-2">Episode</th>
            <th className="text-left py-2 px-2">Grid Version</th>
            <th className="text-left py-2 px-2">Edits</th>
            {agentConfigs.map((a) => (
              <th key={a.id} className="text-right py-2 px-2">{AGENT_NAMES[a.id] || a.id} Gap After</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {events.map((evt, i) => (
            <tr key={i} className="border-b border-slate-100 dark:border-slate-800">
              <td className="py-2 px-2 tabular-nums">{evt.episode}</td>
              <td className="py-2 px-2">v{evt.grid_version}</td>
              <td className="py-2 px-2 text-xs">{evt.edits.map((e) => `(${e.r},${e.c})->${e.to}`).join(', ')}</td>
              {agentConfigs.map((a) => {
                const evals = evalHistory[a.id] ?? [];
                const afterEval = evals.find((e) => e.episode >= evt.episode);
                return (
                  <td key={a.id} className="text-right py-2 px-2 tabular-nums">
                    {afterEval ? afterEval.metrics.gap.toFixed(2) : '-'}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
