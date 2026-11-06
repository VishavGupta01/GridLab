import { useSessionStore } from '../store/sessionStore';

const AGENT_NAMES: Record<string, string> = { q: 'Q-Learning', s: 'SARSA', e: 'Expected SARSA' };

export default function AdaptationPanel() {
  const events = useSessionStore((s) => s.events);
  const evalHistory = useSessionStore((s) => s.evalHistory);
  const agentConfigs = useSessionStore((s) => s.agentConfigs);

  if (events.length === 0) {
    return (
      <div className="p-6 text-center text-slate-400 dark:text-slate-500">
        No grid changes yet. Edit the grid during training to see adaptation metrics.
      </div>
    );
  }

  return (
    <div className="p-3 overflow-x-auto">
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
