import { useSessionStore } from '../store/sessionStore';

const AGENT_NAMES: Record<string, string> = {
  q: 'Q-Learning',
  s: 'SARSA',
  e: 'Expected SARSA',
};

const AGENT_COLORS: Record<string, string> = {
  q: '#3b82f6',
  s: '#ef4444',
  e: '#22c55e',
};

export default function Scorecard() {
  const evalHistory = useSessionStore((s) => s.evalHistory);
  const onlineHistory = useSessionStore((s) => s.onlineHistory);
  const agentConfigs = useSessionStore((s) => s.agentConfigs);

  const rows = agentConfigs.map((cfg) => {
    const evals = evalHistory[cfg.id] ?? [];
    const online = onlineHistory[cfg.id] ?? [];
    const lastEval = evals.length > 0 ? evals[evals.length - 1].metrics : null;
    const last100 = online.slice(-100);
    const avgReturn = last100.length > 0
      ? last100.reduce((acc: number, m: any) => acc + (m.return ?? 0), 0) / last100.length
      : 0;
    const hazardHits = online.reduce((acc: number, m: any) => acc + (m.hazard_hits ?? 0), 0);

    return {
      id: cfg.id,
      algo: AGENT_NAMES[cfg.id] || cfg.algo,
      color: AGENT_COLORS[cfg.id] || '#666',
      gap: lastEval?.gap ?? '-',
      gapPct: lastEval?.gap_pct ?? '-',
      pathRatio: lastEval?.path_ratio ?? '-',
      pctOptimal: lastEval?.pct_optimal ?? '-',
      epsToNearOptimal: lastEval?.episodes_to_near_optimal ?? '-',
      avgReturn: avgReturn.toFixed(1),
      hazardHits,
    };
  });

  const fmt = (v: any, decimals = 2) => {
    if (v === '-' || v === null || v === undefined) return '-';
    return typeof v === 'number' ? v.toFixed(decimals) : v;
  };

  return (
    <div className="p-3 overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-slate-200 dark:border-slate-700">
            <th className="text-left py-2 px-2 text-slate-600 dark:text-slate-400">Agent</th>
            <th className="text-right py-2 px-2 text-slate-600 dark:text-slate-400">Gap</th>
            <th className="text-right py-2 px-2 text-slate-600 dark:text-slate-400">Gap %</th>
            <th className="text-right py-2 px-2 text-slate-600 dark:text-slate-400">Path Ratio</th>
            <th className="text-right py-2 px-2 text-slate-600 dark:text-slate-400">% Optimal</th>
            <th className="text-right py-2 px-2 text-slate-600 dark:text-slate-400">Eps to Near-Opt</th>
            <th className="text-right py-2 px-2 text-slate-600 dark:text-slate-400">Avg Return (100)</th>
            <th className="text-right py-2 px-2 text-slate-600 dark:text-slate-400">Hazard Hits</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="border-b border-slate-100 dark:border-slate-800">
              <td className="py-2 px-2 flex items-center gap-2">
                <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: row.color }} />
                <span className="font-medium text-slate-800 dark:text-slate-200">{row.algo}</span>
              </td>
              <td className="text-right py-2 px-2 tabular-nums">{fmt(row.gap)}</td>
              <td className="text-right py-2 px-2 tabular-nums">{fmt(row.gapPct, 1)}%</td>
              <td className="text-right py-2 px-2 tabular-nums">{fmt(row.pathRatio)}</td>
              <td className="text-right py-2 px-2 tabular-nums">{fmt(row.pctOptimal, 1)}</td>
              <td className="text-right py-2 px-2 tabular-nums">{row.epsToNearOptimal}</td>
              <td className="text-right py-2 px-2 tabular-nums">{row.avgReturn}</td>
              <td className="text-right py-2 px-2 tabular-nums">{row.hazardHits}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
