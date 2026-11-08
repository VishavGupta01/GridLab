import type { AgentConfig } from '../api/client';
import { useSessionStore } from '../store/sessionStore';

const ALGO_OPTIONS = [
  { value: 'q_learning', label: 'Q-Learning' },
  { value: 'sarsa', label: 'SARSA' },
  { value: 'expected_sarsa', label: 'Expected SARSA' },
] as const;

const EXPLORE_OPTIONS = [
  { value: 'epsilon', label: 'Fixed ε' },
  { value: 'decaying_epsilon', label: 'Decaying ε' },
  { value: 'softmax', label: 'Softmax' },
] as const;

interface AgentCardProps {
  config: AgentConfig;
  color: string;
  index: number;
}

export function AgentCard({ config, color, index }: AgentCardProps) {
  const state = useSessionStore((s) => s.state);
  const agentConfigs = useSessionStore((s) => s.agentConfigs);
  const setAgentConfigs = useSessionStore((s) => s.setAgentConfigs);
  const disabled = state === 'running';

  const update = (patch: Partial<AgentConfig>) => {
    const configs = [...agentConfigs];
    configs[index] = { ...configs[index], ...patch };
    setAgentConfigs(configs);
  };

  const updateExploration = (patch: Partial<AgentConfig['exploration']>) => {
    update({ exploration: { ...config.exploration, ...patch } });
  };

  return (
    <div className="p-3 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 space-y-2">
      <div className="flex items-center gap-2">
        <span className="w-3 h-3 rounded-full shrink-0" style={{ background: color }} />
        <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">{config.id.toUpperCase()}</span>
      </div>

      <label className="block">
        <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase">Algorithm</span>
        <select
          value={config.algo}
          onChange={(e) => update({ algo: e.target.value as AgentConfig['algo'] })}
          disabled={disabled}
          className="w-full text-xs px-1.5 py-1 border rounded dark:bg-slate-700 dark:border-slate-600 dark:text-slate-200 disabled:opacity-50"
        >
          {ALGO_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </label>

      <label className="block">
        <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase">Exploration</span>
        <select
          value={config.exploration.type}
          onChange={(e) => updateExploration({ type: e.target.value })}
          disabled={disabled}
          className="w-full text-xs px-1.5 py-1 border rounded dark:bg-slate-700 dark:border-slate-600 dark:text-slate-200 disabled:opacity-50"
        >
          {EXPLORE_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </label>

      {config.exploration.type === 'epsilon' && (
        <Row label={'ε'} value={config.exploration.eps ?? 0.1} disabled={disabled}
          onChange={(v) => updateExploration({ eps: v })} />
      )}
      {config.exploration.type === 'decaying_epsilon' && (
        <>
          <Row label={'ε0'} value={config.exploration.eps0 ?? 1.0} disabled={disabled}
            onChange={(v) => updateExploration({ eps0: v })} />
          <Row label="min" value={config.exploration.eps_min ?? 0.05} disabled={disabled}
            onChange={(v) => updateExploration({ eps_min: v })} />
          <Row label="decay" value={config.exploration.decay ?? 0.995} disabled={disabled}
            onChange={(v) => updateExploration({ decay: v })} />
        </>
      )}
      {config.exploration.type === 'softmax' && (
        <Row label={'τ'} value={config.exploration.tau ?? 1.0} disabled={disabled}
          onChange={(v) => updateExploration({ tau: v })} />
      )}

      <div className="grid grid-cols-3 gap-1">
        <Row label={'α'} value={config.alpha} disabled={disabled} onChange={(v) => update({ alpha: v })} />
        <Row label={'γ'} value={0.99} disabled={true} onChange={() => {}} />
        <Row label="q0" value={config.q_init} disabled={disabled} onChange={(v) => update({ q_init: v })} />
      </div>

    </div>
  );
}

function Row({ label, value, disabled, onChange }: { label: string; value: number; disabled: boolean; onChange: (v: number) => void }) {
  return (
    <label className="block">
      <span className="text-[10px] text-slate-500 dark:text-slate-400">{label}</span>
      <input
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        disabled={disabled}
        className="w-full text-xs px-1.5 py-0.5 border rounded dark:bg-slate-700 dark:border-slate-600 dark:text-slate-200 disabled:opacity-50"
      />
    </label>
  );
}
