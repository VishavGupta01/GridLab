import { useState } from 'react';
import { api } from '../api/client';
import { useSessionStore } from '../store/sessionStore';

const AGENT_NAMES: Record<string, string> = { q: 'Q-Learning', s: 'SARSA', e: 'Expected SARSA' };
const ACTION_NAMES = ['Up', 'Right', 'Down', 'Left'];

export default function Inspector() {
  const sessionId = useSessionStore((s) => s.sessionId);
  const agentConfigs = useSessionStore((s) => s.agentConfigs);
  const [agentId, setAgentId] = useState('q');
  const [s, setS] = useState(0);
  const [a, setA] = useState(1);
  const [r, setR] = useState(-1);
  const [s2, setS2] = useState(1);
  const [terminated, setTerminated] = useState(false);
  const [a2, setA2] = useState<number | undefined>(undefined);
  const [result, setResult] = useState<any>(null);
  const [mode, setMode] = useState<'single' | 'compare'>('compare');

  const fetchExplain = async () => {
    if (!sessionId) return;
    try {
      const data = await api.explain(sessionId, { agent_id: agentId, s, a, r, s2, terminated, a2 });
      setResult(data);
    } catch (e: any) {
      setResult({ error: e.message });
    }
  };

  return (
    <div className="p-3">
      <div className="flex gap-2 mb-3">
        <button
          onClick={() => setMode('single')}
          className={`px-3 py-1 text-sm rounded ${mode === 'single' ? 'bg-blue-500 text-white' : 'bg-slate-200 dark:bg-slate-700'}`}
        >
          Follow Agent
        </button>
        <button
          onClick={() => setMode('compare')}
          className={`px-3 py-1 text-sm rounded ${mode === 'compare' ? 'bg-blue-500 text-white' : 'bg-slate-200 dark:bg-slate-700'}`}
        >
          Compare Rules
        </button>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-3 text-sm">
        <label className="flex flex-col">
          <span className="text-xs text-slate-500">Agent</span>
          <select value={agentId} onChange={(e) => setAgentId(e.target.value)}
            className="border rounded px-2 py-1 dark:bg-slate-800 dark:border-slate-600">
            {agentConfigs.map((c) => <option key={c.id} value={c.id}>{AGENT_NAMES[c.id]}</option>)}
          </select>
        </label>
        <label className="flex flex-col">
          <span className="text-xs text-slate-500">State (s)</span>
          <input type="number" value={s} onChange={(e) => setS(Number(e.target.value))}
            className="border rounded px-2 py-1 w-full dark:bg-slate-800 dark:border-slate-600" />
        </label>
        <label className="flex flex-col">
          <span className="text-xs text-slate-500">Action</span>
          <select value={a} onChange={(e) => setA(Number(e.target.value))}
            className="border rounded px-2 py-1 dark:bg-slate-800 dark:border-slate-600">
            {ACTION_NAMES.map((n, i) => <option key={i} value={i}>{n}</option>)}
          </select>
        </label>
        <label className="flex flex-col">
          <span className="text-xs text-slate-500">Reward</span>
          <input type="number" step={0.1} value={r} onChange={(e) => setR(Number(e.target.value))}
            className="border rounded px-2 py-1 w-full dark:bg-slate-800 dark:border-slate-600" />
        </label>
        <label className="flex flex-col">
          <span className="text-xs text-slate-500">Next State (s')</span>
          <input type="number" value={s2} onChange={(e) => setS2(Number(e.target.value))}
            className="border rounded px-2 py-1 w-full dark:bg-slate-800 dark:border-slate-600" />
        </label>
        <label className="flex flex-col">
          <span className="text-xs text-slate-500">Next Action (a')</span>
          <input type="number" value={a2 ?? ''} onChange={(e) => setA2(e.target.value ? Number(e.target.value) : undefined)}
            className="border rounded px-2 py-1 w-full dark:bg-slate-800 dark:border-slate-600" />
        </label>
        <label className="flex items-end gap-1">
          <input type="checkbox" checked={terminated} onChange={(e) => setTerminated(e.target.checked)} />
          <span className="text-xs">Terminal</span>
        </label>
        <button onClick={fetchExplain}
          className="self-end px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600">
          Explain
        </button>
      </div>

      {result && !result.error && (
        <div className="space-y-3">
          {mode === 'compare' && result.all_agents ? (
            <div className="grid grid-cols-3 gap-3">
              {Object.entries(result.all_agents).map(([aid, data]: [string, any]) => (
                <div key={aid} className="border rounded p-2 dark:border-slate-700">
                  <h4 className="text-sm font-medium mb-1">{AGENT_NAMES[aid]}</h4>
                  <div className="text-xs space-y-1">
                    <p>Target: <span className="font-mono tabular-nums">{data.target?.toFixed(4)}</span></p>
                    <p>TD Error: <span className="font-mono tabular-nums">{data.td_error?.toFixed(4)}</span></p>
                    <p>Q(s,a) before: <span className="font-mono tabular-nums">{data.q_s_a_before?.toFixed(4)}</span></p>
                    <p>Q(s,a) after: <span className="font-mono tabular-nums">{data.q_s_a_after?.toFixed(4)}</span></p>
                    <p className="text-slate-500 italic mt-1">{data.explanation}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="border rounded p-3 dark:border-slate-700">
              <h4 className="text-sm font-medium mb-2">{AGENT_NAMES[result.algo] || result.algo}</h4>
              <div className="text-xs space-y-1">
                <p>Q(s,·) before: [{result.q_before?.map((v: number) => v.toFixed(3)).join(', ')}]</p>
                <p>Target: <span className="font-mono">{result.target?.toFixed(4)}</span></p>
                <p>TD Error: <span className="font-mono">{result.td_error?.toFixed(4)}</span></p>
                <p>Q(s,a) after: <span className="font-mono">{result.q_s_a_after?.toFixed(4)}</span></p>
                <p className="text-slate-500 italic mt-2">{result.explanation}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {result?.error && (
        <div className="text-red-500 text-sm">{result.error}</div>
      )}

      {!result && (
        <div className="text-center text-slate-400 dark:text-slate-500 py-8">
          Enter a transition and click Explain to see the TD update breakdown.
        </div>
      )}
    </div>
  );
}
