import { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useSessionStore } from '../store/sessionStore';

const AGENT_COLORS: Record<string, string> = {
  q: '#3b82f6',
  s: '#ef4444',
  e: '#22c55e',
};

const AGENT_NAMES: Record<string, string> = {
  q: 'Q-Learning',
  s: 'SARSA',
  e: 'Expected SARSA',
};

function downsample<T>(arr: T[], maxPoints: number): T[] {
  if (arr.length <= maxPoints) return arr;
  const step = Math.ceil(arr.length / maxPoints);
  return arr.filter((_, i) => i % step === 0);
}

function ema(data: number[], window: number): number[] {
  if (window <= 1) return data;
  const alpha = 2 / (window + 1);
  const result: number[] = [];
  let prev = data[0] ?? 0;
  for (const v of data) {
    prev = alpha * v + (1 - alpha) * prev;
    result.push(prev);
  }
  return result;
}

interface ChartProps {
  title: string;
  dataKey: string;
  domain?: [number | 'auto', number | 'auto'];
}

function MetricChart({ title, dataKey, domain }: ChartProps) {
  const onlineHistory = useSessionStore((s) => s.onlineHistory);
  const evalHistory = useSessionStore((s) => s.evalHistory);
  const events = useSessionStore((s) => s.events);
  const [smoothing, setSmoothing] = useState(20);

  const isEval = ['gap', 'gap_pct', 'path_ratio', 'pct_optimal'].includes(dataKey);

  const chartData = useMemo(() => {
    const agentIds = Object.keys(isEval ? evalHistory : onlineHistory);
    if (agentIds.length === 0) return [];

    if (isEval) {
      const maxLen = Math.max(...agentIds.map((id) => evalHistory[id]?.length ?? 0));
      const data: any[] = [];
      for (let i = 0; i < maxLen; i++) {
        const point: any = {};
        let hasEp = false;
        for (const id of agentIds) {
          const entry = evalHistory[id]?.[i];
          if (entry) {
            if (!hasEp) { point.episode = entry.episode; hasEp = true; }
            point[id] = entry.metrics[dataKey as keyof typeof entry.metrics];
          }
        }
        if (hasEp) data.push(point);
      }
      return downsample(data, 500);
    }

    const maxLen = Math.max(...agentIds.map((id) => onlineHistory[id]?.length ?? 0));
    const rawData: Record<string, number[]> = {};
    for (const id of agentIds) {
      rawData[id] = (onlineHistory[id] ?? []).map((m: any) => m[dataKey] ?? 0);
    }

    const smoothed: Record<string, number[]> = {};
    for (const id of agentIds) {
      smoothed[id] = ema(rawData[id], smoothing);
    }

    const data: any[] = [];
    for (let i = 0; i < maxLen; i++) {
      const point: any = { episode: i };
      for (const id of agentIds) {
        point[id] = smoothed[id]?.[i] ?? null;
      }
      data.push(point);
    }
    return downsample(data, 500);
  }, [onlineHistory, evalHistory, dataKey, smoothing, isEval]);

  const agentIds = Object.keys(isEval ? evalHistory : onlineHistory);

  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">{title}</h3>
        {!isEval && (
          <label className="flex items-center gap-1 text-xs text-slate-500">
            Smooth
            <input
              type="range"
              min={1}
              max={100}
              value={smoothing}
              onChange={(e) => setSmoothing(Number(e.target.value))}
              className="w-16 h-3"
            />
            <span className="w-6">{smoothing}</span>
          </label>
        )}
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="episode" tick={{ fontSize: 10 }} />
          <YAxis domain={domain} tick={{ fontSize: 10 }} width={50} />
          <Tooltip
            contentStyle={{ fontSize: 11, backgroundColor: 'rgba(255,255,255,0.95)', border: '1px solid #e2e8f0' }}
          />
          <Legend iconSize={8} wrapperStyle={{ fontSize: 11 }} />
          {events.map((ev) => (
            <ReferenceLine
              key={ev.episode}
              x={ev.episode}
              stroke="#94a3b8"
              strokeDasharray="4 4"
              label=""
            />
          ))}
          {agentIds.map((id) => (
            <Line
              key={id}
              dataKey={id}
              name={AGENT_NAMES[id] || id}
              stroke={AGENT_COLORS[id] || '#666'}
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Charts() {
  return (
    <div className="p-3 overflow-y-auto" style={{ maxHeight: '500px' }}>
      <MetricChart title="Steps per Episode" dataKey="steps" />
      <MetricChart title="Return per Episode" dataKey="return" />
      <MetricChart title="Optimality Gap" dataKey="gap" />
      <MetricChart title="Success Rate" dataKey="success" domain={[0, 1]} />
      <MetricChart title="Epsilon" dataKey="eps" domain={[0, 1]} />
      <MetricChart title="% Optimal Actions" dataKey="pct_optimal" domain={[0, 1]} />
    </div>
  );
}
