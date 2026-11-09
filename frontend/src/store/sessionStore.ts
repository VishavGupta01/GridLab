import { create } from 'zustand';
import type { AgentConfig } from '../api/client';
import { api } from '../api/client';
import { useGridStore } from './gridStore';

export type TrainingState = 'idle' | 'running' | 'paused' | 'finished';

export interface OnlineMetrics {
  steps: number;
  return: number;
  success: number;
  hazard_hits: number;
  eps: number;
  td: number;
}

export interface EvalMetrics {
  gap: number;
  gap_pct: number;
  path_ratio: number | null;
  pct_optimal: number;
  episodes_to_near_optimal: number | null;
}

export interface AgentSnapshot {
  q: number[][];
  visits: number[];
  td_ema: number[];
  greedy_policy: number[];
}

export interface GridEvent {
  episode: number;
  grid_version: number;
  edits: { r: number; c: number; to: string }[];
}

export interface SessionState {
  sessionId: string | null;
  state: TrainingState;
  speed: number;
  episode: number;
  budget: number;
  gridVersion: number;

  agentConfigs: AgentConfig[];
  onlineHistory: Record<string, OnlineMetrics[]>;
  evalHistory: Record<string, { episode: number; metrics: EvalMetrics }[]>;
  agentSnapshots: Record<string, AgentSnapshot>;
  optimalSnapshot: { v: number[]; q: number[][]; policy: number[][]; greedy_path?: number[] } | null;
  events: GridEvent[];

  ws: WebSocket | null;

  setAgentConfigs: (configs: AgentConfig[]) => void;
  createSession: () => Promise<void>;
  start: () => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  reset: () => Promise<void>;
  setSpeed: (speed: number) => Promise<void>;
  connectWs: () => void;
  disconnectWs: () => void;
}

const DEFAULT_AGENTS: AgentConfig[] = [
  { id: 'q', algo: 'q_learning', alpha: 0.1, exploration: { type: 'decaying_epsilon', eps0: 1.0, eps_min: 0.05, decay: 0.995 }, q_init: 0, eps_bump_on_change: false, eps_bump_value: 0.3, enabled: true },
  { id: 's', algo: 'sarsa', alpha: 0.1, exploration: { type: 'decaying_epsilon', eps0: 1.0, eps_min: 0.05, decay: 0.995 }, q_init: 0, eps_bump_on_change: false, eps_bump_value: 0.3, enabled: true },
  { id: 'e', algo: 'expected_sarsa', alpha: 0.1, exploration: { type: 'decaying_epsilon', eps0: 1.0, eps_min: 0.05, decay: 0.995 }, q_init: 0, eps_bump_on_change: false, eps_bump_value: 0.3, enabled: true },
];

export const useSessionStore = create<SessionState>((set, get) => ({
  sessionId: null,
  state: 'idle',
  speed: 3,
  episode: 0,
  budget: 2000,
  gridVersion: 1,
  agentConfigs: DEFAULT_AGENTS,
  onlineHistory: { q: [], s: [], e: [] },
  evalHistory: { q: [], s: [], e: [] },
  agentSnapshots: {},
  optimalSnapshot: null,
  events: [],
  ws: null,

  setAgentConfigs: (configs) => set({ agentConfigs: configs }),

  createSession: async () => {
    const grid = useGridStore.getState().toJSON();
    const { agentConfigs, budget } = get();
    const { session_id } = await api.createSession({
      grid,
      agents: agentConfigs,
      master_seed: 42,
      budget,
      eval_every: 10,
    });
    set({
      sessionId: session_id,
      state: 'idle',
      episode: 0,
      onlineHistory: Object.fromEntries(agentConfigs.map((a) => [a.id, []])),
      evalHistory: Object.fromEntries(agentConfigs.map((a) => [a.id, []])),
      agentSnapshots: {},
      optimalSnapshot: null,
      events: [],
    });
    get().connectWs();
  },

  start: async () => {
    const { sessionId } = get();
    if (!sessionId) await get().createSession();
    const sid = get().sessionId!;
    await api.control(sid, 'start');
    set({ state: 'running' });
  },

  pause: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    await api.control(sessionId, 'pause');
    set({ state: 'paused' });
  },

  resume: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    await api.control(sessionId, 'resume');
    set({ state: 'running' });
  },

  reset: async () => {
    get().disconnectWs();
    set({
      sessionId: null,
      state: 'idle',
      episode: 0,
      onlineHistory: Object.fromEntries(get().agentConfigs.map((a) => [a.id, []])),
      evalHistory: Object.fromEntries(get().agentConfigs.map((a) => [a.id, []])),
      agentSnapshots: {},
      optimalSnapshot: null,
      events: [],
    });
  },

  setSpeed: async (speed) => {
    const { sessionId } = get();
    set({ speed });
    if (sessionId) {
      await api.control(sessionId, 'set_speed', speed);
    }
  },

  connectWs: () => {
    const { sessionId, ws: oldWs } = get();
    if (!sessionId) return;
    if (oldWs) oldWs.close();

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/sessions/${sessionId}`);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const state = get();

      if (msg.type === 'metrics') {
        const newHistory = { ...state.onlineHistory };
        for (const [aid, raw] of Object.entries(msg.agents) as [string, any][]) {
          if (!newHistory[aid]) newHistory[aid] = [];
          const m: OnlineMetrics = {
            steps: raw.steps ?? 0,
            return: raw.total_return ?? raw.return ?? 0,
            success: raw.success ? 1 : 0,
            hazard_hits: raw.hazard_hits ?? 0,
            eps: raw.eps ?? 0,
            td: raw.mean_abs_td_error ?? raw.td ?? 0,
          };
          newHistory[aid] = [...newHistory[aid], m];
        }
        set({ onlineHistory: newHistory, episode: msg.episode });
      } else if (msg.type === 'eval') {
        const newHistory = { ...state.evalHistory };
        for (const [aid, m] of Object.entries(msg.agents) as [string, any][]) {
          if (!newHistory[aid]) newHistory[aid] = [];
          newHistory[aid] = [...newHistory[aid], { episode: msg.episode, metrics: m }];
        }
        set({ evalHistory: newHistory });
      } else if (msg.type === 'snapshot') {
        set({
          agentSnapshots: msg.agents,
          optimalSnapshot: msg.optimal,
          episode: msg.episode,
        });
      } else if (msg.type === 'event') {
        set({
          events: [...state.events, msg],
          gridVersion: msg.grid_version,
        });
        if (msg.edits) {
          const gridState = useGridStore.getState();
          for (const edit of msg.edits as { r: number; c: number; to: string }[]) {
            gridState.setCell(edit.r, edit.c, edit.to as any);
          }
          gridState.clearStaged();
        }
      } else if (msg.type === 'status') {
        set({
          state: msg.state,
          speed: msg.speed,
          episode: msg.episode,
          budget: msg.budget,
        });
      }
    };

    ws.onclose = () => set({ ws: null });
    set({ ws });
  },

  disconnectWs: () => {
    const { ws } = get();
    if (ws) {
      ws.close();
      set({ ws: null });
    }
  },
}));
