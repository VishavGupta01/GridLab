const API_BASE = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  return resp.json();
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  presets: () => request<{ presets: string[] }>('/presets'),
  preset: (name: string) => request<GridJSON>(`/presets/${name}`),

  validate: (grid: GridJSON) =>
    request<GridValidation>('/grids/validate', {
      method: 'POST',
      body: JSON.stringify(grid),
    }),

  createSession: (data: SessionCreate) =>
    request<{ session_id: string }>('/sessions', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  control: (sid: string, action: string, speed?: number) =>
    request<{ ok: boolean }>(`/sessions/${sid}/control`, {
      method: 'POST',
      body: JSON.stringify({ action, speed }),
    }),

  edit: (sid: string, edits: CellEdit[], at_episode?: number) =>
    request<{ ok: boolean }>(`/sessions/${sid}/edit`, {
      method: 'POST',
      body: JSON.stringify({ edits, at_episode }),
    }),

  snapshot: (sid: string) =>
    request<any>(`/sessions/${sid}/snapshot`),

  rollout: (sid: string, agent_id: string, greedy = true, seed = 0) =>
    request<RolloutResult>(`/sessions/${sid}/rollout`, {
      method: 'POST',
      body: JSON.stringify({ agent_id, greedy, seed }),
    }),

  explain: (sid: string, data: ExplainRequest) =>
    request<any>(`/sessions/${sid}/explain`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  exportData: (sid: string) =>
    request<any>(`/sessions/${sid}/export`),
};

// Types
export interface GridJSON {
  version: number;
  name: string;
  rows: string[];
  goal_rewards: Record<string, number>;
  settings: GridSettings;
}

export interface GridSettings {
  step_cost: number;
  hazard_penalty: number;
  mud_cost: number;
  goal_reward: number;
  slip_prob: number;
  max_steps: number;
  gamma: number;
}

export interface AgentConfig {
  id: string;
  algo: 'q_learning' | 'sarsa' | 'expected_sarsa';
  alpha: number;
  exploration: { type: string; eps?: number; eps0?: number; eps_min?: number; decay?: number; tau?: number };
  q_init: number;
  eps_bump_on_change: boolean;
  eps_bump_value: number;
}

export interface SessionCreate {
  grid: GridJSON;
  agents: AgentConfig[];
  master_seed: number;
  budget: number;
  eval_every: number;
}

export interface CellEdit {
  r: number;
  c: number;
  to: string;
}

export interface GridValidation {
  valid: boolean;
  solvable: boolean;
  issues: string[];
  optimal_path_length: number | null;
  v_star_s0: number | null;
}

export interface RolloutStep {
  s: number;
  a: number;
  r: number;
  s2: number;
  done: boolean;
}

export interface RolloutResult {
  agent_id: string;
  steps: RolloutStep[];
  total_return: number;
  success: boolean;
  optimal_steps: number | null;
}

export interface ExplainRequest {
  agent_id: string;
  s: number;
  a: number;
  r: number;
  s2: number;
  terminated: boolean;
  a2?: number;
}
