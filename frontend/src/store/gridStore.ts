import { create } from 'zustand';
import type { CellEdit, GridJSON, GridSettings, GridValidation } from '../api/client';
import { api } from '../api/client';

const CELL_TYPES = ['.', '#', 'S', 'G', 'H', 'M'] as const;
export type CellType = (typeof CELL_TYPES)[number];

export interface GridState {
  rows: string[][];
  name: string;
  version: number;
  goalRewards: Record<string, number>;
  settings: GridSettings;
  validation: GridValidation | null;
  stagedEdits: CellEdit[];
  undoStack: string[][][];
  redoStack: string[][][];

  nRows: number;
  nCols: number;

  setFromJSON: (g: GridJSON) => void;
  setCell: (r: number, c: number, type: CellType) => void;
  stageEdit: (r: number, c: number, to: string) => void;
  clearStaged: () => void;
  undo: () => void;
  redo: () => void;
  clear: () => void;
  resize: (rows: number, cols: number) => void;
  toJSON: () => GridJSON;
  validate: () => Promise<void>;
}

function parseRows(rowStrings: string[]): string[][] {
  return rowStrings.map((r) => r.split(''));
}

function rowsToStrings(rows: string[][]): string[] {
  return rows.map((r) => r.join(''));
}

const DEFAULT_SETTINGS: GridSettings = {
  step_cost: -1,
  hazard_penalty: -100,
  mud_cost: -5,
  goal_reward: 10,
  slip_prob: 0,
  max_steps: 200,
  gamma: 0.99,
};

function defaultGrid(): string[][] {
  const grid: string[][] = [];
  for (let r = 0; r < 10; r++) {
    grid.push(Array(10).fill('.'));
  }
  grid[0][0] = 'S';
  grid[9][9] = 'G';
  return grid;
}

export const useGridStore = create<GridState>((set, get) => ({
  rows: defaultGrid(),
  name: 'untitled',
  version: 1,
  goalRewards: { '9,9': 10 },
  settings: { ...DEFAULT_SETTINGS },
  validation: null,
  stagedEdits: [],
  undoStack: [],
  redoStack: [],
  nRows: 10,
  nCols: 10,

  setFromJSON: (g) => {
    const rows = parseRows(g.rows);
    set({
      rows,
      name: g.name,
      version: g.version,
      goalRewards: g.goal_rewards,
      settings: { ...DEFAULT_SETTINGS, ...g.settings },
      nRows: rows.length,
      nCols: rows[0]?.length ?? 0,
      stagedEdits: [],
      undoStack: [],
      redoStack: [],
    });
    get().validate();
  },

  setCell: (r, c, type) => {
    const { rows, undoStack, goalRewards } = get();
    const newUndo = [...undoStack, rows.map((row) => [...row])];
    const newRows = rows.map((row) => [...row]);

    if (type === 'S') {
      for (let ri = 0; ri < newRows.length; ri++)
        for (let ci = 0; ci < newRows[ri].length; ci++)
          if (newRows[ri][ci] === 'S') newRows[ri][ci] = '.';
    }

    newRows[r][c] = type;

    const newGoals = { ...goalRewards };
    if (type === 'G' && !newGoals[`${r},${c}`]) {
      newGoals[`${r},${c}`] = get().settings.goal_reward;
    }
    if (type !== 'G') {
      delete newGoals[`${r},${c}`];
    }

    set({
      rows: newRows,
      undoStack: newUndo.slice(-50),
      redoStack: [],
      goalRewards: newGoals,
    });
    get().validate();
  },

  stageEdit: (r, c, to) => {
    const { stagedEdits } = get();
    const filtered = stagedEdits.filter((e) => !(e.r === r && e.c === c));
    set({ stagedEdits: [...filtered, { r, c, to }] });
  },

  clearStaged: () => set({ stagedEdits: [] }),

  undo: () => {
    const { undoStack, rows } = get();
    if (undoStack.length === 0) return;
    const prev = undoStack[undoStack.length - 1];
    set({
      rows: prev,
      undoStack: undoStack.slice(0, -1),
      redoStack: [...get().redoStack, rows],
      nRows: prev.length,
      nCols: prev[0]?.length ?? 0,
    });
    get().validate();
  },

  redo: () => {
    const { redoStack, rows } = get();
    if (redoStack.length === 0) return;
    const next = redoStack[redoStack.length - 1];
    set({
      rows: next,
      redoStack: redoStack.slice(0, -1),
      undoStack: [...get().undoStack, rows],
      nRows: next.length,
      nCols: next[0]?.length ?? 0,
    });
    get().validate();
  },

  clear: () => {
    const { rows, undoStack, nRows, nCols } = get();
    const newRows = Array.from({ length: nRows }, () => Array(nCols).fill('.'));
    newRows[0][0] = 'S';
    newRows[nRows - 1][nCols - 1] = 'G';
    set({
      rows: newRows,
      undoStack: [...undoStack, rows],
      redoStack: [],
      goalRewards: { [`${nRows - 1},${nCols - 1}`]: get().settings.goal_reward },
    });
    get().validate();
  },

  resize: (newR, newC) => {
    newR = Math.max(5, Math.min(20, newR));
    newC = Math.max(5, Math.min(20, newC));
    const { rows, undoStack } = get();
    const resized: string[][] = [];
    for (let r = 0; r < newR; r++) {
      const row: string[] = [];
      for (let c = 0; c < newC; c++) {
        row.push(r < rows.length && c < rows[r].length ? rows[r][c] : '.');
      }
      resized.push(row);
    }
    set({
      rows: resized,
      nRows: newR,
      nCols: newC,
      undoStack: [...undoStack, rows],
      redoStack: [],
    });
    get().validate();
  },

  toJSON: (): GridJSON => {
    const { rows, name, version, goalRewards, settings } = get();
    return {
      version,
      name,
      rows: rowsToStrings(rows),
      goal_rewards: goalRewards,
      settings,
    };
  },

  validate: async () => {
    try {
      const result = await api.validate(get().toJSON());
      set({ validation: result });
    } catch {
      set({ validation: null });
    }
  },
}));
