import { create } from 'zustand';
import type { CellType } from './gridStore';

export type OverlayType = 'none' | 'q_heatmap' | 'policy' | 'visits' | 'td_error' | 'optimal' | 'disagreement';
export type TabType = 'charts' | 'race' | 'adaptation';

export interface UIState {
  activeTool: CellType;
  selectedOverlay: OverlayType;
  selectedAgent: string;
  activeTab: TabType;
  darkMode: boolean;
  isPainting: boolean;

  setTool: (tool: CellType) => void;
  setOverlay: (overlay: OverlayType) => void;
  setSelectedAgent: (id: string) => void;
  setTab: (tab: TabType) => void;
  toggleDarkMode: () => void;
  setIsPainting: (v: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  activeTool: '.',
  selectedOverlay: 'none',
  selectedAgent: 'q',
  activeTab: 'charts',
  darkMode: false,
  isPainting: false,

  setTool: (tool) => set({ activeTool: tool }),
  setOverlay: (overlay) => set({ selectedOverlay: overlay }),
  setSelectedAgent: (id) => set({ selectedAgent: id }),
  setTab: (tab) => set({ activeTab: tab }),
  toggleDarkMode: () =>
    set((s) => {
      const next = !s.darkMode;
      document.documentElement.classList.toggle('dark', next);
      return { darkMode: next };
    }),
  setIsPainting: (v) => set({ isPainting: v }),
}));
