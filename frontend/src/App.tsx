import { useEffect } from 'react';
import AdaptationPanel from './components/AdaptationPanel';
import { AgentCard } from './components/AgentCard';
import Charts from './components/Charts';
import { GridCanvas } from './components/GridCanvas';
import RaceView from './components/RaceView';
import { RunBar } from './components/RunBar';
import { Toolbox } from './components/Toolbox';
import { useGridStore } from './store/gridStore';
import { useSessionStore } from './store/sessionStore';
import { useUIStore, type TabType } from './store/uiStore';

const AGENT_COLORS = ['#3b82f6', '#ef4444', '#22c55e'];

const TABS: { id: TabType; label: string }[] = [
  { id: 'charts', label: 'Learning Curves' },
  { id: 'race', label: 'Race' },
  { id: 'adaptation', label: 'Adaptation' },
];

function App() {
  const activeTab = useUIStore((s) => s.activeTab);
  const setTab = useUIStore((s) => s.setTab);
  const agentConfigs = useSessionStore((s) => s.agentConfigs);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) return;

      const toolKeys: Record<string, string> = { '1': '.', '2': '#', '3': 'S', '4': 'G', '5': 'H', '6': 'M' };
      if (toolKeys[e.key]) {
        e.preventDefault();
        useUIStore.getState().setTool(toolKeys[e.key] as any);
      }
      if (e.key === ' ') {
        e.preventDefault();
        const state = useSessionStore.getState();
        if (state.state === 'running') state.pause();
        else if (state.state === 'paused') state.resume();
        else if (state.state === 'idle') state.start();
      }
      if (e.key === 'z' && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
        e.preventDefault();
        useGridStore.getState().undo();
      }
      if (e.key === 'z' && (e.metaKey || e.ctrlKey) && e.shiftKey) {
        e.preventDefault();
        useGridStore.getState().redo();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-200 flex flex-col">
      <RunBar />

      <div className="flex flex-1 overflow-hidden">
        <div className="w-48 flex-shrink-0 border-r border-slate-200 dark:border-slate-700 overflow-y-auto">
          <Toolbox />
        </div>

        <div className="flex-1 p-4 flex items-center justify-center overflow-hidden">
          <GridCanvas />
        </div>

        <div className="w-64 flex-shrink-0 border-l border-slate-200 dark:border-slate-700 overflow-y-auto p-2 space-y-2">
          {agentConfigs.map((cfg, i) => (
            <AgentCard key={cfg.id} config={cfg} color={AGENT_COLORS[i]} index={i} />
          ))}
        </div>
      </div>

      <div className="border-t border-slate-200 dark:border-slate-700" style={{ minHeight: '250px', maxHeight: '45vh' }}>
        <div className="flex border-b border-slate-200 dark:border-slate-700">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setTab(tab.id)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                  : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="overflow-auto" style={{ maxHeight: 'calc(45vh - 40px)' }}>
          {activeTab === 'charts' && <Charts />}
          {activeTab === 'race' && <RaceView />}
          {activeTab === 'adaptation' && <AdaptationPanel />}
        </div>
      </div>
    </div>
  );
}

export default App;
