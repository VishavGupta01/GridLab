import { useSessionStore } from '../store/sessionStore';
import { useUIStore } from '../store/uiStore';

export function RunBar() {
  const { state, episode, budget, speed, gridVersion, start, pause, resume, reset, setSpeed } = useSessionStore();
  const { darkMode, toggleDarkMode } = useUIStore();
  const setBudget = (v: number) => useSessionStore.setState({ budget: Math.max(1, Math.min(50000, v)) });

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 shrink-0">
      <span className="text-sm font-bold text-slate-700 dark:text-slate-200 mr-2">GridLab</span>
      <div className="flex items-center gap-1">
        {state === 'idle' || state === 'finished' ? (
          <button onClick={start} className="px-3 py-1 text-sm font-medium rounded bg-green-500 hover:bg-green-600 text-white">
            Train
          </button>
        ) : state === 'running' ? (
          <button onClick={pause} className="px-3 py-1 text-sm font-medium rounded bg-amber-500 hover:bg-amber-600 text-white">
            Pause
          </button>
        ) : (
          <button onClick={resume} className="px-3 py-1 text-sm font-medium rounded bg-green-500 hover:bg-green-600 text-white">
            Resume
          </button>
        )}
        <button onClick={reset} className="px-3 py-1 text-sm font-medium rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200">
          Reset
        </button>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500 dark:text-slate-400">Speed</span>
        <input
          type="range" min={1} max={10} value={speed}
          onChange={(e) => setSpeed(+e.target.value)}
          className="w-20 h-1 accent-blue-500"
        />
        <span className="text-xs text-slate-600 dark:text-slate-300 w-4 text-center">{speed}</span>
      </div>

      <div className="text-sm text-slate-600 dark:text-slate-300 font-mono">
        Episode {episode} / {budget}
      </div>

      {state === 'idle' && (
        <div className="flex items-center gap-1">
          <span className="text-xs text-slate-500 dark:text-slate-400">Budget</span>
          <input
            type="number" min={1} max={50000} step={100} value={budget}
            onChange={(e) => setBudget(+e.target.value)}
            className="w-20 text-xs px-1.5 py-0.5 border rounded dark:bg-slate-700 dark:border-slate-600 dark:text-slate-200"
          />
        </div>
      )}

      <div className="px-2 py-0.5 text-xs rounded bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300">
        Grid v{gridVersion}
      </div>

      <div className="ml-auto">
        <button
          onClick={toggleDarkMode}
          className="px-2 py-1 text-sm rounded hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300"
          title="Toggle dark mode"
        >
          {darkMode ? 'Light' : 'Dark'}
        </button>
      </div>
    </div>
  );
}
