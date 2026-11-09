import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { CellType } from '../store/gridStore';
import { useGridStore } from '../store/gridStore';
import { useSessionStore } from '../store/sessionStore';
import { useUIStore } from '../store/uiStore';

const BRUSHES: { type: CellType; label: string; key: string; color: string }[] = [
  { type: '.', label: 'Erase', key: '1', color: 'bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-500' },
  { type: '#', label: 'Wall', key: '2', color: 'bg-slate-800 dark:bg-slate-400' },
  { type: 'S', label: 'Start', key: '3', color: 'bg-blue-500' },
  { type: 'G', label: 'Goal', key: '4', color: 'bg-green-500' },
];

export function Toolbox() {
  const activeTool = useUIStore((s) => s.activeTool);
  const setTool = useUIStore((s) => s.setTool);
  const { nRows, nCols, resize, undo, redo, clear, setFromJSON, toJSON, stagedEdits, clearStaged } = useGridStore();
  const validation = useGridStore((s) => s.validation);
  const sessionState = useSessionStore((s) => s.state);
  const sessionId = useSessionStore((s) => s.sessionId);
  const [presets, setPresets] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.presets().then((r) => setPresets(r.presets)).catch(() => {});
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      const idx = parseInt(e.key);
      if (idx >= 1 && idx <= 4) {
        setTool(BRUSHES[idx - 1].type);
        e.preventDefault();
      }
      if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
        undo();
        e.preventDefault();
      }
      if (e.key === 'z' && (e.ctrlKey || e.metaKey) && e.shiftKey) {
        redo();
        e.preventDefault();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [setTool, undo, redo]);

  const loadPreset = async (name: string) => {
    if (!name) return;
    try {
      const data = await api.preset(name);
      setFromJSON(data);
    } catch {}
  };

  const exportJSON = () => {
    const data = toJSON();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${data.name || 'grid'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const importJSON = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string);
        setFromJSON(data);
      } catch {}
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const applyEdits = async () => {
    if (!sessionId || stagedEdits.length === 0) return;
    await api.edit(sessionId, stagedEdits);
    for (const edit of stagedEdits) {
      useGridStore.getState().setCell(edit.r, edit.c, edit.to as CellType);
    }
    clearStaged();
  };

  return (
    <div className="w-48 flex flex-col gap-3 p-3 bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 overflow-y-auto">
      <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Brushes</div>
      <div className="flex flex-col gap-1">
        {BRUSHES.map((b) => (
          <button
            key={b.type}
            onClick={() => setTool(b.type)}
            className={`flex items-center gap-2 px-2 py-1.5 rounded text-sm transition-colors ${
              activeTool === b.type
                ? 'bg-blue-100 dark:bg-blue-900 ring-2 ring-blue-500'
                : 'hover:bg-slate-100 dark:hover:bg-slate-700'
            }`}
          >
            <span className={`w-4 h-4 rounded-sm ${b.color} shrink-0`} />
            <span className="flex-1 text-left text-slate-700 dark:text-slate-200">{b.label}</span>
            <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">{b.key}</span>
          </button>
        ))}
      </div>

      <hr className="border-slate-200 dark:border-slate-700" />

      <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Size</div>
      <div className="flex gap-2 items-center">
        <label className="text-xs text-slate-500 dark:text-slate-400">R</label>
        <input
          type="number" min={5} max={20} value={nRows}
          onChange={(e) => resize(+e.target.value, nCols)}
          className="w-12 px-1 py-0.5 text-xs border rounded dark:bg-slate-700 dark:border-slate-600 dark:text-slate-200"
        />
        <label className="text-xs text-slate-500 dark:text-slate-400">C</label>
        <input
          type="number" min={5} max={20} value={nCols}
          onChange={(e) => resize(nRows, +e.target.value)}
          className="w-12 px-1 py-0.5 text-xs border rounded dark:bg-slate-700 dark:border-slate-600 dark:text-slate-200"
        />
      </div>

      <div className="flex gap-1">
        <button onClick={undo} className="flex-1 text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300" title="Ctrl+Z">
          Undo
        </button>
        <button onClick={redo} className="flex-1 text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300" title="Ctrl+Shift+Z">
          Redo
        </button>
      </div>
      <button onClick={clear} className="text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300">
        Clear
      </button>

      <hr className="border-slate-200 dark:border-slate-700" />

      <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Presets</div>
      <select
        onChange={(e) => loadPreset(e.target.value)}
        defaultValue=""
        className="text-xs px-2 py-1 border rounded dark:bg-slate-700 dark:border-slate-600 dark:text-slate-200"
      >
        <option value="">Select preset...</option>
        {presets.map((p) => (
          <option key={p} value={p}>{p}</option>
        ))}
      </select>

      <div className="flex gap-1">
        <button onClick={exportJSON} className="flex-1 text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300">
          Export
        </button>
        <button onClick={() => fileInputRef.current?.click()} className="flex-1 text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300">
          Import
        </button>
        <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={importJSON} />
      </div>

      {validation && (
        <div className={`text-xs p-2 rounded ${
          validation.valid && validation.solvable
            ? 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300'
            : 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300'
        }`}>
          {validation.valid && validation.solvable ? (
            <>
              Solvable
              {validation.optimal_path_length != null && <> &middot; Path: {validation.optimal_path_length}</>}
              {validation.v_star_s0 != null && <> &middot; V*: {validation.v_star_s0.toFixed(1)}</>}
            </>
          ) : (
            validation.issues.join(', ') || 'Invalid grid'
          )}
        </div>
      )}

      {sessionState !== 'idle' && stagedEdits.length > 0 && (
        <button
          onClick={applyEdits}
          className="text-xs px-2 py-2 rounded bg-amber-500 hover:bg-amber-600 text-white font-semibold"
        >
          Apply Edits ({stagedEdits.length})
        </button>
      )}

    </div>
  );
}
