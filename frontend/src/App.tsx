import { useEffect, useState } from 'react'

function App() {
  const [status, setStatus] = useState<string>('connecting...')

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((d) => setStatus(d.status))
      .catch(() => setStatus('error'))
  }, [])

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-slate-800 dark:text-slate-100 mb-4">
          GridLab
        </h1>
        <p className="text-lg text-slate-600 dark:text-slate-400">
          Interactive Arena for Comparing TD-Learning Agents
        </p>
        <div className="mt-6 inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-slate-800 shadow">
          <span
            className={`w-2 h-2 rounded-full ${
              status === 'ok' ? 'bg-green-500' : status === 'error' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
            }`}
          />
          <span className="text-sm text-slate-700 dark:text-slate-300">
            backend: {status}
          </span>
        </div>
      </div>
    </div>
  )
}

export default App
