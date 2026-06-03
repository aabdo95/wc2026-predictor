import { Loader2, AlertTriangle } from 'lucide-react'

export function Loading({ label = 'Loading…' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-24 text-text-secondary">
      <Loader2 className="h-7 w-7 animate-spin text-primary" />
      <span className="text-sm">{label}</span>
    </div>
  )
}

export function ErrorState({ error }) {
  return (
    <div className="mx-auto max-w-md card border-danger/40 px-6 py-8 text-center">
      <AlertTriangle className="mx-auto mb-3 h-8 w-8 text-danger" />
      <p className="font-semibold text-text-primary">Couldn’t load data</p>
      <p className="mt-1 text-sm text-text-secondary">{error}</p>
      <p className="mt-4 text-xs text-text-secondary">
        Is the API running on <span className="font-mono">localhost:8000</span>? Try{' '}
        <span className="font-mono">make backend</span>.
      </p>
    </div>
  )
}
