import { useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { AlertTriangle, Info, MoveHorizontal } from 'lucide-react'
import { useApi } from '../hooks/useApi.js'
import KnockoutBracket from '../components/KnockoutBracket.jsx'
import MostLikelyBracket from '../components/MostLikelyBracket.jsx'
import { Loading, ErrorState } from '../components/States.jsx'

const TABS = [
  { key: 'funnel',  label: 'Probability Funnel' },
  { key: 'bracket', label: 'Most Likely Bracket' },
]

export default function Bracket() {
  const { data, loading, error } = useApi('/api/bracket')
  const reduce = useReducedMotion()
  const [tab, setTab] = useState('funnel')

  if (loading) return <Loading label="Loading knockout bracket…" />
  if (error)   return <ErrorState error={error} />

  const fadeProps = reduce
    ? {}
    : {
        initial:    { opacity: 0, y: 6 },
        animate:    { opacity: 1, y: 0 },
        exit:       { opacity: 0, y: -4 },
        transition: { duration: 0.2, ease: 'easeOut' },
      }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <header className="border-b border-border pb-5">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
          Knockout stage
        </p>
        <h1 className="mt-1 text-3xl font-extrabold tracking-tight sm:text-4xl">
          The Road to the Final
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-text-secondary">
          Two views of the tournament: the full probability distribution across all simulations,
          and a traditional bracket built from the most likely survivor at each slot.
        </p>
      </header>

      {/* Tab switcher */}
      <div
        role="tablist"
        aria-label="Bracket view"
        className="flex w-fit gap-0.5 rounded-xl border border-border bg-surface p-1"
      >
        {TABS.map((t) => (
          <button
            key={t.key}
            role="tab"
            type="button"
            id={`tab-${t.key}`}
            aria-selected={tab === t.key}
            aria-controls={`panel-${t.key}`}
            onClick={() => setTab(t.key)}
            className={`min-h-[40px] cursor-pointer rounded-lg px-4 py-2 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-primary ${
              tab === t.key
                ? 'bg-surface-elevated text-primary shadow-sm'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab panels */}
      <AnimatePresence mode="wait" initial={false}>
        {tab === 'funnel' ? (
          <motion.div key="funnel" {...fadeProps} role="tabpanel" id="panel-funnel" aria-labelledby="tab-funnel">
            {/* Info note for funnel */}
            <div className="mb-4 flex items-start gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 text-xs text-text-secondary">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-secondary" />
              <p>
                The Round-of-32 draw is randomised in every simulation, so these are each team's odds of
                reaching a round{' '}
                <span className="text-text-primary">averaged over all possible brackets</span> — the field
                narrowing, rather than fixed matchups.
              </p>
            </div>

            <div className="flex items-center gap-1.5 mb-3 text-[11px] text-text-secondary sm:hidden">
              <MoveHorizontal className="h-3.5 w-3.5" />
              Scroll sideways to follow the rounds
            </div>

            <div className="card px-2 py-4 sm:px-4">
              <KnockoutBracket rounds={data.rounds} />
            </div>
          </motion.div>
        ) : (
          <motion.div key="bracket" {...fadeProps} role="tabpanel" id="panel-bracket" aria-labelledby="tab-bracket">
            {/* Disclaimer */}
            <div className="mb-4 flex items-start gap-2 rounded-lg border border-secondary/30 bg-secondary/10 px-3 py-2.5 text-xs text-secondary">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <p>
                This shows the most likely team at each bracket position — actual matchups vary
                across 50,000 simulations. Seeds 1 and 2 (Spain &amp; Argentina) are placed in
                opposite halves so the most probable final is shown.
              </p>
            </div>

            <div className="flex items-center gap-1.5 mb-3 text-[11px] text-text-secondary sm:hidden">
              <MoveHorizontal className="h-3.5 w-3.5" />
              Scroll sideways to view the full bracket
            </div>

            <div className="card px-2 py-4 sm:px-4">
              <MostLikelyBracket rounds={data.rounds} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
