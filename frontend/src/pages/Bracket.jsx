import { motion, useReducedMotion } from 'framer-motion'
import { Info, MoveHorizontal } from 'lucide-react'
import { useApi } from '../hooks/useApi.js'
import KnockoutBracket from '../components/KnockoutBracket.jsx'
import { Loading, ErrorState } from '../components/States.jsx'

export default function Bracket() {
  const { data, loading, error } = useApi('/api/bracket')
  const reduce = useReducedMotion()

  if (loading) return <Loading label="Loading knockout bracket…" />
  if (error) return <ErrorState error={error} />

  return (
    <div className="space-y-6">
      <header className="border-b border-border pb-5">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">Knockout stage</p>
        <h1 className="mt-1 text-3xl font-extrabold tracking-tight sm:text-4xl">The Road to the Final</h1>
        <p className="mt-2 max-w-2xl text-sm text-text-secondary">
          Each round shows the teams most likely to reach it, with their probability of getting
          there. The leader of every round is highlighted, narrowing to the projected champion.
        </p>
      </header>

      {/* Honest framing: marginal probabilities, not fixed matchups */}
      <div className="flex items-start gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 text-xs text-text-secondary">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-secondary" />
        <p>
          The Round-of-32 draw is randomised in every simulation, so these are each team’s odds of
          reaching a round <span className="text-text-primary">averaged over all possible brackets</span> —
          the field narrowing, rather than fixed head-to-head matchups.
        </p>
      </div>

      <div className="flex items-center gap-1.5 text-[11px] text-text-secondary sm:hidden">
        <MoveHorizontal className="h-3.5 w-3.5" />
        Scroll sideways to follow the rounds
      </div>

      <motion.div
        initial={reduce ? false : { opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        className="card px-2 py-4 sm:px-4"
      >
        <KnockoutBracket rounds={data.rounds} />
      </motion.div>
    </div>
  )
}
