import { useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import { flag } from '../utils/flags.js'
import ConfidenceMeter from './ConfidenceMeter.jsx'
import MatchDetail from './MatchDetail.jsx'

const CONFIDENCE = {
  high: { label: 'High', cls: 'bg-primary/15 text-primary ring-primary/30' },
  medium: { label: 'Medium', cls: 'bg-secondary/15 text-secondary ring-secondary/30' },
  low: { label: 'Low', cls: 'bg-danger/15 text-danger ring-danger/30' },
}

function ConfidenceBadge({ level }) {
  const c = CONFIDENCE[level] ?? CONFIDENCE.low
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-inset ${c.cls}`}
    >
      {c.label}
    </span>
  )
}

export default function MatchCard({ match }) {
  const [open, setOpen] = useState(false)
  const reduce = useReducedMotion()

  return (
    <div className="card overflow-hidden transition-colors hover:border-border/80">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full cursor-pointer px-4 py-3.5 text-left transition-colors hover:bg-surface-elevated/40"
      >
        {/* Teams + probability bar */}
        <div className="flex items-center gap-3 sm:gap-4">
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <span className="text-xl leading-none">{flag(match.home)}</span>
            <span className="min-w-0 truncate text-sm font-semibold">{match.home}</span>
          </div>

          <div className="w-24 shrink-0 sm:w-44 md:w-56">
            <ConfidenceMeter
              home={match.p_home_win}
              draw={match.p_draw}
              away={match.p_away_win}
            />
          </div>

          <div className="flex min-w-0 flex-1 items-center justify-end gap-2">
            <span className="min-w-0 truncate text-right text-sm font-semibold">{match.away}</span>
            <span className="text-xl leading-none">{flag(match.away)}</span>
          </div>
        </div>

        {/* Predicted scoreline + confidence + group + chevron */}
        <div className="mt-3 flex items-center justify-center gap-2.5 text-sm text-text-secondary">
          <span className="rounded bg-surface-elevated px-1.5 py-0.5 text-[11px] font-medium">
            Grp {match.group}
          </span>
          <span>
            Predicted{' '}
            <span className="stat-num font-semibold text-text-primary">
              {match.expected_scoreline}
            </span>
          </span>
          <ConfidenceBadge level={match.confidence} />
          <ChevronDown
            className={`h-4 w-4 transition-transform duration-200 ${open ? 'rotate-180 text-primary' : ''}`}
          />
        </div>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="detail"
            initial={reduce ? false : { height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={reduce ? { opacity: 0 } : { height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <MatchDetail matchId={match.match_id} home={match.home} away={match.away} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
