import { motion, useReducedMotion } from 'framer-motion'
import { useApi } from '../hooks/useApi.js'
import GroupCard from '../components/GroupCard.jsx'
import { Loading, ErrorState } from '../components/States.jsx'

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.04, delayChildren: 0.05 } },
}

const item = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] } },
}

function LegendDot({ className, children }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`h-2.5 w-2.5 rounded-sm ${className}`} />
      <span className="text-text-secondary">{children}</span>
    </span>
  )
}

export default function Groups() {
  const { data, loading, error } = useApi('/api/groups')
  const reduce = useReducedMotion()

  if (loading) return <Loading label="Loading group standings…" />
  if (error) return <ErrorState error={error} />

  return (
    <div className="space-y-8">
      <header className="border-b border-border pb-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
              50,000 simulations
            </p>
            <h1 className="mt-1 text-3xl font-extrabold tracking-tight sm:text-4xl">Group Stage</h1>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
            <LegendDot className="bg-primary">Top 2 advance</LegendDot>
            <LegendDot className="bg-secondary">3rd — best-third race</LegendDot>
            <LegendDot className="bg-border">4th eliminated</LegendDot>
          </div>
        </div>
        <p className="mt-3 max-w-2xl text-sm text-text-secondary">
          Teams ranked by expected points. The shaded fill behind each row tracks its probability of
          reaching the knockouts — hover any team for exact advance and win-group odds.
        </p>
      </header>

      <motion.div
        variants={container}
        initial={reduce ? false : 'hidden'}
        animate="show"
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
      >
        {data.map((g) => (
          <motion.div key={g.group} variants={reduce ? undefined : item}>
            <GroupCard group={g.group} teams={g.teams} />
          </motion.div>
        ))}
      </motion.div>
    </div>
  )
}
