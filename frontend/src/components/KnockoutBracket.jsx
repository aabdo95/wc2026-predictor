import { Trophy } from 'lucide-react'
import { flag } from '../utils/flags.js'
import { pct } from '../utils/colors.js'

// Each round lists every team's probability of *reaching* that round, sorted
// descending. The draw is randomised per simulation, so there are no fixed
// matchups — we show the top-N survivors per round (the bracket "funnel").
const ROUNDS = [
  { key: 'R32', label: 'Round of 32', slots: 32 },
  { key: 'R16', label: 'Round of 16', slots: 16 },
  { key: 'QF', label: 'Quarter-finals', slots: 8 },
  { key: 'SF', label: 'Semi-finals', slots: 4 },
  { key: 'Final', label: 'Final', slots: 2 },
  { key: 'Winner', label: 'Champion', slots: 1 },
]

function TeamChip({ team, prob, leader }) {
  return (
    <div
      className={`relative overflow-hidden rounded-md border px-2 py-1.5 transition-colors ${
        leader ? 'border-primary/60 bg-primary/10' : 'border-border bg-surface hover:border-border/80'
      }`}
    >
      <div className="flex items-center gap-1.5">
        <span className="shrink-0 text-sm leading-none">{flag(team)}</span>
        <span className="min-w-0 flex-1 truncate text-xs font-medium">{team}</span>
        <span className="stat-num shrink-0 text-[11px] text-text-secondary">
          {Math.round(prob * 100)}%
        </span>
      </div>
      {/* reach-probability underline */}
      <span
        className="absolute bottom-0 left-0 h-0.5 bg-primary/70"
        style={{ width: `${(prob * 100).toFixed(1)}%` }}
        aria-hidden
      />
    </div>
  )
}

function ChampionCard({ team, prob }) {
  return (
    <div className="relative w-full rounded-xl border border-primary/50 bg-gradient-to-b from-primary/20 to-surface px-4 py-5 text-center shadow-[0_0_36px_-10px_rgba(16,185,129,0.55)]">
      <Trophy className="mx-auto h-6 w-6 text-primary" />
      <div className="mt-3 text-4xl leading-none">{flag(team)}</div>
      <div className="mt-2 font-bold">{team}</div>
      <div className="stat-num mt-1 text-lg font-bold text-primary">{pct(prob)}</div>
      <div className="text-[10px] uppercase tracking-[0.15em] text-text-secondary">to lift the cup</div>
    </div>
  )
}

export default function KnockoutBracket({ rounds }) {
  return (
    <div className="overflow-x-auto pb-4">
      <div className="flex min-w-max items-stretch gap-0">
        {ROUNDS.map((round, ri) => {
          const teams = (rounds[round.key] ?? []).slice(0, round.slots)
          const isChampion = round.key === 'Winner'
          return (
            <div
              key={round.key}
              className={`flex shrink-0 flex-col px-4 ${isChampion ? 'w-48' : 'w-44'} ${
                ri > 0 ? 'border-l border-border/50' : ''
              }`}
            >
              <div className="mb-3 flex items-baseline justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wide text-text-primary">
                  {round.label}
                </h3>
                {!isChampion && (
                  <span className="stat-num text-[10px] text-text-secondary">{round.slots}</span>
                )}
              </div>

              {isChampion ? (
                teams[0] && <ChampionCard team={teams[0].team} prob={teams[0].prob} />
              ) : (
                <div className="flex flex-col gap-1.5">
                  {teams.map((t, i) => (
                    <TeamChip key={t.team} team={t.team} prob={t.prob} leader={i === 0} />
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
