import { TrendingUp } from 'lucide-react'
import { flag } from '../utils/flags.js'
import { pct } from '../utils/colors.js'

// Finishing-position tier drives the left rail + the row's qualification meter.
//  1st/2nd → green (advancing), 3rd → amber (possible best third), 4th → neutral.
const TIERS = [
  { rail: 'bg-primary', fill: 'rgba(16,185,129,0.16)' },
  { rail: 'bg-primary', fill: 'rgba(16,185,129,0.16)' },
  { rail: 'bg-secondary', fill: 'rgba(245,158,11,0.14)' },
  { rail: 'bg-transparent', fill: 'rgba(148,163,184,0.08)' },
]

function gd(value) {
  const v = Number(value)
  const s = v.toFixed(2)
  return v > 0 ? `+${s}` : s
}

function TeamRow({ team, idx }) {
  const tier = TIERS[idx] ?? TIERS[3]
  return (
    <div className="group/row relative flex items-stretch">
      {/* tier rail */}
      <span className={`w-[3px] shrink-0 ${tier.rail}`} aria-hidden />

      <div className="relative flex-1 overflow-hidden">
        {/* qualification meter: width ∝ P(advance), tinted by tier */}
        <div
          className="absolute inset-y-0 left-0 transition-[width,background-color] duration-300 group-hover/row:brightness-150"
          style={{ width: `${(team.advance_prob * 100).toFixed(1)}%`, backgroundColor: tier.fill }}
          aria-hidden
        />

        <div className="relative flex items-center gap-2.5 px-3 py-2">
          {/* Left region (rank · flag · name). On hover the name fades out and
              the advance / win-group stats crossfade in over the SAME space,
              backed opaque so nothing shows through. xPts/xGD are siblings
              outside this box, so they stay put and visible. */}
          <div className="relative flex min-w-0 flex-1 items-center gap-2.5 overflow-hidden">
            <span className="stat-num w-3 shrink-0 text-center text-[11px] text-text-secondary">
              {idx + 1}
            </span>
            <span className="shrink-0 text-lg leading-none">{flag(team.team)}</span>
            <span className="min-w-0 flex-1 truncate text-sm font-medium transition-opacity duration-150 group-hover/row:opacity-0">
              {team.team}
            </span>

            <div className="pointer-events-none absolute inset-0 flex items-center gap-2 bg-surface-elevated pl-1 pr-2 opacity-0 transition-opacity duration-150 group-hover/row:opacity-100">
              <TrendingUp className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span className="whitespace-nowrap text-[11px]">
                <span className="text-text-secondary">Adv </span>
                <span className="stat-num font-semibold text-primary">{pct(team.advance_prob)}</span>
              </span>
              <span className="whitespace-nowrap text-[11px]">
                <span className="text-text-secondary">Win </span>
                <span className="stat-num font-semibold text-text-primary">
                  {pct(team.win_group_prob)}
                </span>
              </span>
            </div>
          </div>

          <span className="stat-num w-11 text-right text-sm font-semibold tracking-tight">
            {team.expected_points.toFixed(2)}
          </span>
          <span
            className={`stat-num w-12 text-right text-xs ${
              team.expected_gd > 0
                ? 'text-primary'
                : team.expected_gd < 0
                  ? 'text-danger'
                  : 'text-text-secondary'
            }`}
          >
            {gd(team.expected_gd)}
          </span>
        </div>
      </div>
    </div>
  )
}

export default function GroupCard({ group, teams }) {
  const sorted = [...teams].sort((a, b) => b.expected_points - a.expected_points)
  return (
    <div className="card group/card overflow-hidden transition-all duration-300 hover:border-primary/40 hover:shadow-[0_0_0_1px_rgba(16,185,129,0.15),0_12px_30px_-12px_rgba(0,0,0,0.6)]">
      <div className="flex items-center justify-between border-b border-border bg-surface-elevated/40 px-3 py-2.5">
        <div className="flex items-baseline gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-text-secondary">
            Group
          </span>
          <span className="stat-num text-xl font-bold leading-none text-primary">{group}</span>
        </div>
        <div className="flex items-center gap-3 text-[9px] font-medium uppercase tracking-[0.15em] text-text-secondary">
          <span className="w-11 text-right">xPts</span>
          <span className="w-12 text-right">xGD</span>
        </div>
      </div>
      <div className="divide-y divide-border/50">
        {sorted.map((team, idx) => (
          <TeamRow key={team.team} team={team} idx={idx} />
        ))}
      </div>
    </div>
  )
}
