import { Loader2 } from 'lucide-react'
import { useApi } from '../hooks/useApi.js'

// snake_case feature → human-readable label. Falls back to title-casing.
const FEATURE_LABELS = {
  elo_diff: 'ELO gap',
  elo_diff_adj: 'ELO gap (venue-adj.)',
  elo_home: 'Home ELO',
  elo_away: 'Away ELO',
  elo_home_adj: 'Home ELO (venue-adj.)',
  elo_expected_home: 'ELO win expectancy',
  squad_value_ratio: 'Squad value ratio',
  home_squad_value: 'Home squad value',
  away_squad_value: 'Away squad value',
  home_avg_player_value: 'Home avg player value',
  away_avg_player_value: 'Away avg player value',
  h2h_count: 'Head-to-head meetings',
  h2h_gd_avg: 'Head-to-head goal diff',
  h2h_home_win_rate: 'Head-to-head home wins',
  h2h_away_win_rate: 'Head-to-head away wins',
  h2h_draw_rate: 'Head-to-head draws',
  h2h_avg_goals_home: 'Head-to-head home goals',
  h2h_avg_goals_away: 'Head-to-head away goals',
  tournament_tier: 'Match importance',
  is_neutral: 'Neutral venue',
  is_qualifier: 'Qualifier context',
}

function humanize(feature) {
  if (FEATURE_LABELS[feature]) return FEATURE_LABELS[feature]
  if (/_last5$/.test(feature)) return 'Recent form (5)'
  if (/_last10$/.test(feature)) return 'Recent form (10)'
  return feature.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function StatChip({ label, value }) {
  return (
    <div className="card-elevated px-3 py-2 text-center">
      <div className="text-[10px] uppercase tracking-wide text-text-secondary">{label}</div>
      <div className="stat-num mt-0.5 text-lg font-semibold">{value}</div>
    </div>
  )
}

// Diverging contribution bar centred at zero: green pushes toward the
// predicted outcome, red pushes against it.
function ShapBar({ feature, shap, maxAbs }) {
  const positive = shap >= 0
  const half = maxAbs > 0 ? (Math.abs(shap) / maxAbs) * 50 : 0
  return (
    <div className="flex items-center gap-3">
      <span className="w-32 shrink-0 truncate text-xs text-text-secondary" title={feature}>
        {humanize(feature)}
      </span>
      <div className="relative h-4 flex-1 rounded bg-bg/60">
        <span className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-border" />
        <span
          className={`absolute inset-y-0 ${positive ? 'left-1/2 rounded-r' : 'right-1/2 rounded-l'}`}
          style={{ width: `${half}%`, backgroundColor: positive ? '#10b981' : '#ef4444' }}
        />
      </div>
      <span
        className={`stat-num w-12 text-right text-xs ${positive ? 'text-primary' : 'text-danger'}`}
      >
        {positive ? '+' : ''}
        {shap.toFixed(2)}
      </span>
    </div>
  )
}

export default function MatchDetail({ matchId, home, away }) {
  const { data, loading, error } = useApi(`/api/matches/${matchId}`)

  if (loading)
    return (
      <div className="flex items-center gap-2 px-4 py-6 text-sm text-text-secondary">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        Loading explanation…
      </div>
    )
  if (error)
    return <div className="px-4 py-6 text-sm text-danger">Couldn’t load explanation: {error}</div>

  const features = data.top_features ?? []
  const maxAbs = features.reduce((m, f) => Math.max(m, Math.abs(f.shap_value)), 0)

  return (
    <div className="space-y-5 border-t border-border bg-surface/40 px-4 py-5 sm:px-5">
      {/* Expected goals */}
      <div className="grid grid-cols-2 gap-3">
        <StatChip label={`${home} xG`} value={data.xg_home?.toFixed(2) ?? 'N/A'} />
        <StatChip label={`${away} xG`} value={data.xg_away?.toFixed(2) ?? 'N/A'} />
      </div>

      {/* Plain-language explanation */}
      {data.explanation && (
        <p className="text-sm leading-relaxed text-text-primary/90">{data.explanation}</p>
      )}

      {/* SHAP contributions */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">
            Top factors
          </h4>
          <span className="text-[10px] text-text-secondary">
            <span className="text-primary">green</span> = supports ·{' '}
            <span className="text-danger">red</span> = against
          </span>
        </div>
        <div className="space-y-2">
          {features.slice(0, 5).map((f) => (
            <ShapBar key={f.feature} feature={f.feature} shap={f.shap_value} maxAbs={maxAbs} />
          ))}
        </div>
      </div>
    </div>
  )
}
