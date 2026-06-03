import { motion, useReducedMotion } from 'framer-motion'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import {
  Database,
  Cpu,
  Layers,
  Sigma,
  GitBranch,
  ExternalLink,
  ChevronRight,
  FlaskConical,
  BarChart2,
  CircleDot,
  Globe,
} from 'lucide-react'
import { useApi } from '../hooks/useApi.js'
import { COLORS } from '../utils/colors.js'
import { Loading, ErrorState } from '../components/States.jsx'

// ── Feature label humanizer ──────────────────────────────────────────────────
const FEAT_LABELS = {
  elo_expected_home:    'ELO Exp. Home Win',
  elo_diff_adj:         'ELO Diff. (adj.)',
  elo_diff:             'ELO Difference',
  h2h_gd_avg:           'H2H Goal Diff Avg',
  h2h_count:            'H2H Meetings Count',
  away_squad_value:     'Away Squad Value',
  elo_away:             'Away ELO Rating',
  h2h_avg_goals_away:   'H2H Away Goals Avg',
  away_comp_ga_last10:  'Away Comp. GA (L10)',
  home_comp_ga_last10:  'Home Comp. GA (L10)',
  h2h_home_win_rate:    'H2H Home Win Rate',
  home_ga_last10:       'Home GA Last 10',
  squad_value_ratio:    'Squad Value Ratio',
  elo_home_adj:         'Home ELO (adj.)',
  tournament_tier:      'Tournament Tier',
}
const humanize = (f) =>
  FEAT_LABELS[f] ?? f.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

// ── Pipeline steps ───────────────────────────────────────────────────────────
const PIPELINE = [
  {
    icon: Database,
    label: 'Data Collection',
    detail: '21 k+ international matches · squad values · ELO ratings · H2H history',
  },
  {
    icon: Cpu,
    label: 'Feature Engineering',
    detail: '49 features across ELO, form, H2H, squad quality, and tournament context',
  },
  {
    icon: Layers,
    label: 'Dual-Model Ensemble',
    detail: 'Dixon-Coles Poisson + XGBoost / LightGBM / CatBoost / LogReg stack',
  },
  {
    icon: Sigma,
    label: 'Monte Carlo Simulation',
    detail: '50,000 simulated tournaments — full draw, group stage, knockout rounds',
  },
  {
    icon: BarChart2,
    label: 'Live Dashboard',
    detail: 'FastAPI backend · React + Recharts frontend · SHAP explanations per match',
  },
]

// ── Data sources ─────────────────────────────────────────────────────────────
const SOURCES = [
  {
    icon: Globe,
    name: 'International Results',
    desc: 'All international match results since 1872 via martj42/international_results (GitHub).',
    tag: 'Match history',
  },
  {
    icon: CircleDot,
    name: 'ELO Ratings',
    desc: 'Team strength computed from scratch using K-factor rules (WC K=40, friendly K=10) with +100 home advantage.',
    tag: 'Team strength',
  },
  {
    icon: Database,
    name: 'Transfermarkt Squad Values',
    desc: 'Market values per squad from WC-cycle snapshots (2006 – 2026) via salimt/football-datasets.',
    tag: 'Squad quality',
  },
  {
    icon: FlaskConical,
    name: 'FIFA WC 2026 Fixtures',
    desc: 'Official 48-team group-stage schedule with confirmed venues and kick-off times.',
    tag: 'Tournament',
  },
]

// ── Tech stack badges ─────────────────────────────────────────────────────────
const STACK = [
  { label: 'Python 3.9',    color: 'bg-blue-500/15 text-blue-400 ring-blue-500/30' },
  { label: 'XGBoost',       color: 'bg-emerald-500/15 text-emerald-400 ring-emerald-500/30' },
  { label: 'LightGBM',      color: 'bg-emerald-500/15 text-emerald-400 ring-emerald-500/30' },
  { label: 'CatBoost',      color: 'bg-emerald-500/15 text-emerald-400 ring-emerald-500/30' },
  { label: 'Dixon-Coles',   color: 'bg-violet-500/15 text-violet-400 ring-violet-500/30' },
  { label: 'scikit-learn',  color: 'bg-blue-500/15 text-blue-400 ring-blue-500/30' },
  { label: 'SHAP',          color: 'bg-secondary/15 text-secondary ring-secondary/30' },
  { label: 'FastAPI',       color: 'bg-teal-500/15 text-teal-400 ring-teal-500/30' },
  { label: 'React 18',      color: 'bg-cyan-500/15 text-cyan-400 ring-cyan-500/30' },
  { label: 'Recharts',      color: 'bg-cyan-500/15 text-cyan-400 ring-cyan-500/30' },
  { label: 'Framer Motion', color: 'bg-pink-500/15 text-pink-400 ring-pink-500/30' },
  { label: 'Tailwind CSS',  color: 'bg-sky-500/15 text-sky-400 ring-sky-500/30' },
]

// ── Custom tooltip for feature chart ─────────────────────────────────────────
function FeatTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="rounded-lg border border-border bg-surface-elevated px-3 py-2 text-xs shadow-lg">
      <div className="font-semibold text-text-primary">{humanize(d.feature)}</div>
      <div className="stat-num mt-0.5 text-primary">
        {(d.importance * 100).toFixed(2)}% importance
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function About() {
  const { data: info, loading: loadingInfo, error: errorInfo } = useApi('/api/model-info')
  const { data: feat, loading: loadingFeat, error: errorFeat } = useApi('/api/features?top=15')
  const reduce = useReducedMotion()

  if (loadingInfo || loadingFeat) return <Loading label="Loading model info…" />
  if (errorInfo) return <ErrorState error={errorInfo} />
  if (errorFeat) return <ErrorState error={errorFeat} />

  const chartData = feat.top_features.map((f) => ({
    feature: f.feature,
    importance: f.importance,
  }))

  const fadeUp = (i) =>
    reduce
      ? {}
      : {
          initial:    { opacity: 0, y: 16 },
          animate:    { opacity: 1, y: 0 },
          transition: { duration: 0.35, ease: 'easeOut', delay: i * 0.06 },
        }

  const KPI_CARDS = [
    { label: 'Test Accuracy', value: `${(info.accuracy * 100).toFixed(1)}%`,  sub: 'on held-out WC matches' },
    { label: 'Log-Loss',      value: info.log_loss.toFixed(3),                 sub: 'lower is better' },
    { label: 'Brier Score',   value: info.brier.toFixed(3),                    sub: 'probability calibration' },
    { label: 'Features',      value: info.n_features,                          sub: `across ${info.models.length} model families` },
  ]

  return (
    <div className="space-y-10">

      {/* ── Page header ───────────────────────────────────────────────────── */}
      <header className="border-b border-border pb-5">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
          Model info
        </p>
        <h1 className="mt-1 text-3xl font-extrabold tracking-tight sm:text-4xl">
          How the Predictions Work
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-text-secondary">
          A dual-model pipeline combining a Dixon-Coles Poisson model with a gradient-boosted
          ensemble, calibrated on 50,000 Monte Carlo tournament simulations.
        </p>
      </header>

      {/* ── KPI bar ───────────────────────────────────────────────────────── */}
      <section aria-label="Key metrics" className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {KPI_CARDS.map((k, i) => (
          <motion.div key={k.label} {...fadeUp(i)} className="card px-4 py-4">
            <div className="stat-num text-2xl font-extrabold tabular-nums text-primary">
              {k.value}
            </div>
            <div className="mt-1 text-xs font-semibold text-text-primary">{k.label}</div>
            <div className="mt-0.5 text-[11px] text-text-secondary">{k.sub}</div>
          </motion.div>
        ))}
      </section>

      {/* ── Pipeline ──────────────────────────────────────────────────────── */}
      <section aria-labelledby="pipeline-heading">
        <h2 id="pipeline-heading" className="text-sm font-bold uppercase tracking-wide text-text-primary mb-5">
          How It Works
        </h2>
        <div className="relative">
          {/* vertical connector line on sm+ */}
          <div
            aria-hidden
            className="absolute left-5 top-5 hidden h-[calc(100%-2.5rem)] w-px bg-border sm:block"
          />
          <ol className="space-y-3">
            {PIPELINE.map((step, i) => (
              <motion.li
                key={step.label}
                {...fadeUp(i)}
                className="relative flex items-start gap-4"
              >
                <div className="relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-border bg-surface-elevated">
                  <step.icon className="h-4 w-4 text-primary" aria-hidden />
                </div>
                <div className="card flex-1 px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className="stat-num text-[10px] text-text-secondary">
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    <span className="text-sm font-bold text-text-primary">{step.label}</span>
                    {i < PIPELINE.length - 1 && (
                      <ChevronRight
                        className="ml-auto h-4 w-4 shrink-0 text-border"
                        aria-hidden
                      />
                    )}
                  </div>
                  <p className="mt-1 text-xs text-text-secondary">{step.detail}</p>
                </div>
              </motion.li>
            ))}
          </ol>
        </div>
      </section>

      {/* ── Performance metrics ───────────────────────────────────────────── */}
      <section aria-labelledby="perf-heading">
        <h2 id="perf-heading" className="text-sm font-bold uppercase tracking-wide text-text-primary mb-5">
          Model Performance
        </h2>
        <div className="grid gap-4 sm:grid-cols-2">

          {/* Ensemble */}
          <div className="card overflow-hidden">
            <div className="border-b border-border px-4 py-3">
              <h3 className="text-xs font-bold uppercase tracking-wide text-text-primary">
                Gradient-Boosted Ensemble
              </h3>
              <p className="mt-0.5 text-[11px] capitalize text-text-secondary">
                {info.models.join(' · ')} · {info.ensemble_type.replace('_', ' ')}
              </p>
            </div>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border bg-surface text-[10px] uppercase tracking-wide text-text-secondary">
                  <th className="px-4 py-2 text-left font-semibold">Metric</th>
                  <th className="px-4 py-2 text-right font-semibold">Value</th>
                  <th className="px-4 py-2 text-right font-semibold">Set</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                <tr>
                  <td className="px-4 py-2.5 text-text-primary">Log-Loss</td>
                  <td className="stat-num px-4 py-2.5 text-right font-semibold text-primary">
                    {info.log_loss.toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 text-right text-text-secondary">Test</td>
                </tr>
                <tr>
                  <td className="px-4 py-2.5 text-text-primary">Accuracy</td>
                  <td className="stat-num px-4 py-2.5 text-right font-semibold text-primary">
                    {(info.accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-2.5 text-right text-text-secondary">Test</td>
                </tr>
                <tr>
                  <td className="px-4 py-2.5 text-text-primary">Brier Score</td>
                  <td className="stat-num px-4 py-2.5 text-right font-semibold text-primary">
                    {info.brier.toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 text-right text-text-secondary">Test</td>
                </tr>
                <tr>
                  <td className="px-4 py-2.5 text-text-primary">Training date</td>
                  <td
                    className="stat-num px-4 py-2.5 text-right text-text-secondary"
                    colSpan={2}
                  >
                    {info.training_date}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Dixon-Coles backtest */}
          <div className="card overflow-hidden">
            <div className="border-b border-border px-4 py-3">
              <h3 className="text-xs font-bold uppercase tracking-wide text-text-primary">
                Dixon-Coles Poisson Model
              </h3>
              <p className="mt-0.5 text-[11px] text-text-secondary">
                Fitted on {info.dixon_coles.num_matches.toLocaleString()} matches ·{' '}
                {info.dixon_coles.n_teams} teams · γ={info.dixon_coles.gamma.toFixed(3)} ·
                ρ={info.dixon_coles.rho.toFixed(4)}
              </p>
            </div>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border bg-surface text-[10px] uppercase tracking-wide text-text-secondary">
                  <th className="px-4 py-2 text-left font-semibold">Backtest</th>
                  <th className="px-4 py-2 text-right font-semibold">Log-Loss</th>
                  <th className="px-4 py-2 text-right font-semibold">Matches</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                <tr>
                  <td className="px-4 py-2.5 text-text-primary">WC 2018</td>
                  <td className="stat-num px-4 py-2.5 text-right font-semibold text-secondary">
                    {info.dixon_coles.wc2018_log_loss.toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 text-right text-text-secondary">64</td>
                </tr>
                <tr>
                  <td className="px-4 py-2.5 text-text-primary">WC 2022</td>
                  <td className="stat-num px-4 py-2.5 text-right font-semibold text-secondary">
                    {info.dixon_coles.wc2022_log_loss.toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 text-right text-text-secondary">64</td>
                </tr>
                <tr>
                  <td className="px-4 py-2.5 text-text-primary">Post-2018 all intl.</td>
                  <td className="stat-num px-4 py-2.5 text-right font-semibold text-secondary">
                    {info.dixon_coles.post2018_log_loss.toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 text-right text-text-secondary">—</td>
                </tr>
                <tr>
                  <td className="px-4 py-2.5 text-text-primary">Blend weight</td>
                  <td
                    className="stat-num px-4 py-2.5 text-right text-text-secondary"
                    colSpan={2}
                  >
                    DC {(info.combined.dixon_coles_weight * 100).toFixed(1)}% · Ens.{' '}
                    {(info.combined.ensemble_weight * 100).toFixed(3)}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ── Feature importance chart ──────────────────────────────────────── */}
      <section aria-labelledby="feat-heading">
        <h2 id="feat-heading" className="text-sm font-bold uppercase tracking-wide text-text-primary mb-1">
          Feature Importance
        </h2>
        <p className="mb-5 text-xs text-text-secondary">
          Mean L1-normalised importance across XGBoost, LightGBM &amp; CatBoost — top{' '}
          {feat.top_features.length} of {feat.n_features} features.
        </p>
        <div className="card px-4 pb-4 pt-2">
          <ResponsiveContainer width="100%" height={chartData.length * 34 + 16}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 4, right: 64, bottom: 4, left: 180 }}
            >
              <XAxis
                type="number"
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{ fontSize: 10, fill: 'var(--color-text-secondary)' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="feature"
                width={180}
                tickFormatter={(f) => {
                  const label = humanize(f)
                  return label.length > 22 ? label.slice(0, 21) + '…' : label
                }}
                tick={{ fontSize: 11, fill: 'var(--color-text-secondary)' }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<FeatTooltip />} cursor={{ fill: 'rgba(148,163,184,0.06)' }} />
              <Bar dataKey="importance" radius={[0, 3, 3, 0]} maxBarSize={18}>
                {chartData.map((_, i) => (
                  <Cell
                    key={i}
                    fill={i === 0 ? COLORS.primary : COLORS.secondary}
                    fillOpacity={i === 0 ? 1 : Math.max(0.35, 0.9 - i * 0.04)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* ── Data sources ─────────────────────────────────────────────────── */}
      <section aria-labelledby="sources-heading">
        <h2 id="sources-heading" className="text-sm font-bold uppercase tracking-wide text-text-primary mb-5">
          Data Sources
        </h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {SOURCES.map((s, i) => (
            <motion.div key={s.name} {...fadeUp(i)} className="card flex gap-3 px-4 py-4">
              <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-border bg-surface-elevated">
                <s.icon className="h-4 w-4 text-primary" aria-hidden />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-text-primary">{s.name}</span>
                  <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
                    {s.tag}
                  </span>
                </div>
                <p className="mt-1 text-xs text-text-secondary">{s.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Tech stack ───────────────────────────────────────────────────── */}
      <section aria-labelledby="stack-heading">
        <h2 id="stack-heading" className="text-sm font-bold uppercase tracking-wide text-text-primary mb-4">
          Tech Stack
        </h2>
        <div className="flex flex-wrap gap-2">
          {STACK.map((t) => (
            <span
              key={t.label}
              className={`rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-inset ${t.color}`}
            >
              {t.label}
            </span>
          ))}
        </div>
      </section>

      {/* ── GitHub CTA ───────────────────────────────────────────────────── */}
      <section className="card flex flex-col items-center gap-4 px-6 py-8 text-center sm:flex-row sm:text-left">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-border bg-surface-elevated">
          <GitBranch className="h-6 w-6 text-primary" aria-hidden />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="font-bold text-text-primary">View the source code</h3>
          <p className="mt-0.5 text-xs text-text-secondary">
            Full pipeline: data collection, feature engineering, model training, simulation and
            this dashboard — all open source.
          </p>
        </div>
        <a
          href="https://github.com/aabdo95/wc2026-predictor"
          target="_blank"
          rel="noopener noreferrer"
          className="flex shrink-0 items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 px-4 py-2.5 text-sm font-semibold text-primary transition-colors hover:bg-primary/20 focus-visible:outline-2 focus-visible:outline-primary"
        >
          GitHub
          <ExternalLink className="h-3.5 w-3.5" aria-hidden />
        </a>
      </section>

    </div>
  )
}
