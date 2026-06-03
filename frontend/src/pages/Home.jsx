import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
import {
  Users,
  ListChecks,
  GitBranch,
  Brain,
  ArrowRight,
  Trophy,
  Cpu,
  Layers,
  Sigma,
} from 'lucide-react'
import { useApi } from '../hooks/useApi.js'
import { COLORS, pct } from '../utils/colors.js'
import { flag } from '../utils/flags.js'
import { Loading, ErrorState } from '../components/States.jsx'

const NAV_CARDS = [
  {
    to: '/groups',
    icon: Users,
    title: 'Groups',
    desc: '12 groups, expected points & qualification odds',
  },
  {
    to: '/matches',
    icon: ListChecks,
    title: 'Matches',
    desc: '72 group-stage predictions with explanations',
  },
  {
    to: '/bracket',
    icon: GitBranch,
    title: 'Bracket',
    desc: 'Knockout path from Round of 32 to the final',
  },
  {
    to: '/about',
    icon: Brain,
    title: 'Model Info',
    desc: 'How the model works and how it performs',
  },
]

function StatCard({ icon: Icon, value, label }) {
  return (
    <div className="card flex items-center gap-4 px-5 py-4">
      <div className="rounded-lg bg-surface-elevated p-2.5 text-primary">
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <div className="stat-num text-2xl font-bold text-text-primary">{value}</div>
        <div className="text-xs text-text-secondary">{label}</div>
      </div>
    </div>
  )
}

// Y-axis tick: flag emoji + team name.
function TeamTick({ x, y, payload }) {
  return (
    <text x={x} y={y} dy={4} textAnchor="end" className="font-sans" fill={COLORS.textPrimary} fontSize={13}>
      <tspan fontSize={15}>{flag(payload.value)}</tspan>
      <tspan dx={6}>{payload.value}</tspan>
    </text>
  )
}

function WinnerTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="card-elevated px-3 py-2 text-sm shadow-lg">
      <div className="font-semibold">
        {flag(d.team)} {d.team}
      </div>
      <div className="stat-num text-primary">{pct(d.win_prob)} to win</div>
      <div className="mt-1 text-xs text-text-secondary">
        Model {pct(d.data_model_prob)} · Market {pct(d.market_prob)}
      </div>
    </div>
  )
}

export default function Home() {
  const { data, loading, error } = useApi('/api/overview')

  if (loading) return <Loading label="Loading tournament overview…" />
  if (error) return <ErrorState error={error} />

  const chartData = data.top_10.map((t) => ({ ...t, prob: +(t.win_prob * 100).toFixed(1) }))

  const stats = [
    { icon: ListChecks, value: data.total_matches, label: 'Matches Predicted' },
    { icon: Sigma, value: data.n_simulations.toLocaleString(), label: 'Simulations' },
    { icon: Layers, value: data.model.n_features, label: 'ML Features' },
    { icon: Cpu, value: '4-Model', label: 'Ensemble' },
  ]

  return (
    <div className="space-y-12">
      {/* Hero */}
      <motion.section
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs text-text-secondary">
          <Trophy className="h-3.5 w-3.5 text-primary" />
          Powered by 50,000 Monte Carlo simulations
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl">
          FIFA World Cup 2026
          <span className="block text-primary">Predictions</span>
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-base text-text-secondary sm:text-lg">
          A Dixon-Coles Poisson model blended with a four-model gradient-boosted
          ensemble, calibrated on 25 years of international results and projected
          across the full 48-team tournament.
        </p>
      </motion.section>

      {/* Stats row */}
      <motion.section
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="grid grid-cols-2 gap-4 lg:grid-cols-4"
      >
        {stats.map((s) => (
          <StatCard key={s.label} {...s} />
        ))}
      </motion.section>

      {/* Winner odds chart */}
      <motion.section
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="card px-4 py-6 sm:px-6"
      >
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-lg font-bold">Championship Odds — Top 10</h2>
          <span className="text-xs text-text-secondary">75% model · 25% market prior</span>
        </div>
        <p className="mb-4 text-sm text-text-secondary">Probability of winning the tournament</p>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 0, right: 48, left: 24, bottom: 0 }}
            barCategoryGap="22%"
          >
            <XAxis type="number" hide domain={[0, 'dataMax']} />
            <YAxis
              type="category"
              dataKey="team"
              width={150}
              tickLine={false}
              axisLine={false}
              tick={<TeamTick />}
              interval={0}
            />
            <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} content={<WinnerTooltip />} />
            <Bar dataKey="prob" radius={[0, 6, 6, 0]} maxBarSize={26}>
              {chartData.map((d, i) => (
                <Cell key={d.team} fill={i === 0 ? COLORS.primary : '#0e8f6e'} />
              ))}
              <LabelList
                dataKey="prob"
                position="right"
                formatter={(v) => `${v}%`}
                className="stat-num"
                fill={COLORS.textPrimary}
                fontSize={12}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </motion.section>

      {/* Navigation cards */}
      <motion.section
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
      >
        {NAV_CARDS.map(({ to, icon: Icon, title, desc }) => (
          <Link
            key={to}
            to={to}
            className="card group px-5 py-5 transition-colors hover:border-primary/50 hover:bg-surface-elevated"
          >
            <Icon className="mb-3 h-6 w-6 text-primary" />
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">{title}</h3>
              <ArrowRight className="h-4 w-4 text-text-secondary transition-transform group-hover:translate-x-1 group-hover:text-primary" />
            </div>
            <p className="mt-1 text-sm text-text-secondary">{desc}</p>
          </Link>
        ))}
      </motion.section>
    </div>
  )
}
