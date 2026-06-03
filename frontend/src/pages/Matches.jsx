import { useMemo, useState } from 'react'
import { Search, X } from 'lucide-react'
import { useApi } from '../hooks/useApi.js'
import MatchCard from '../components/MatchCard.jsx'
import { Loading, ErrorState } from '../components/States.jsx'

const GROUPS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
const CONF_RANK = { high: 0, medium: 1, low: 2 }

function maxProb(m) {
  return Math.max(m.p_home_win, m.p_draw, m.p_away_win)
}

export default function Matches() {
  const { data, loading, error } = useApi('/api/matches')
  const [group, setGroup] = useState('All')
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState('confidence') // default: confidence (date sort pending schedule dates)

  const matches = useMemo(() => {
    if (!data) return []
    const q = query.trim().toLowerCase()
    let list = data.filter((m) => {
      if (group !== 'All' && m.group !== group) return false
      if (q && !m.home.toLowerCase().includes(q) && !m.away.toLowerCase().includes(q)) return false
      return true
    })
    list = [...list]
    if (sort === 'confidence') {
      list.sort(
        (a, b) => (CONF_RANK[a.confidence] - CONF_RANK[b.confidence]) || maxProb(b) - maxProb(a),
      )
    } else {
      // group: by letter then by the API's schedule order (preserved from fetch)
      const idx = new Map(data.map((m, i) => [m.match_id, i]))
      list.sort((a, b) => a.group.localeCompare(b.group) || idx.get(a.match_id) - idx.get(b.match_id))
    }
    return list
  }, [data, group, query, sort])

  if (loading) return <Loading label="Loading match predictions…" />
  if (error) return <ErrorState error={error} />

  return (
    <div className="space-y-6">
      <header className="border-b border-border pb-5">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
          Group stage · 72 matches
        </p>
        <h1 className="mt-1 text-3xl font-extrabold tracking-tight sm:text-4xl">
          Match Predictions
        </h1>
        <p className="mt-2 text-sm text-text-secondary">
          Outcome probabilities per match — <span className="text-primary">home win</span> /{' '}
          <span className="text-secondary">draw</span> / <span className="text-danger">away win</span>.
          Click any match for its expected goals and the factors behind the call.
        </p>
      </header>

      {/* Filter / sort bar */}
      <div className="sticky top-[57px] z-20 -mx-2 flex flex-wrap items-center gap-3 rounded-xl border border-border bg-bg/85 px-3 py-3 backdrop-blur">
        <div className="relative w-full min-w-0 sm:w-auto sm:flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-secondary" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search team…"
            className="w-full rounded-lg border border-border bg-surface py-2 pl-9 pr-8 text-sm outline-none transition-colors placeholder:text-text-secondary focus:border-primary/50"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              aria-label="Clear search"
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-text-secondary hover:text-text-primary"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        <select
          value={group}
          onChange={(e) => setGroup(e.target.value)}
          aria-label="Filter by group"
          className="cursor-pointer rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none transition-colors focus:border-primary/50"
        >
          <option value="All">All groups</option>
          {GROUPS.map((g) => (
            <option key={g} value={g}>
              Group {g}
            </option>
          ))}
        </select>

        <div className="flex items-center rounded-lg border border-border bg-surface p-0.5 text-sm">
          {[
            ['confidence', 'Confidence'],
            ['group', 'Group'],
          ].map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => setSort(key)}
              className={`cursor-pointer rounded-md px-3 py-1.5 font-medium transition-colors ${
                sort === key
                  ? 'bg-surface-elevated text-primary'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <p className="text-xs text-text-secondary">
        {matches.length} {matches.length === 1 ? 'match' : 'matches'}
        {group !== 'All' && ` · Group ${group}`}
      </p>

      {matches.length === 0 ? (
        <div className="card px-6 py-16 text-center text-text-secondary">
          No matches found{query && ` for “${query}”`}.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {matches.map((m) => (
            <MatchCard key={m.match_id} match={m} />
          ))}
        </div>
      )}
    </div>
  )
}
