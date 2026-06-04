// Static data layer — mirrors backend/main.py exactly so the deployed build
// can serve every endpoint from pre-baked JSON with no API running.
//
// Each getter reproduces the transformation its FastAPI counterpart performs.
// getStaticData(path) is the path-based dispatcher used by useApi.js, so React
// components keep calling useApi('/api/...') unchanged.

import {
  groupStandings,
  matchPredictions,
  bracket,
  knockoutProbabilities,
  matchExplanations,
  overview,
  modelInfo,
  features,
} from '../data/index.js'

// Rounds in tournament order (mirrors KO_ROUNDS in backend/main.py).
const KO_ROUNDS = ['R32', 'R16', 'QF', 'SF', 'Final', 'Winner']
const CONF_RANK = { high: 0, medium: 1, low: 2 }

// Schedule order = insertion order of match_predictions.json.
const MATCH_ORDER = matchPredictions.map((m) => m.match_id)
const MATCH_BY_ID = new Map(matchPredictions.map((m) => [m.match_id, m]))
const ORDER_INDEX = new Map(MATCH_ORDER.map((id, i) => [id, i]))

// lower-cased team -> { group, row } (mirrors DataStore.team_index)
const TEAM_INDEX = new Map()
for (const [grp, rows] of Object.entries(groupStandings)) {
  for (const row of rows) TEAM_INDEX.set(row.team.toLowerCase(), { group: grp, row })
}

function maxProb(m) {
  return Math.max(m.p_home_win, m.p_draw, m.p_away_win)
}

function sortTeamsByPoints(rows) {
  return [...rows].sort((a, b) => b.expected_points - a.expected_points)
}

// --- /api/overview ---------------------------------------------------------
export function getOverview() {
  return overview
}

// --- /api/groups -----------------------------------------------------------
export function getGroups() {
  return Object.keys(groupStandings)
    .sort()
    .map((group) => ({ group, teams: sortTeamsByPoints(groupStandings[group]) }))
}

// --- /api/groups/{group_id} ------------------------------------------------
export function getGroup(groupId) {
  const grp = String(groupId).toUpperCase()
  if (!groupStandings[grp]) {
    throw new Error(`Group '${groupId}' not found`)
  }
  const matches = matchPredictions.filter((m) => m.group === grp)
  return { group: grp, teams: sortTeamsByPoints(groupStandings[grp]), matches }
}

// --- /api/matches ----------------------------------------------------------
export function getMatches({ group, team, sort = 'group' } = {}) {
  let matches = [...matchPredictions]

  if (group) {
    const grp = group.toUpperCase()
    matches = matches.filter((m) => m.group === grp)
  }
  if (team) {
    const t = team.toLowerCase()
    matches = matches.filter((m) => m.home.toLowerCase() === t || m.away.toLowerCase() === t)
  }

  if (sort === 'confidence') {
    matches.sort(
      (a, b) => (CONF_RANK[a.confidence] ?? 3) - (CONF_RANK[b.confidence] ?? 3) || maxProb(b) - maxProb(a),
    )
  } else if (sort === 'group') {
    matches.sort((a, b) => a.group.localeCompare(b.group) || ORDER_INDEX.get(a.match_id) - ORDER_INDEX.get(b.match_id))
  }
  // sort === 'date': preserve schedule (insertion) order.

  return matches
}

// --- /api/matches/{match_id} ----------------------------------------------
export function getMatch(matchId) {
  let m = MATCH_BY_ID.get(matchId)
  if (!m && /^\d+$/.test(String(matchId))) {
    const idx = parseInt(matchId, 10) - 1
    if (idx >= 0 && idx < MATCH_ORDER.length) m = MATCH_BY_ID.get(MATCH_ORDER[idx])
  }
  if (!m) {
    throw new Error(`Match '${matchId}' not found`)
  }
  const exp = matchExplanations[m.match_id] || {}
  return {
    match_id: m.match_id,
    group: m.group,
    home: m.home,
    away: m.away,
    p_home_win: m.p_home_win,
    p_draw: m.p_draw,
    p_away_win: m.p_away_win,
    most_likely_result: m.most_likely_result,
    most_likely_score: m.expected_scoreline,
    xg_home: m.xg_home,
    xg_away: m.xg_away,
    confidence: m.confidence,
    predicted: exp.predicted ?? null,
    explanation: exp.explanation ?? null,
    top_features: exp.top_features ?? [],
  }
}

// --- /api/bracket ----------------------------------------------------------
export function getBracket() {
  return { method: bracket.method, rounds: bracket.rounds }
}

// --- /api/teams/{team_name} -----------------------------------------------
export function getTeam(teamName) {
  const entry = TEAM_INDEX.get(String(teamName).toLowerCase())
  if (!entry) {
    throw new Error(`Team '${teamName}' not found`)
  }
  const { row, group } = entry
  const ko = knockoutProbabilities[row.team] || {}
  const knockout = {}
  for (const r of KO_ROUNDS) if (r in ko) knockout[r] = ko[r]
  return {
    team: row.team,
    group,
    expected_points: row.expected_points,
    expected_gd: row.expected_gd,
    expected_gf: row.expected_gf,
    win_group_prob: row.win_group_prob,
    advance_prob: row.advance_prob,
    avg_finish: row.avg_finish,
    knockout,
  }
}

// --- /api/features ---------------------------------------------------------
export function getFeatures(top = 15) {
  return {
    source: features.source,
    n_features: features.n_features,
    top_features: features.top_features.slice(0, top),
  }
}

// --- /api/model-info -------------------------------------------------------
export function getModelInfo() {
  return modelInfo
}

// --- path dispatcher -------------------------------------------------------
// Maps an API path (the same string components pass to useApi) to its baked
// response. Throws on unknown paths so useApi can surface a clear error.
export function getStaticData(path) {
  const [rawPath, queryStr = ''] = path.split('?')
  const query = Object.fromEntries(new URLSearchParams(queryStr))

  if (rawPath === '/api/overview') return getOverview()
  if (rawPath === '/api/groups') return getGroups()
  if (rawPath === '/api/matches') return getMatches(query)
  if (rawPath === '/api/bracket') return getBracket()
  if (rawPath === '/api/model-info') return getModelInfo()
  if (rawPath === '/api/features') return getFeatures(query.top ? parseInt(query.top, 10) : 15)

  const groupMatch = rawPath.match(/^\/api\/groups\/(.+)$/)
  if (groupMatch) return getGroup(decodeURIComponent(groupMatch[1]))

  const matchMatch = rawPath.match(/^\/api\/matches\/(.+)$/)
  if (matchMatch) return getMatch(decodeURIComponent(matchMatch[1]))

  const teamMatch = rawPath.match(/^\/api\/teams\/(.+)$/)
  if (teamMatch) return getTeam(decodeURIComponent(teamMatch[1]))

  throw new Error(`No static data for '${path}'`)
}
