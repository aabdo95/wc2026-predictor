import { flag } from '../utils/flags.js'

// Standard 32-team seeded bracket: visual row 0..31 maps to these 0-indexed seeds.
// Seed 1 (row 0) and Seed 2 (row 16) are in opposite halves → they meet in the Final.
const BRACKET_SEED_MAP = [
   0, 31, 15, 16,  7, 24,  8, 23,   // top quarter
   3, 28, 12, 19,  4, 27, 11, 20,   // second quarter
   1, 30, 14, 17,  6, 25,  9, 22,   // third quarter
   2, 29, 13, 18,  5, 26, 10, 21,   // bottom quarter
]

const ROUND_KEYS   = ['R32', 'R16', 'QF', 'SF', 'Final', 'Winner']
const ROUND_LABELS = ['Round of 32', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Final', 'Champion']

// Build bracket positions by propagating the most-likely team through each slot.
// cols[k][i] = { team: string, prob: number (for round k) } | null
function buildBracketPositions(rounds) {
  // Prob lookup: team → { R32: n, R16: n, … }
  const probMap = {}
  ROUND_KEYS.forEach((key) => {
    ;(rounds[key] ?? []).forEach((t) => {
      if (!probMap[t.team]) probMap[t.team] = {}
      probMap[t.team][key] = t.prob
    })
  })

  const r32Sorted = (rounds.R32 ?? []).slice(0, 32) // already sorted by prob desc
  // Map each visual row to the correct seeded team
  const r32Col = BRACKET_SEED_MAP.map((s) => {
    const team = r32Sorted[s]?.team ?? null
    return team ? { team, prob: probMap[team]?.R32 ?? 0 } : null
  })

  const cols = [r32Col]
  let prev = r32Col

  for (let ri = 1; ri < ROUND_KEYS.length; ri++) {
    const key = ROUND_KEYS[ri]
    const curr = []
    for (let i = 0; i < prev.length / 2; i++) {
      const a = prev[2 * i]
      const b = prev[2 * i + 1]
      // Pick the candidate with the higher probability of reaching THIS round
      const pa = a ? (probMap[a.team]?.[key] ?? 0) : -1
      const pb = b ? (probMap[b.team]?.[key] ?? 0) : -1
      const winner = pa >= pb ? a : b
      curr.push(winner ? { team: winner.team, prob: probMap[winner.team]?.[key] ?? 0 } : null)
    }
    cols.push(curr)
    prev = curr
  }

  return cols
}

// --- Layout constants -------------------------------------------------------
const SLOT_H    = 22   // px — height of one team chip
const SLOT_GAP  = 2    // px — gap between chips within same round
const UNIT      = SLOT_H + SLOT_GAP  // 24px per bracket "row"
const COL_W     = 142  // px — width of one column
const CONN_W    = 20   // px — width reserved between columns for connector lines
const HEADER_H  = 22   // px — column label height
const PAD_TOP   = HEADER_H + 8

const TOTAL_H = 32 * UNIT + PAD_TOP
const TOTAL_W = ROUND_KEYS.length * COL_W + (ROUND_KEYS.length - 1) * CONN_W  // 952

// Center-Y of team at position i in round k (0-indexed, round 0 = R32 with 32 slots)
function cy(i, k) {
  return (2 * i + 1) * UNIT * Math.pow(2, k) / 2 + PAD_TOP
}
// Left-X of column k
function cx(k) {
  return k * (COL_W + CONN_W)
}

// --- Sub-components ---------------------------------------------------------
const BORDER_COLOR = '#1e293b'
const LINE_COLOR   = '#2d3b52'

function TeamChip({ team, prob, isChampion }) {
  if (!team) return null
  return (
    <div
      className={`flex items-center gap-1.5 rounded px-1.5 ${
        isChampion
          ? 'border border-primary bg-primary/15 font-bold shadow-[0_0_14px_-4px_rgba(16,185,129,0.5)] ring-1 ring-inset ring-primary/40'
          : 'border border-border bg-surface hover:border-border/70'
      }`}
      style={{ height: SLOT_H }}
      title={`${team} — ${Math.round(prob * 100)}%`}
    >
      <span className="shrink-0 text-[13px] leading-none">{flag(team)}</span>
      <span
        className={`min-w-0 flex-1 truncate text-[11px] font-medium ${
          isChampion ? 'text-primary' : 'text-text-primary'
        }`}
      >
        {team}
      </span>
      <span
        className={`stat-num ml-auto shrink-0 text-[10px] ${
          isChampion ? 'text-primary' : 'text-text-secondary'
        }`}
      >
        {Math.round(prob * 100)}%
      </span>
    </div>
  )
}

export default function MostLikelyBracket({ rounds }) {
  const cols = buildBracketPositions(rounds)

  // Connector SVG lines between adjacent rounds
  const lines = []
  for (let k = 0; k < ROUND_KEYS.length - 1; k++) {
    const col = cols[k]
    for (let i = 0; i < col.length; i += 2) {
      const j = Math.floor(i / 2)
      const y0  = cy(i,     k)
      const y1  = cy(i + 1, k)
      const yN  = cy(j,     k + 1)
      const x1  = cx(k) + COL_W          // right edge of current column
      const x2  = cx(k + 1)              // left edge of next column
      const xM  = x1 + CONN_W / 2        // midpoint of connector gap

      lines.push(
        <line key={`rh-${k}-${i}`}   x1={x1}  y1={y0} x2={xM}  y2={y0} />,
        <line key={`rh-${k}-${i+1}`} x1={x1}  y1={y1} x2={xM}  y2={y1} />,
        <line key={`rv-${k}-${j}`}   x1={xM}  y1={y0} x2={xM}  y2={y1} />,
        <line key={`lh-${k}-${j}`}   x1={xM}  y1={yN} x2={x2}  y2={yN} />,
      )
    }
  }

  return (
    <div className="overflow-x-auto pb-2">
      <div style={{ position: 'relative', width: TOTAL_W, height: TOTAL_H }}>

        {/* Column headers */}
        {ROUND_LABELS.map((label, k) => (
          <div
            key={k}
            style={{ position: 'absolute', left: cx(k), top: 0, width: COL_W }}
            className={`truncate text-[10px] font-semibold uppercase tracking-wide ${
              k === 5 ? 'text-primary' : 'text-text-secondary'
            }`}
          >
            {label}
          </div>
        ))}

        {/* SVG connector lines */}
        <svg
          width={TOTAL_W}
          height={TOTAL_H}
          style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
          aria-hidden
        >
          <g stroke={LINE_COLOR} strokeWidth={1} fill="none">
            {lines}
          </g>
          {/* Half-time separator line to visually divide the bracket halves */}
          <line
            x1={0} y1={16 * UNIT + PAD_TOP}
            x2={TOTAL_W} y2={16 * UNIT + PAD_TOP}
            stroke={BORDER_COLOR} strokeWidth={1} strokeDasharray="4 4"
          />
        </svg>

        {/* Team chips */}
        {cols.map((col, k) =>
          col.map((entry, i) => {
            if (!entry) return null
            const isChampion = k === 5
            return (
              <div
                key={`chip-${k}-${i}`}
                style={{
                  position: 'absolute',
                  left: cx(k),
                  top: cy(i, k) - SLOT_H / 2,
                  width: COL_W,
                }}
              >
                <TeamChip team={entry.team} prob={entry.prob} isChampion={isChampion} />
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
