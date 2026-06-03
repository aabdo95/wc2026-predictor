import { COLORS } from '../utils/colors.js'

// Stacked outcome-probability bar: home win (green) · draw (amber) · away win
// (red). Percentages are overlaid inside any segment wide enough to hold them.
const SEGMENTS = [
  { key: 'home', color: COLORS.primary },
  { key: 'draw', color: COLORS.secondary },
  { key: 'away', color: COLORS.danger },
]

export default function ConfidenceMeter({ home, draw, away, height = 'h-7' }) {
  const values = { home, draw, away }
  const label = `Home win ${Math.round(home * 100)}%, draw ${Math.round(
    draw * 100,
  )}%, away win ${Math.round(away * 100)}%`

  return (
    <div
      className={`flex w-full ${height} overflow-hidden rounded-md ring-1 ring-inset ring-black/20`}
      role="img"
      aria-label={label}
    >
      {SEGMENTS.map(({ key, color }) => {
        const p = values[key]
        const widthPct = p * 100
        return (
          <div
            key={key}
            className="flex items-center justify-center overflow-hidden"
            style={{ width: `${widthPct}%`, backgroundColor: color }}
          >
            {widthPct >= 16 && (
              <span className="stat-num px-1 text-[11px] font-semibold text-black/80">
                {Math.round(widthPct)}%
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}
