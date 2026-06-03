// Design-system colors as JS constants, for use where Tailwind classes can't
// reach — e.g. Recharts fills/strokes and inline SVG.
export const COLORS = {
  bg: '#0a0e17',
  surface: '#111827',
  surfaceElevated: '#1a2332',
  border: '#1e293b',
  primary: '#10b981', // green
  secondary: '#f59e0b', // amber
  danger: '#ef4444', // red
  textPrimary: '#f1f5f9',
  textSecondary: '#94a3b8',
}

// Outcome → color (home win = green, draw = amber, away win = red).
export const OUTCOME_COLORS = {
  home_win: COLORS.primary,
  draw: COLORS.secondary,
  away_win: COLORS.danger,
}

// Confidence badge → color.
export const CONFIDENCE_COLORS = {
  high: COLORS.primary,
  medium: COLORS.secondary,
  low: COLORS.danger,
}

export function pct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return 'N/A'
  return `${(x * 100).toFixed(digits)}%`
}
