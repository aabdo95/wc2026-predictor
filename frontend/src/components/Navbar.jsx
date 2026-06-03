import { NavLink } from 'react-router-dom'
import { Trophy } from 'lucide-react'

const links = [
  { to: '/', label: 'Home', end: true },
  { to: '/groups', label: 'Groups' },
  { to: '/matches', label: 'Matches' },
  { to: '/bracket', label: 'Bracket' },
  { to: '/about', label: 'Model' },
]

export default function Navbar() {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-bg/80 backdrop-blur">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        <NavLink to="/" className="flex items-center gap-2 font-bold tracking-tight">
          <Trophy className="h-5 w-5 text-primary" />
          <span className="text-text-primary">
            WC<span className="text-primary">2026</span>
          </span>
        </NavLink>
        <div className="flex items-center gap-1 sm:gap-2">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.end}
              className={({ isActive }) =>
                `rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-surface-elevated text-primary'
                    : 'text-text-secondary hover:text-text-primary hover:bg-surface'
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </div>
      </nav>
    </header>
  )
}
