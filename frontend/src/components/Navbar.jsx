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
      <nav className="mx-auto flex max-w-7xl items-center justify-between gap-2 px-3 py-3 sm:px-6 lg:px-8">
        <NavLink to="/" className="flex shrink-0 items-center gap-2 font-bold tracking-tight">
          <Trophy className="h-5 w-5 text-primary" />
          <span className="text-text-primary">
            WC<span className="text-primary">2026</span>
          </span>
        </NavLink>
        <div className="flex min-w-0 items-center justify-end gap-0.5 overflow-x-auto [scrollbar-width:none] sm:gap-1.5 [&::-webkit-scrollbar]:hidden">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.end}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-lg px-2 py-1.5 text-[13px] font-medium transition-colors sm:px-3 sm:text-sm ${
                  isActive
                    ? 'bg-surface-elevated text-primary'
                    : 'text-text-secondary hover:bg-surface hover:text-text-primary'
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
