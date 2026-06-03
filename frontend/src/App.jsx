import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import Home from './pages/Home.jsx'
import Groups from './pages/Groups.jsx'
import Matches from './pages/Matches.jsx'
import Bracket from './pages/Bracket.jsx'
import About from './pages/About.jsx'

export default function App() {
  return (
    <div className="min-h-screen bg-bg text-text-primary">
      <Navbar />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/groups" element={<Groups />} />
          <Route path="/matches" element={<Matches />} />
          <Route path="/bracket" element={<Bracket />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <footer className="border-t border-border py-6 text-center text-sm text-text-secondary">
        WC2026 Predictor · Dixon-Coles + gradient-boosted ensemble · 50,000 Monte Carlo simulations
      </footer>
    </div>
  )
}
