import { useEffect, useState } from 'react'
import axios from 'axios'
import { getStaticData } from './useStaticData.js'

export const API_BASE = 'http://localhost:8000'

// When VITE_USE_STATIC=true (production / Vercel build), every request is served
// from pre-baked JSON and no network call is made. In development it is false,
// so the live FastAPI backend on :8000 is used — but if that backend is
// unreachable, we transparently fall back to the same static data.
const USE_STATIC = import.meta.env.VITE_USE_STATIC === 'true'

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
})

/**
 * Fetch a backend endpoint and track loading/error state.
 *
 * @param {string|null} path  API path (e.g. "/api/overview"). Pass null to skip.
 * @param {Array} deps        extra dependencies that should re-trigger the fetch.
 * @returns {{data: any, loading: boolean, error: string|null}}
 */
export function useApi(path, deps = []) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(Boolean(path))
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!path) return
    let cancelled = false
    setLoading(true)
    setError(null)

    async function load() {
      // Deployed build: resolve from bundled static data, no network.
      if (USE_STATIC) {
        try {
          const d = getStaticData(path)
          if (!cancelled) setData(d)
        } catch (err) {
          if (!cancelled) setError(err.message || 'No static data available')
        } finally {
          if (!cancelled) setLoading(false)
        }
        return
      }

      // Dev build: hit the live API, but fall back to static data on failure
      // (e.g. backend not running) so the dashboard still renders.
      try {
        const res = await api.get(path)
        if (!cancelled) setData(res.data)
      } catch (err) {
        try {
          const d = getStaticData(path)
          if (!cancelled) setData(d)
        } catch {
          if (!cancelled) {
            const detail = err.response?.data?.detail || err.message || 'Request failed'
            setError(detail)
          }
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps])

  return { data, loading, error }
}
