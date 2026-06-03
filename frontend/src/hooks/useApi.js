import { useEffect, useState } from 'react'
import axios from 'axios'

export const API_BASE = 'http://localhost:8000'

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

    api
      .get(path)
      .then((res) => {
        if (!cancelled) setData(res.data)
      })
      .catch((err) => {
        if (cancelled) return
        const detail =
          err.response?.data?.detail ||
          err.message ||
          'Request failed'
        setError(detail)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps])

  return { data, loading, error }
}
