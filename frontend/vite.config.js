import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server on Vite's default port; the FastAPI backend allows CORS for :5173.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
})
