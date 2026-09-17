import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to FastAPI during development
      '/chat': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/history': 'http://localhost:8000',
    }
  }
})
