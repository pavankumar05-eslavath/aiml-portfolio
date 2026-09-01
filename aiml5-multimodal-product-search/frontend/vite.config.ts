import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

/**
 * Vite configuration.
 *
 * The dev server proxies `/api` to the backend so the browser only ever talks to
 * one origin during development. That keeps cookies and relative image URLs
 * working and means CORS is not on the critical path locally. In the container
 * build, nginx performs the same proxying (see `nginx.conf`), so the frontend
 * code never needs to know the backend's address.
 */
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': new URL('./src', import.meta.url).pathname },
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Split the framework and icon library out of the application chunk so a
        // change to app code does not invalidate the vendor bundle in browser caches.
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (/[\\/]node_modules[\\/](react|react-dom|react-router|react-router-dom|scheduler)[\\/]/.test(id)) {
              return 'vendor'
            }
            if (id.includes('lucide-react')) return 'icons'
          }
          return undefined
        },
      },
    },
  },
})
