import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Read .env from project root instead of frontend/
  envDir: '../',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    // Output directory for production build
    outDir: 'dist',
    // Generate sourcemaps for debugging
    sourcemap: false,
    // Minify for production (using esbuild, the default)
    minify: true,
    // Split chunks for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['framer-motion', 'lucide-react'],
        },
      },
    },
  },
  server: {
    // Development server port
    port: 5173,
    // Proxy API calls to FastAPI backend during development
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/chat': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/companies': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/transcript': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
