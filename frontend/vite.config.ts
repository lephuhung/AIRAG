import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5174,
    allowedHosts: ['rag.hatinh.local', 'localhost', '127.0.0.1'],
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_URL || 'http://backend:8080',
        changeOrigin: true,
        // STT/transcribe can be slow on a cold Whisper model load (first call
        // loads large-v3, ~70s). The proxy default timeout is too short and
        // kills the request mid-flight → the browser sees a generic failure.
        timeout: 600_000,       // 10 minutes socket timeout
        proxyTimeout: 600_000,  // 10 minutes upstream response timeout
      },
      '/static': {
        target: process.env.VITE_API_PROXY_URL || 'http://backend:8080',
        changeOrigin: true,
      },
      // Proxy MinIO presigned PUT requests (local dev only)
      // Frontend rewrites presigned URLs from localhost:9000 → /minio-direct/
      '/minio-direct': {
        target: process.env.VITE_MINIO_PROXY_URL || 'http://minio:9000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/minio-direct/, ''),
        // Large file uploads need extended timeout (default is too short)
        timeout: 600_000,       // 10 minutes socket timeout
        proxyTimeout: 600_000,  // 10 minutes upstream response timeout
      },
    },
  },
})
