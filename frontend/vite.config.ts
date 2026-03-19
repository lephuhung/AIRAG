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
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      // Proxy MinIO presigned PUT requests (local dev only)
      // Frontend rewrites presigned URLs from localhost:9000 → /minio-direct/
      '/minio-direct': {
        target: 'http://localhost:9000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/minio-direct/, ''),
        // Large file uploads need extended timeout (default is too short)
        timeout: 600_000,       // 10 minutes socket timeout
        proxyTimeout: 600_000,  // 10 minutes upstream response timeout
      },
    },
  },
})
