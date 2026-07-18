import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

const apiProxy = process.env.VITE_API_PROXY || 'http://127.0.0.1:8780'
// Compose sets these inside the web container; bind mounts often miss inotify.
const inDocker =
  process.env.CHOKIDAR_USEPOLLING === 'true' || Boolean(process.env.VITE_API_PROXY)
const hmrHost = process.env.VITE_HMR_HOST || 'localhost'

export default defineConfig({
  plugins: [
    vue(),
    {
      // Legacy Python HTML viewers → Vue project workspace. Keep
      // /watch/.../events and /frames proxied to the playground API.
      name: 'redirect-legacy-watch-html',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const raw = req.url || ''
          const path = raw.split('?')[0] || ''
          const m = path.match(/^\/watch\/([^/]+)\/?$/)
          if (!m) {
            next()
            return
          }
          const name = decodeURIComponent(m[1])
          res.statusCode = 302
          res.setHeader('Location', `/projects/${encodeURIComponent(name)}`)
          res.end()
        })
      },
    },
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    watch: {
      usePolling: inDocker,
      interval: Number(process.env.CHOKIDAR_INTERVAL || 300),
    },
    hmr: inDocker
      ? {
          // Browser is on the Mac host; container IP (OrbStack) breaks the WS.
          protocol: 'ws',
          host: hmrHost,
          port: 5173,
          clientPort: 5173,
        }
      : true,
    proxy: {
      '/api': { target: apiProxy, changeOrigin: true },
      '/watch': { target: apiProxy, changeOrigin: true },
      '/theater': { target: apiProxy, changeOrigin: true },
      '/gens': { target: apiProxy, changeOrigin: true },
    },
  },
})
