import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vuetify from 'vite-plugin-vuetify'
import compression from 'vite-plugin-compression'

const target = process.env.RETROGRAD_URL ?? 'http://127.0.0.1:8471'

// Compressed siblings are written next to every text asset so the server can
// answer `Accept-Encoding` without compressing at request time.
const compressed = /\.(js|mjs|css|html|json|svg|txt|map)$/i

export default defineConfig({
  plugins: [
    vue(),
    vuetify({ autoImport: true }),
    compression({ algorithm: 'brotliCompress', ext: '.br', filter: compressed, verbose: false }),
    compression({ algorithm: 'gzip', ext: '.gz', filter: compressed, verbose: false }),
  ],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      // No buffering and no timeout: server-sent event streams stay open for
      // the life of a run and must reach the page frame by frame.
      '/v1': { target, changeOrigin: false, timeout: 0, proxyTimeout: 0, ws: false },
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    target: 'es2022',
    manifest: true,
    sourcemap: false,
    // The entry must stay a file, never an inline script: the page is served
    // under `script-src 'self'`.
    assetsInlineLimit: (file) => (file.endsWith('.js') ? false : undefined),
    modulePreload: { polyfill: false },
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (/[\\/](echarts|zrender|vue-echarts)[\\/]/.test(id)) return 'echarts'
          if (/[\\/]vuetify[\\/]/.test(id)) return 'vuetify'
          if (
            /[\\/](markdown-it|dompurify|mdurl|uc\.micro|linkify-it|entities|punycode\.js)[\\/]/.test(
              id,
            )
          )
            return 'markdown'
          return undefined
        },
      },
    },
  },
})
