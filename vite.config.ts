import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import crypto from 'node:crypto'

// Node 18 compatibility polyfill for crypto.hash
if (typeof (crypto as any).hash !== 'function') {
  (crypto as any).hash = function (algorithm: string, data: crypto.BinaryLike, outputEncoding: crypto.BinaryToTextEncoding = 'hex') {
    return crypto.createHash(algorithm).update(data).digest(outputEncoding)
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-three': ['three', '@react-three/fiber', '@react-three/drei'],
        }
      }
    },
    chunkSizeWarningLimit: 1200,
  }
})
