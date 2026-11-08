import { defineConfig } from "vite";

// `npm run dev` proxies the websocket to the Python server (python ../server.py).
export default defineConfig({
  server: {
    proxy: { "/ws": { target: "ws://127.0.0.1:8765", ws: true } },
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 2000 },
});
