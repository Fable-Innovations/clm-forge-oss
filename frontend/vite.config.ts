import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // bind IPv4 explicitly: "localhost" can resolve to ::1 only, so 127.0.0.1:5173 would refuse
    host: "127.0.0.1",
    port: 5173,
    // the page and /api (proxied below) are one origin: no CORS grant from the dev server either
    cors: false,
    proxy: {
      "/api": { target: `http://127.0.0.1:${process.env.FORGE_PORT || 8765}`, changeOrigin: false },
    },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1500 },
});
