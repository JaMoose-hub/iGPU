import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig({
  base: "./",
  plugins: [react()],
  root: "src",
  server: {
    host: "127.0.0.1",
    port: 1420,
    strictPort: true
  },
  build: {
    target: "esnext",
    outDir: "../dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "src/index.html"),
        tools: resolve(__dirname, "src/tools.html"),
        tasks: resolve(__dirname, "src/tasks.html"),
        search: resolve(__dirname, "src/search.html"),
        gamepath: resolve(__dirname, "src/gamepath.html"),
        standby: resolve(__dirname, "src/standby.html"),
        hud: resolve(__dirname, "src/hud.html")
      }
    }
  }
});
