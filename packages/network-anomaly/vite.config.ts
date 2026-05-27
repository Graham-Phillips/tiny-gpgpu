import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/index.ts",
      fileName: "tiny-webgpu-network-anomaly",
      formats: ["es"]
    },
    rollupOptions: {
      external: ["@tiny-webgpu/compute"]
    },
    outDir: "dist",
    emptyOutDir: false
  }
});
