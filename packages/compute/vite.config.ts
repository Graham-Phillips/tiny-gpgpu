import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/index.ts",
      fileName: "tiny-webgpu-compute",
      formats: ["es"]
    },
    outDir: "dist",
    emptyOutDir: false
  }
});
