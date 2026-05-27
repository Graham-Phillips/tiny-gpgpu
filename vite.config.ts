import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    alias: {
      "@tiny-webgpu/compute": "/packages/compute/src",
      "@tiny-webgpu/network-anomaly": "/packages/network-anomaly/src"
    }
  },
  test: {
    environment: "node",
    include: ["packages/*/tests/**/*.test.ts"]
  }
});
