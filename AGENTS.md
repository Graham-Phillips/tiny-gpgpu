# Development Guide For AI Agents

This repo is a small npm workspace with two libraries and one browser benchmark/demo.

## Package Boundaries

- `packages/compute` is `@tiny-webgpu/compute`.
  - Keep it generic.
  - It may contain WebGPU setup, buffers, typed GPU vectors, kernels, dispatch helpers, readback helpers, and general compute utilities.
  - It must not contain network, FIX, protobuf, packet, anomaly, or visualization concepts.

- `packages/network-anomaly` is `@tiny-webgpu/network-anomaly`.
  - It depends on `@tiny-webgpu/compute`.
  - It owns feature-oriented anomaly logic for network-derived numeric arrays.
  - It should accept arrays or GPU-resident vectors produced by the compute package.
  - It should emit visualization-friendly results, but should not implement the main visualization UI.

- The visualization project is separate. This repo should produce clean scored data, layers, flags, buckets, and compact event outputs that another project can render.

## Important Design Direction

- Prefer feature-first APIs: parsers for FIX/protobuf/packet captures belong in application-specific code or later adapters. Core anomaly functions should operate on numeric typed arrays or GPU-resident vectors.
- Support interactive exploration. Users are expected to load one large dataset, adjust settings repeatedly, run multiple passes, and avoid re-uploading the same data.
- Keep GPU data resident where possible. Read back compact result layers or selected outputs rather than full intermediate arrays.
- Maintain one-shot convenience APIs for ergonomics, but implement serious workflows through resident GPU objects and reusable detectors.

## Benchmarking Rules

- WebGPU performance must be measured in a real browser with a real adapter.
- Node unit tests are useful for CPU helpers and TypeScript coverage, but they do not validate actual GPU execution.
- The browser benchmark at `/benchmark.html` compares pure JavaScript with WebGPU.
- Keep benchmark metrics explicit about what is included:
  - one-shot GPU timing includes upload, setup, dispatch, synchronization, and readback
  - resident rerun timing keeps input data and detector setup on the GPU
- Do not claim performance numbers unless they were measured on the current machine/browser.

## Testing And Build

Run these before handing work back:

```sh
npm run typecheck
npm test
npm run build
```

Use `npm run dev` and open `http://127.0.0.1:5173/benchmark.html` for browser-side benchmark work.

## Coding Notes

- Preserve strict TypeScript.
- Keep public APIs small and explicit.
- Prefer typed arrays and structured GPU buffer wrappers over untyped objects.
- Destroy GPU buffers created by examples/benchmarks after use.
- Do not introduce a frontend framework.
- Do not move visualization concerns into the library packages.
