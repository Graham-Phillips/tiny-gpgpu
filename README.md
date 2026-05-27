# Tiny WebGPU Workspace

A small npm workspace for browser WebGPU compute.

@tiny-gpgpu/compute - generic low level layer. Wraps WEbGPU setup, buffer creation, typed GPU vectors, readback, WGSL compute pipeline creation, dispatch and workgroup helpers. Stays close to native WebGPU: users provide WGSL and explicit bindings. Future may provide a simple DSL and interpreter.

@tiny-gpgpu/network-anomaly - ought to be in a separate repo, here for convenience for now. Includes rolling z-score anomaly detection and burst-density detection, with both one-shot and resident GPU dfetector apis fpr repeated interactive runs.



This repo is split into two libraries:

- `@tiny-webgpu/compute`: a generic WebGPU compute wrapper.
- `@tiny-webgpu/network-anomaly`: network-oriented anomaly helpers built on top of `@tiny-webgpu/compute`.

The dependency direction is intentionally one-way:

```text
@tiny-webgpu/network-anomaly -> @tiny-webgpu/compute
@tiny-webgpu/compute -> no domain-specific dependencies
```

## Install

```sh
npm install
```

## Develop

```sh
npm run dev
```

The demo runs a browser benchmark that compares a pure JavaScript rolling z-score anomaly pass with the WebGPU path from `@tiny-webgpu/network-anomaly`. WebGPU requires a secure context, but `localhost` and `127.0.0.1` are treated as secure by browsers.

Open either:

```text
http://127.0.0.1:5173/
http://127.0.0.1:5173/benchmark.html
```

The reported WebGPU time includes buffer upload, compute dispatch, queue synchronization, and readback. That makes it useful for end-to-end browser performance decisions rather than only measuring shader execution.

The benchmark reports both one-shot and resident speedups. One-shot runs include upload/setup/dispatch/readback. Resident runs upload the input data once, keep it in a `GpuVector`, reuse a detector, and only change run parameters plus read back result buffers.

Benchmark modes:

- `Rolling z-score`: JavaScript vs WebGPU one-shot vs WebGPU resident, reading scores and flags.
- `Resident multi-pass`: several resident rolling z-score passes with different settings, modeling interactive exploration.
- `Resident flags-only`: rolling z-score that emits only anomaly flags, avoiding score-buffer creation/readback.
- `Burst density`: JavaScript vs WebGPU one-shot vs WebGPU resident for dense timestamp clusters.

## Build

```sh
npm run build
```

This builds both packages into their own `dist` directories.

## Compute Package

`@tiny-webgpu/compute` stays close to native WebGPU. It handles device setup, buffer upload/readback, compute pipeline creation, dispatch, and workgroup helpers without hiding WGSL or the underlying `GPUDevice`.

```ts
import { initWebGPU, workgroupCount } from "@tiny-webgpu/compute";

const shader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&output)) {
    return;
  }
  output[i] = input[i] * 2.0;
}
`;

const gpu = await initWebGPU();
const input = new Float32Array([1, 2, 3, 4]);
const inputBuffer = gpu.createBufferFromData(input);
const outputBuffer = gpu.createBuffer(input.byteLength);
const kernel = gpu.createKernel({ code: shader });

kernel.run({
  bindings: [{ buffer: inputBuffer }, { buffer: outputBuffer }],
  workgroups: [workgroupCount(input.length, 64)]
});

await gpu.device.queue.onSubmittedWorkDone();
const output = await gpu.readBuffer(outputBuffer, input.byteLength, Float32Array);
```

## Network Anomaly Package

`@tiny-webgpu/network-anomaly` is feature-first: parse network, FIX, protobuf, or packet data in application code, extract numeric features, then run GPU kernels over arrays such as timestamps, inter-arrival times, message sizes, per-flow counters, or encoded field values.

The package now exposes an architecture boundary for future WebGPU/WASM/JS backends:

```ts
import { createWebGpuAnomalyBackend } from "@tiny-webgpu/network-anomaly";

const backend = createWebGpuAnomalyBackend(gpu);
const dataset = await backend.createDataset({
  timestampsMs: packetTimestampsMs,
  interArrivalMs
});

const timing = await dataset.rollingZScore({ windowSize: 32, threshold: 3 });
const burstFlags = await dataset.burstDensityFlags({ windowSize: 10, minCount: 5 });

dataset.destroy();
```

The direct functions remain available for simple one-shot usage.

```ts
import { initWebGPU } from "@tiny-webgpu/compute";
import {
  burstDensityAnomalies,
  packetInterArrivalTimes,
  rollingZScoreAnomalies
} from "@tiny-webgpu/network-anomaly";

const gpu = await initWebGPU();

const packetTimestampsMs = new Float32Array([0, 2, 4, 80, 82, 83, 84, 85]);
const interArrivalMs = packetInterArrivalTimes(packetTimestampsMs);

const timing = await rollingZScoreAnomalies(gpu, interArrivalMs, {
  windowSize: 32,
  threshold: 3
});

const bursts = await burstDensityAnomalies(gpu, packetTimestampsMs, {
  windowSize: 10,
  minCount: 5
});
```

For repeated interactive runs over the same dataset, keep the feature array resident on the GPU:

```ts
import { initWebGPU } from "@tiny-webgpu/compute";
import { createRollingZScoreDetector } from "@tiny-webgpu/network-anomaly";

const gpu = await initWebGPU();
const values = gpu.createVectorFromData(interArrivalMs, Float32Array);
const detector = createRollingZScoreDetector(gpu, values);

const first = await detector.run({ windowSize: 32, threshold: 3 });
const second = await detector.run({ windowSize: 64, threshold: 4 });

const secondFlags = await second.flags.read();

first.scores.destroy();
first.flags.destroy();
second.scores.destroy();
second.flags.destroy();
values.destroy();
gpu.destroy();
```

Current anomaly helpers:

- `createWebGpuAnomalyBackend(gpu)` creates the current backend implementation behind a backend-oriented API.
- `AnomalyBackend` and `ResidentAnomalyDataset` define the future WebGPU/WASM/JS boundary.
- `AnomalyEvent`, `AnomalyLayer`, `DenseAnomalyLayer`, and `SparseAnomalyLayer` define visualization-facing result shapes.
- `flagsToEvents(...)` converts dense flags/scores into sparse events for markers or drill-down.
- `rollingZScoreAnomalies(gpu, values, options)` flags timing or numeric feature spikes against a trailing window.
- `createRollingZScoreDetector(gpu, values)` reuses a GPU-resident input vector and compiled detector for repeated runs.
- `GpuRollingZScoreDetector#runFlags(options)` emits only flags for lower readback cost.
- `burstDensityAnomalies(gpu, sortedTimestamps, options)` flags dense event clusters in a sliding time window.
- `createBurstDensityDetector(gpu, sortedTimestamps)` reuses GPU-resident timestamps for repeated burst-density runs.
- `packetInterArrivalTimes(timestamps)` converts sorted packet/message timestamps into deltas.
