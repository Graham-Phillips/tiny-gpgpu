var y = Object.defineProperty;
var w = (t, e, r) => e in t ? y(t, e, { enumerable: !0, configurable: !0, writable: !0, value: r }) : t[e] = r;
var i = (t, e, r) => w(t, typeof e != "symbol" ? e + "" : e, r);
import { workgroupCount as u } from "@tiny-webgpu/compute";
function R(t) {
  if (t.length < 2)
    return new Float32Array();
  const e = new Float32Array(t.length - 1);
  for (let r = 1; r < t.length; r += 1)
    e[r - 1] = t[r] - t[r - 1];
  return e;
}
function _(t) {
  for (let e = 1; e < t.length; e += 1)
    if (t[e] < t[e - 1])
      return !1;
  return !0;
}
function h(t, e) {
  return new D(t, e);
}
function p(t, e) {
  return new x(t, e);
}
class D {
  constructor(e, r) {
    i(this, "kernel");
    i(this, "flagsOnlyKernel");
    this.gpu = e, this.values = r, this.kernel = e.createKernel({
      label: "rolling z-score anomalies",
      code: S
    }), this.flagsOnlyKernel = e.createKernel({
      label: "rolling z-score flags",
      code: z
    });
  }
  async run(e) {
    g(e);
    const r = this.gpu.createBufferFromData(
      m(this.values.length, e),
      {
        label: "rolling z-score params",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }
    ), s = this.gpu.createVector(this.values.length, Float32Array, {
      label: "rolling z-score scores"
    }), a = this.gpu.createVector(this.values.length, Uint32Array, {
      label: "rolling z-score flags"
    });
    return this.kernel.run({
      bindings: [this.values.binding(), { buffer: r }, s.binding(), a.binding()],
      workgroups: [u(this.values.length, 64)],
      label: "rolling z-score anomalies"
    }), await this.gpu.device.queue.onSubmittedWorkDone(), r.destroy(), { scores: s, flags: a };
  }
  async runFlags(e) {
    g(e);
    const r = this.gpu.createBufferFromData(
      m(this.values.length, e),
      {
        label: "rolling z-score flags params",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }
    ), s = this.gpu.createVector(this.values.length, Uint32Array, {
      label: "rolling z-score flags"
    });
    return this.flagsOnlyKernel.run({
      bindings: [this.values.binding(), { buffer: r }, s.binding()],
      workgroups: [u(this.values.length, 64)],
      label: "rolling z-score flags"
    }), await this.gpu.device.queue.onSubmittedWorkDone(), r.destroy(), { flags: s };
  }
}
class x {
  constructor(e, r) {
    i(this, "kernel");
    i(this, "flagsOnlyKernel");
    this.gpu = e, this.sortedTimestamps = r, this.kernel = e.createKernel({
      label: "burst density anomalies",
      code: F
    }), this.flagsOnlyKernel = e.createKernel({
      label: "burst density flags",
      code: U
    });
  }
  async run(e) {
    c(e);
    const r = this.gpu.createBufferFromData(
      f(this.sortedTimestamps.length, e),
      {
        label: "burst density params",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }
    ), s = this.gpu.createVector(this.sortedTimestamps.length, Uint32Array, {
      label: "burst density counts"
    }), a = this.gpu.createVector(this.sortedTimestamps.length, Uint32Array, {
      label: "burst density flags"
    });
    return this.kernel.run({
      bindings: [this.sortedTimestamps.binding(), { buffer: r }, s.binding(), a.binding()],
      workgroups: [u(this.sortedTimestamps.length, 64)],
      label: "burst density anomalies"
    }), await this.gpu.device.queue.onSubmittedWorkDone(), r.destroy(), { counts: s, flags: a };
  }
  async runFlags(e) {
    c(e);
    const r = this.gpu.createBufferFromData(
      f(this.sortedTimestamps.length, e),
      {
        label: "burst density flags params",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }
    ), s = this.gpu.createVector(this.sortedTimestamps.length, Uint32Array, {
      label: "burst density flags"
    });
    return this.flagsOnlyKernel.run({
      bindings: [this.sortedTimestamps.binding(), { buffer: r }, s.binding()],
      workgroups: [u(this.sortedTimestamps.length, 64)],
      label: "burst density flags"
    }), await this.gpu.device.queue.onSubmittedWorkDone(), r.destroy(), { flags: s };
  }
}
async function A(t, e, r) {
  b(e);
  const s = t.createVectorFromData(e, Float32Array, {
    label: "rolling z-score values"
  }), n = await h(t, s).run(r), [o, l] = await Promise.all([
    n.scores.read(),
    n.flags.read()
  ]);
  return s.destroy(), n.scores.destroy(), n.flags.destroy(), { scores: o, flags: l };
}
async function k(t, e, r) {
  if (b(e), c(r), !_(e))
    throw new Error("sortedTimestamps must be sorted in ascending order.");
  const s = t.createVectorFromData(e, Float32Array, {
    label: "burst density timestamps"
  }), n = await p(t, s).run(r), [o, l] = await Promise.all([
    n.counts.read(),
    n.flags.read()
  ]);
  return s.destroy(), n.counts.destroy(), n.flags.destroy(), { counts: o, flags: l };
}
function b(t) {
  if (t.length === 0)
    throw new Error("values must contain at least one item.");
}
function g(t) {
  v(t.windowSize, "windowSize"), d(t.threshold, "threshold"), d(t.minStdDev ?? 1e-6, "minStdDev");
}
function c(t) {
  d(t.windowSize, "windowSize"), v(t.minCount, "minCount");
}
function m(t, e) {
  const r = new ArrayBuffer(16), s = new Uint32Array(r), a = new Float32Array(r);
  return s[0] = t, s[1] = e.windowSize, a[2] = e.threshold, a[3] = e.minStdDev ?? 1e-6, r;
}
function f(t, e) {
  const r = new ArrayBuffer(16), s = new Uint32Array(r), a = new Float32Array(r);
  return s[0] = t, s[1] = e.minCount, a[2] = e.windowSize, r;
}
function v(t, e) {
  if (!Number.isInteger(t) || t <= 0)
    throw new Error(`${e} must be a positive integer.`);
}
function d(t, e) {
  if (!Number.isFinite(t) || t <= 0)
    throw new Error(`${e} must be a positive number.`);
}
const S = (
  /* wgsl */
  `
struct Params {
  length: u32,
  window_size: u32,
  threshold: f32,
  min_std_dev: f32,
}

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= params.length) {
    return;
  }

  let start = select(0u, index - params.window_size, index > params.window_size);
  let count = index - start;

  if (count < 2u) {
    scores[index] = 0.0;
    flags[index] = 0u;
    return;
  }

  var sum = 0.0;
  var cursor = start;

  loop {
    if (cursor >= index) {
      break;
    }

    sum = sum + values[cursor];
    cursor = cursor + 1u;
  }

  let mean = sum / f32(count);
  var variance_sum = 0.0;
  cursor = start;

  loop {
    if (cursor >= index) {
      break;
    }

    let delta = values[cursor] - mean;
    variance_sum = variance_sum + (delta * delta);
    cursor = cursor + 1u;
  }

  let std_dev = max(sqrt(variance_sum / f32(count)), params.min_std_dev);
  let score = abs(values[index] - mean) / std_dev;
  scores[index] = score;
  flags[index] = select(0u, 1u, score >= params.threshold);
}
`
), F = (
  /* wgsl */
  `
struct Params {
  length: u32,
  min_count: u32,
  window_size: f32,
  padding: f32,
}

@group(0) @binding(0) var<storage, read> timestamps: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= params.length) {
    return;
  }

  let current = timestamps[index];
  var count = 0u;
  var cursor = i32(index);

  loop {
    if (cursor < 0) {
      break;
    }

    let timestamp = timestamps[u32(cursor)];

    if ((current - timestamp) > params.window_size) {
      break;
    }

    count = count + 1u;
    cursor = cursor - 1;
  }

  counts[index] = count;
  flags[index] = select(0u, 1u, count >= params.min_count);
}
`
), z = (
  /* wgsl */
  `
struct Params {
  length: u32,
  window_size: u32,
  threshold: f32,
  min_std_dev: f32,
}

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= params.length) {
    return;
  }

  let start = select(0u, index - params.window_size, index > params.window_size);
  let count = index - start;

  if (count < 2u) {
    flags[index] = 0u;
    return;
  }

  var sum = 0.0;
  var cursor = start;

  loop {
    if (cursor >= index) {
      break;
    }

    sum = sum + values[cursor];
    cursor = cursor + 1u;
  }

  let mean = sum / f32(count);
  var variance_sum = 0.0;
  cursor = start;

  loop {
    if (cursor >= index) {
      break;
    }

    let delta = values[cursor] - mean;
    variance_sum = variance_sum + (delta * delta);
    cursor = cursor + 1u;
  }

  let std_dev = max(sqrt(variance_sum / f32(count)), params.min_std_dev);
  let score = abs(values[index] - mean) / std_dev;
  flags[index] = select(0u, 1u, score >= params.threshold);
}
`
), U = (
  /* wgsl */
  `
struct Params {
  length: u32,
  min_count: u32,
  window_size: f32,
  padding: f32,
}

@group(0) @binding(0) var<storage, read> timestamps: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= params.length) {
    return;
  }

  let current = timestamps[index];
  var count = 0u;
  var cursor = i32(index);

  loop {
    if (cursor < 0) {
      break;
    }

    let timestamp = timestamps[u32(cursor)];

    if ((current - timestamp) > params.window_size) {
      break;
    }

    count = count + 1u;
    cursor = cursor - 1;
  }

  flags[index] = select(0u, 1u, count >= params.min_count);
}
`
);
class P {
  constructor(e) {
    i(this, "kind", "webgpu");
    this.gpu = e;
  }
  rollingZScore(e, r) {
    return A(this.gpu, e, r);
  }
  burstDensity(e, r) {
    return k(this.gpu, e, r);
  }
  async createDataset(e) {
    return new B(this.gpu, e);
  }
}
function G(t) {
  return new P(t);
}
class B {
  constructor(e, r) {
    i(this, "backend", "webgpu");
    i(this, "itemCount");
    i(this, "interArrivalMs");
    i(this, "timestampsMs");
    i(this, "rollingDetector");
    i(this, "burstDetector");
    var s, a;
    if (this.gpu = e, !r.interArrivalMs && !r.timestampsMs)
      throw new Error("At least one feature array is required.");
    this.interArrivalMs = r.interArrivalMs ? e.createVectorFromData(r.interArrivalMs, Float32Array, {
      label: "resident inter-arrival times"
    }) : void 0, this.timestampsMs = r.timestampsMs ? e.createVectorFromData(r.timestampsMs, Float32Array, {
      label: "resident timestamps"
    }) : void 0, this.itemCount = ((s = r.interArrivalMs) == null ? void 0 : s.length) ?? ((a = r.timestampsMs) == null ? void 0 : a.length) ?? 0;
  }
  async rollingZScore(e) {
    const r = await this.getRollingDetector().run(e), [s, a] = await Promise.all([r.scores.read(), r.flags.read()]);
    return r.scores.destroy(), r.flags.destroy(), { scores: s, flags: a };
  }
  async rollingZScoreFlags(e) {
    const r = await this.getRollingDetector().runFlags(e), s = await r.flags.read();
    return r.flags.destroy(), s;
  }
  async burstDensity(e) {
    const r = await this.getBurstDetector().run(e), [s, a] = await Promise.all([r.counts.read(), r.flags.read()]);
    return r.counts.destroy(), r.flags.destroy(), { counts: s, flags: a };
  }
  async burstDensityFlags(e) {
    const r = await this.getBurstDetector().runFlags(e), s = await r.flags.read();
    return r.flags.destroy(), s;
  }
  destroy() {
    var e, r;
    (e = this.interArrivalMs) == null || e.destroy(), (r = this.timestampsMs) == null || r.destroy();
  }
  getRollingDetector() {
    if (!this.interArrivalMs)
      throw new Error("interArrivalMs is required for rolling z-score analysis.");
    return this.rollingDetector ?? (this.rollingDetector = h(this.gpu, this.interArrivalMs)), this.rollingDetector;
  }
  getBurstDetector() {
    if (!this.timestampsMs)
      throw new Error("timestampsMs is required for burst-density analysis.");
    return this.burstDetector ?? (this.burstDetector = p(this.gpu, this.timestampsMs)), this.burstDetector;
  }
}
function V(t, e, r, s) {
  return {
    id: t,
    kind: e,
    length: r.length,
    scores: s,
    flags: r
  };
}
function C(t, e) {
  var s, a;
  const r = [];
  for (let n = 0; n < t.length; n += 1) {
    if (t[n] === 0)
      continue;
    const o = ((s = e.scores) == null ? void 0 : s[n]) ?? t[n];
    r.push({
      index: n,
      timestampMs: (a = e.timestampsMs) == null ? void 0 : a[n],
      score: o,
      kind: e.kind,
      severity: M(o, e.lowThreshold, e.highThreshold),
      feature: e.feature
    });
  }
  return r;
}
function K(t, e, r) {
  return {
    id: t,
    kind: e,
    events: r
  };
}
function M(t, e = 3, r = 6) {
  return t >= r ? "high" : t >= e ? "medium" : "low";
}
export {
  x as GpuBurstDensityDetector,
  D as GpuRollingZScoreDetector,
  P as WebGpuAnomalyBackend,
  B as WebGpuResidentDataset,
  k as burstDensityAnomalies,
  p as createBurstDensityDetector,
  V as createDenseLayer,
  h as createRollingZScoreDetector,
  K as createSparseLayer,
  G as createWebGpuAnomalyBackend,
  C as flagsToEvents,
  _ as isSortedAscending,
  R as packetInterArrivalTimes,
  A as rollingZScoreAnomalies,
  M as severityFromScore
};
