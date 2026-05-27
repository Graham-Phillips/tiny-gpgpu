import { type GpuVector, type WebGPUCompute, type WebGPUKernel, workgroupCount } from "@tiny-webgpu/compute";

export interface RollingZScoreOptions {
  windowSize: number;
  threshold: number;
  minStdDev?: number;
}

export interface RollingZScoreResult {
  scores: Float32Array;
  flags: Uint32Array;
}

export interface GpuRollingZScoreResult {
  scores: GpuVector<Float32Array>;
  flags: GpuVector<Uint32Array>;
}

export interface GpuFlagsResult {
  flags: GpuVector<Uint32Array>;
}

export interface BurstDensityOptions {
  windowSize: number;
  minCount: number;
}

export interface BurstDensityResult {
  counts: Uint32Array;
  flags: Uint32Array;
}

export interface GpuBurstDensityResult {
  counts: GpuVector<Uint32Array>;
  flags: GpuVector<Uint32Array>;
}

export function packetInterArrivalTimes(timestamps: Float32Array): Float32Array {
  if (timestamps.length < 2) {
    return new Float32Array();
  }

  const deltas = new Float32Array(timestamps.length - 1);

  for (let index = 1; index < timestamps.length; index += 1) {
    deltas[index - 1] = timestamps[index] - timestamps[index - 1];
  }

  return deltas;
}

export function isSortedAscending(values: Float32Array): boolean {
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] < values[index - 1]) {
      return false;
    }
  }

  return true;
}

export function createRollingZScoreDetector(
  gpu: WebGPUCompute,
  values: GpuVector<Float32Array>
): GpuRollingZScoreDetector {
  return new GpuRollingZScoreDetector(gpu, values);
}

export function createBurstDensityDetector(
  gpu: WebGPUCompute,
  sortedTimestamps: GpuVector<Float32Array>
): GpuBurstDensityDetector {
  return new GpuBurstDensityDetector(gpu, sortedTimestamps);
}

export class GpuRollingZScoreDetector {
  private readonly kernel: WebGPUKernel;
  private readonly flagsOnlyKernel: WebGPUKernel;

  constructor(
    private readonly gpu: WebGPUCompute,
    private readonly values: GpuVector<Float32Array>
  ) {
    this.kernel = gpu.createKernel({
      label: "rolling z-score anomalies",
      code: rollingZScoreShader
    });
    this.flagsOnlyKernel = gpu.createKernel({
      label: "rolling z-score flags",
      code: rollingZScoreFlagsOnlyShader
    });
  }

  async run(options: RollingZScoreOptions): Promise<GpuRollingZScoreResult> {
    validateRollingZScoreOptions(options);

    const paramsBuffer = this.gpu.createBufferFromData(
      createRollingZScoreParams(this.values.length, options),
      {
        label: "rolling z-score params",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }
    );
    const scores = this.gpu.createVector(this.values.length, Float32Array, {
      label: "rolling z-score scores"
    });
    const flags = this.gpu.createVector(this.values.length, Uint32Array, {
      label: "rolling z-score flags"
    });

    this.kernel.run({
      bindings: [this.values.binding(), { buffer: paramsBuffer }, scores.binding(), flags.binding()],
      workgroups: [workgroupCount(this.values.length, 64)],
      label: "rolling z-score anomalies"
    });

    await this.gpu.device.queue.onSubmittedWorkDone();
    paramsBuffer.destroy();

    return { scores, flags };
  }

  async runFlags(options: RollingZScoreOptions): Promise<GpuFlagsResult> {
    validateRollingZScoreOptions(options);

    const paramsBuffer = this.gpu.createBufferFromData(
      createRollingZScoreParams(this.values.length, options),
      {
        label: "rolling z-score flags params",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }
    );
    const flags = this.gpu.createVector(this.values.length, Uint32Array, {
      label: "rolling z-score flags"
    });

    this.flagsOnlyKernel.run({
      bindings: [this.values.binding(), { buffer: paramsBuffer }, flags.binding()],
      workgroups: [workgroupCount(this.values.length, 64)],
      label: "rolling z-score flags"
    });

    await this.gpu.device.queue.onSubmittedWorkDone();
    paramsBuffer.destroy();

    return { flags };
  }
}

export class GpuBurstDensityDetector {
  private readonly kernel: WebGPUKernel;
  private readonly flagsOnlyKernel: WebGPUKernel;

  constructor(
    private readonly gpu: WebGPUCompute,
    private readonly sortedTimestamps: GpuVector<Float32Array>
  ) {
    this.kernel = gpu.createKernel({
      label: "burst density anomalies",
      code: burstDensityShader
    });
    this.flagsOnlyKernel = gpu.createKernel({
      label: "burst density flags",
      code: burstDensityFlagsOnlyShader
    });
  }

  async run(options: BurstDensityOptions): Promise<GpuBurstDensityResult> {
    validateBurstDensityOptions(options);

    const paramsBuffer = this.gpu.createBufferFromData(
      createBurstDensityParams(this.sortedTimestamps.length, options),
      {
        label: "burst density params",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }
    );
    const counts = this.gpu.createVector(this.sortedTimestamps.length, Uint32Array, {
      label: "burst density counts"
    });
    const flags = this.gpu.createVector(this.sortedTimestamps.length, Uint32Array, {
      label: "burst density flags"
    });

    this.kernel.run({
      bindings: [this.sortedTimestamps.binding(), { buffer: paramsBuffer }, counts.binding(), flags.binding()],
      workgroups: [workgroupCount(this.sortedTimestamps.length, 64)],
      label: "burst density anomalies"
    });

    await this.gpu.device.queue.onSubmittedWorkDone();
    paramsBuffer.destroy();

    return { counts, flags };
  }

  async runFlags(options: BurstDensityOptions): Promise<GpuFlagsResult> {
    validateBurstDensityOptions(options);

    const paramsBuffer = this.gpu.createBufferFromData(
      createBurstDensityParams(this.sortedTimestamps.length, options),
      {
        label: "burst density flags params",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      }
    );
    const flags = this.gpu.createVector(this.sortedTimestamps.length, Uint32Array, {
      label: "burst density flags"
    });

    this.flagsOnlyKernel.run({
      bindings: [this.sortedTimestamps.binding(), { buffer: paramsBuffer }, flags.binding()],
      workgroups: [workgroupCount(this.sortedTimestamps.length, 64)],
      label: "burst density flags"
    });

    await this.gpu.device.queue.onSubmittedWorkDone();
    paramsBuffer.destroy();

    return { flags };
  }
}

export async function rollingZScoreAnomalies(
  gpu: WebGPUCompute,
  values: Float32Array,
  options: RollingZScoreOptions
): Promise<RollingZScoreResult> {
  validateValues(values);
  const valuesVector = gpu.createVectorFromData(values, Float32Array, {
    label: "rolling z-score values"
  });
  const detector = createRollingZScoreDetector(gpu, valuesVector);
  const residentResult = await detector.run(options);
  const [scores, flags] = await Promise.all([
    residentResult.scores.read(),
    residentResult.flags.read()
  ]);

  valuesVector.destroy();
  residentResult.scores.destroy();
  residentResult.flags.destroy();

  return { scores, flags };
}

export async function burstDensityAnomalies(
  gpu: WebGPUCompute,
  sortedTimestamps: Float32Array,
  options: BurstDensityOptions
): Promise<BurstDensityResult> {
  validateValues(sortedTimestamps);
  validateBurstDensityOptions(options);

  if (!isSortedAscending(sortedTimestamps)) {
    throw new Error("sortedTimestamps must be sorted in ascending order.");
  }

  const timestamps = gpu.createVectorFromData(sortedTimestamps, Float32Array, {
    label: "burst density timestamps"
  });
  const detector = createBurstDensityDetector(gpu, timestamps);
  const residentResult = await detector.run(options);
  const [counts, flags] = await Promise.all([
    residentResult.counts.read(),
    residentResult.flags.read()
  ]);

  timestamps.destroy();
  residentResult.counts.destroy();
  residentResult.flags.destroy();

  return { counts, flags };
}

function validateValues(values: Float32Array): void {
  if (values.length === 0) {
    throw new Error("values must contain at least one item.");
  }
}

function validateRollingZScoreOptions(options: RollingZScoreOptions): void {
  validatePositiveInteger(options.windowSize, "windowSize");
  validatePositiveNumber(options.threshold, "threshold");
  validatePositiveNumber(options.minStdDev ?? 0.000001, "minStdDev");
}

function validateBurstDensityOptions(options: BurstDensityOptions): void {
  validatePositiveNumber(options.windowSize, "windowSize");
  validatePositiveInteger(options.minCount, "minCount");
}

function createRollingZScoreParams(length: number, options: RollingZScoreOptions): ArrayBuffer {
  const params = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(params);
  const paramsF32 = new Float32Array(params);
  paramsU32[0] = length;
  paramsU32[1] = options.windowSize;
  paramsF32[2] = options.threshold;
  paramsF32[3] = options.minStdDev ?? 0.000001;
  return params;
}

function createBurstDensityParams(length: number, options: BurstDensityOptions): ArrayBuffer {
  const params = new ArrayBuffer(16);
  const paramsU32 = new Uint32Array(params);
  const paramsF32 = new Float32Array(params);
  paramsU32[0] = length;
  paramsU32[1] = options.minCount;
  paramsF32[2] = options.windowSize;
  return params;
}

function validatePositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function validatePositiveNumber(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive number.`);
  }
}

const rollingZScoreShader = /* wgsl */ `
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
`;

const burstDensityShader = /* wgsl */ `
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
`;

const rollingZScoreFlagsOnlyShader = /* wgsl */ `
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
`;

const burstDensityFlagsOnlyShader = /* wgsl */ `
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
`;
