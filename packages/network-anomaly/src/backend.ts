import type { GpuVector, WebGPUCompute } from "@tiny-webgpu/compute";
import {
  burstDensityAnomalies,
  type BurstDensityOptions,
  type BurstDensityResult,
  createBurstDensityDetector,
  createRollingZScoreDetector,
  type GpuBurstDensityDetector,
  type GpuRollingZScoreDetector,
  rollingZScoreAnomalies,
  type RollingZScoreOptions,
  type RollingZScoreResult
} from "./anomaly";
import type { NumericFeatureSet } from "./model";

export interface AnomalyBackend {
  readonly kind: "webgpu" | "wasm" | "js";
  rollingZScore(values: Float32Array, options: RollingZScoreOptions): Promise<RollingZScoreResult>;
  burstDensity(timestampsMs: Float32Array, options: BurstDensityOptions): Promise<BurstDensityResult>;
  createDataset(features: NumericFeatureSet): Promise<ResidentAnomalyDataset>;
}

export interface ResidentAnomalyDataset {
  readonly backend: AnomalyBackend["kind"];
  readonly itemCount: number;
  rollingZScore(options: RollingZScoreOptions): Promise<RollingZScoreResult>;
  rollingZScoreFlags(options: RollingZScoreOptions): Promise<Uint32Array>;
  burstDensity(options: BurstDensityOptions): Promise<BurstDensityResult>;
  burstDensityFlags(options: BurstDensityOptions): Promise<Uint32Array>;
  destroy(): void;
}

export class WebGpuAnomalyBackend implements AnomalyBackend {
  readonly kind = "webgpu" as const;

  constructor(private readonly gpu: WebGPUCompute) {}

  rollingZScore(values: Float32Array, options: RollingZScoreOptions): Promise<RollingZScoreResult> {
    return rollingZScoreAnomalies(this.gpu, values, options);
  }

  burstDensity(
    timestampsMs: Float32Array,
    options: BurstDensityOptions
  ): Promise<BurstDensityResult> {
    return burstDensityAnomalies(this.gpu, timestampsMs, options);
  }

  async createDataset(features: NumericFeatureSet): Promise<WebGpuResidentDataset> {
    return new WebGpuResidentDataset(this.gpu, features);
  }
}

export function createWebGpuAnomalyBackend(gpu: WebGPUCompute): WebGpuAnomalyBackend {
  return new WebGpuAnomalyBackend(gpu);
}

export class WebGpuResidentDataset implements ResidentAnomalyDataset {
  readonly backend = "webgpu" as const;
  readonly itemCount: number;
  private readonly interArrivalMs?: GpuVector<Float32Array>;
  private readonly timestampsMs?: GpuVector<Float32Array>;
  private rollingDetector?: GpuRollingZScoreDetector;
  private burstDetector?: GpuBurstDensityDetector;

  constructor(
    private readonly gpu: WebGPUCompute,
    features: NumericFeatureSet
  ) {
    if (!features.interArrivalMs && !features.timestampsMs) {
      throw new Error("At least one feature array is required.");
    }

    this.interArrivalMs = features.interArrivalMs
      ? gpu.createVectorFromData(features.interArrivalMs, Float32Array, {
          label: "resident inter-arrival times"
        })
      : undefined;
    this.timestampsMs = features.timestampsMs
      ? gpu.createVectorFromData(features.timestampsMs, Float32Array, {
          label: "resident timestamps"
        })
      : undefined;
    this.itemCount = features.interArrivalMs?.length ?? features.timestampsMs?.length ?? 0;
  }

  async rollingZScore(options: RollingZScoreOptions): Promise<RollingZScoreResult> {
    const result = await this.getRollingDetector().run(options);
    const [scores, flags] = await Promise.all([result.scores.read(), result.flags.read()]);

    result.scores.destroy();
    result.flags.destroy();

    return { scores, flags };
  }

  async rollingZScoreFlags(options: RollingZScoreOptions): Promise<Uint32Array> {
    const result = await this.getRollingDetector().runFlags(options);
    const flags = await result.flags.read();

    result.flags.destroy();

    return flags;
  }

  async burstDensity(options: BurstDensityOptions): Promise<BurstDensityResult> {
    const result = await this.getBurstDetector().run(options);
    const [counts, flags] = await Promise.all([result.counts.read(), result.flags.read()]);

    result.counts.destroy();
    result.flags.destroy();

    return { counts, flags };
  }

  async burstDensityFlags(options: BurstDensityOptions): Promise<Uint32Array> {
    const result = await this.getBurstDetector().runFlags(options);
    const flags = await result.flags.read();

    result.flags.destroy();

    return flags;
  }

  destroy(): void {
    this.interArrivalMs?.destroy();
    this.timestampsMs?.destroy();
  }

  private getRollingDetector(): GpuRollingZScoreDetector {
    if (!this.interArrivalMs) {
      throw new Error("interArrivalMs is required for rolling z-score analysis.");
    }

    this.rollingDetector ??= createRollingZScoreDetector(this.gpu, this.interArrivalMs);
    return this.rollingDetector;
  }

  private getBurstDetector(): GpuBurstDensityDetector {
    if (!this.timestampsMs) {
      throw new Error("timestampsMs is required for burst-density analysis.");
    }

    this.burstDetector ??= createBurstDensityDetector(this.gpu, this.timestampsMs);
    return this.burstDetector;
  }
}
