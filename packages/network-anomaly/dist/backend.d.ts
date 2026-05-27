import type { WebGPUCompute } from "@tiny-webgpu/compute";
import { type BurstDensityOptions, type BurstDensityResult, type RollingZScoreOptions, type RollingZScoreResult } from "./anomaly";
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
export declare class WebGpuAnomalyBackend implements AnomalyBackend {
    private readonly gpu;
    readonly kind: "webgpu";
    constructor(gpu: WebGPUCompute);
    rollingZScore(values: Float32Array, options: RollingZScoreOptions): Promise<RollingZScoreResult>;
    burstDensity(timestampsMs: Float32Array, options: BurstDensityOptions): Promise<BurstDensityResult>;
    createDataset(features: NumericFeatureSet): Promise<WebGpuResidentDataset>;
}
export declare function createWebGpuAnomalyBackend(gpu: WebGPUCompute): WebGpuAnomalyBackend;
export declare class WebGpuResidentDataset implements ResidentAnomalyDataset {
    private readonly gpu;
    readonly backend: "webgpu";
    readonly itemCount: number;
    private readonly interArrivalMs?;
    private readonly timestampsMs?;
    private rollingDetector?;
    private burstDetector?;
    constructor(gpu: WebGPUCompute, features: NumericFeatureSet);
    rollingZScore(options: RollingZScoreOptions): Promise<RollingZScoreResult>;
    rollingZScoreFlags(options: RollingZScoreOptions): Promise<Uint32Array>;
    burstDensity(options: BurstDensityOptions): Promise<BurstDensityResult>;
    burstDensityFlags(options: BurstDensityOptions): Promise<Uint32Array>;
    destroy(): void;
    private getRollingDetector;
    private getBurstDetector;
}
