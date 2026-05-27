import { type GpuVector, type WebGPUCompute } from "@tiny-webgpu/compute";
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
export declare function packetInterArrivalTimes(timestamps: Float32Array): Float32Array;
export declare function isSortedAscending(values: Float32Array): boolean;
export declare function createRollingZScoreDetector(gpu: WebGPUCompute, values: GpuVector<Float32Array>): GpuRollingZScoreDetector;
export declare function createBurstDensityDetector(gpu: WebGPUCompute, sortedTimestamps: GpuVector<Float32Array>): GpuBurstDensityDetector;
export declare class GpuRollingZScoreDetector {
    private readonly gpu;
    private readonly values;
    private readonly kernel;
    private readonly flagsOnlyKernel;
    constructor(gpu: WebGPUCompute, values: GpuVector<Float32Array>);
    run(options: RollingZScoreOptions): Promise<GpuRollingZScoreResult>;
    runFlags(options: RollingZScoreOptions): Promise<GpuFlagsResult>;
}
export declare class GpuBurstDensityDetector {
    private readonly gpu;
    private readonly sortedTimestamps;
    private readonly kernel;
    private readonly flagsOnlyKernel;
    constructor(gpu: WebGPUCompute, sortedTimestamps: GpuVector<Float32Array>);
    run(options: BurstDensityOptions): Promise<GpuBurstDensityResult>;
    runFlags(options: BurstDensityOptions): Promise<GpuFlagsResult>;
}
export declare function rollingZScoreAnomalies(gpu: WebGPUCompute, values: Float32Array, options: RollingZScoreOptions): Promise<RollingZScoreResult>;
export declare function burstDensityAnomalies(gpu: WebGPUCompute, sortedTimestamps: Float32Array, options: BurstDensityOptions): Promise<BurstDensityResult>;
