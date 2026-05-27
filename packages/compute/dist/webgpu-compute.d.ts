export type TypedArray = Float32Array | Float64Array | Int8Array | Int16Array | Int32Array | Uint8Array | Uint8ClampedArray | Uint16Array | Uint32Array;
export type TypedArrayConstructor<T extends TypedArray> = {
    readonly BYTES_PER_ELEMENT: number;
    new (length: number): T;
    new (buffer: ArrayBuffer): T;
};
export type BufferData = BufferSource | ArrayBufferView<ArrayBufferLike>;
export interface WebGPUComputeOptions {
    powerPreference?: GPUPowerPreference;
    requiredFeatures?: GPUFeatureName[];
    requiredLimits?: Record<string, number>;
}
export interface KernelOptions {
    code: string;
    entryPoint?: string;
    label?: string;
    bindGroupLayout?: GPUBindGroupLayout;
    constants?: Record<string, GPUPipelineConstantValue>;
}
export interface RunKernelOptions {
    bindings: GPUBindingResource[];
    workgroups: [number, number?, number?];
    label?: string;
}
export interface BufferOptions {
    label?: string;
    usage?: GPUBufferUsageFlags;
}
export interface GpuVectorOptions {
    label?: string;
    usage?: GPUBufferUsageFlags;
}
export declare class WebGPUUnavailableError extends Error {
    constructor(message?: string);
}
export declare function initWebGPU(options?: WebGPUComputeOptions): Promise<WebGPUCompute>;
export declare class WebGPUCompute {
    readonly adapter: GPUAdapter;
    readonly device: GPUDevice;
    constructor(adapter: GPUAdapter, device: GPUDevice);
    createBuffer(byteLength: number, options?: BufferOptions): GPUBuffer;
    createBufferFromData(data: BufferData, options?: BufferOptions): GPUBuffer;
    createVector<T extends TypedArray>(length: number, ArrayType: TypedArrayConstructor<T>, options?: GpuVectorOptions): GpuVector<T>;
    createVectorFromData<T extends TypedArray>(data: T, ArrayType: TypedArrayConstructor<T>, options?: GpuVectorOptions): GpuVector<T>;
    createReadbackBuffer(byteLength: number, label?: string): GPUBuffer;
    createKernel(options: KernelOptions): WebGPUKernel;
    readBuffer<T extends TypedArray>(source: GPUBuffer, byteLength: number, ArrayType: TypedArrayConstructor<T>): Promise<T>;
    destroy(): void;
}
export declare class GpuVector<T extends TypedArray> {
    private readonly gpu;
    readonly buffer: GPUBuffer;
    readonly length: number;
    readonly byteLength: number;
    readonly ArrayType: TypedArrayConstructor<T>;
    constructor(gpu: WebGPUCompute, buffer: GPUBuffer, length: number, byteLength: number, ArrayType: TypedArrayConstructor<T>);
    binding(): GPUBindingResource;
    read(): Promise<T>;
    destroy(): void;
}
export declare class WebGPUKernel {
    private readonly device;
    readonly pipeline: GPUComputePipeline;
    constructor(device: GPUDevice, pipeline: GPUComputePipeline);
    run(options: RunKernelOptions): void;
}
