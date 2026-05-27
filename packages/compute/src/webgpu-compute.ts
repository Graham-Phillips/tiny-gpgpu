export type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint8ClampedArray
  | Uint16Array
  | Uint32Array;

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

export class WebGPUUnavailableError extends Error {
  constructor(message = "WebGPU is not available in this browser context.") {
    super(message);
    this.name = "WebGPUUnavailableError";
  }
}

export async function initWebGPU(options: WebGPUComputeOptions = {}): Promise<WebGPUCompute> {
  if (!navigator.gpu) {
    throw new WebGPUUnavailableError(
      "WebGPU is not available. Use a secure context and a browser with WebGPU support."
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference
  });

  if (!adapter) {
    throw new WebGPUUnavailableError("No compatible WebGPU adapter was found.");
  }

  const device = await adapter.requestDevice({
    requiredFeatures: options.requiredFeatures,
    requiredLimits: options.requiredLimits
  });

  return new WebGPUCompute(adapter, device);
}

export class WebGPUCompute {
  constructor(
    public readonly adapter: GPUAdapter,
    public readonly device: GPUDevice
  ) {}

  createBuffer(byteLength: number, options: BufferOptions = {}): GPUBuffer {
    if (!Number.isInteger(byteLength) || byteLength <= 0) {
      throw new Error("byteLength must be a positive integer.");
    }

    return this.device.createBuffer({
      label: options.label,
      size: alignTo(byteLength, 4),
      usage:
        options.usage ??
        (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)
    });
  }

  createBufferFromData(data: BufferData, options: BufferOptions = {}): GPUBuffer {
    const buffer = this.createBuffer(data.byteLength, options);
    this.device.queue.writeBuffer(buffer, 0, data as BufferSource);
    return buffer;
  }

  createVector<T extends TypedArray>(
    length: number,
    ArrayType: TypedArrayConstructor<T>,
    options: GpuVectorOptions = {}
  ): GpuVector<T> {
    if (!Number.isInteger(length) || length <= 0) {
      throw new Error("length must be a positive integer.");
    }

    const byteLength = length * ArrayType.BYTES_PER_ELEMENT;
    const buffer = this.createBuffer(byteLength, options);
    return new GpuVector(this, buffer, length, byteLength, ArrayType);
  }

  createVectorFromData<T extends TypedArray>(
    data: T,
    ArrayType: TypedArrayConstructor<T>,
    options: GpuVectorOptions = {}
  ): GpuVector<T> {
    const buffer = this.createBufferFromData(data, options);
    return new GpuVector(this, buffer, data.length, data.byteLength, ArrayType);
  }

  createReadbackBuffer(byteLength: number, label?: string): GPUBuffer {
    return this.createBuffer(byteLength, {
      label,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
  }

  createKernel(options: KernelOptions): WebGPUKernel {
    const module = this.device.createShaderModule({
      label: options.label ? `${options.label} shader` : undefined,
      code: options.code
    });

    const pipeline = this.device.createComputePipeline({
      label: options.label,
      layout: options.bindGroupLayout
        ? this.device.createPipelineLayout({ bindGroupLayouts: [options.bindGroupLayout] })
        : "auto",
      compute: {
        module,
        entryPoint: options.entryPoint ?? "main",
        constants: options.constants
      }
    });

    return new WebGPUKernel(this.device, pipeline);
  }

  async readBuffer<T extends TypedArray>(
    source: GPUBuffer,
    byteLength: number,
    ArrayType: TypedArrayConstructor<T>
  ): Promise<T> {
    const readback = this.createReadbackBuffer(byteLength, "readback");
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(source, 0, readback, 0, alignTo(byteLength, 4));
    this.device.queue.submit([commandEncoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0, byteLength);
    readback.unmap();
    readback.destroy();

    return new ArrayType(copy);
  }

  destroy(): void {
    this.device.destroy();
  }
}

export class GpuVector<T extends TypedArray> {
  constructor(
    private readonly gpu: WebGPUCompute,
    public readonly buffer: GPUBuffer,
    public readonly length: number,
    public readonly byteLength: number,
    public readonly ArrayType: TypedArrayConstructor<T>
  ) {}

  binding(): GPUBindingResource {
    return { buffer: this.buffer };
  }

  read(): Promise<T> {
    return this.gpu.readBuffer(this.buffer, this.byteLength, this.ArrayType);
  }

  destroy(): void {
    this.buffer.destroy();
  }
}

export class WebGPUKernel {
  constructor(
    private readonly device: GPUDevice,
    public readonly pipeline: GPUComputePipeline
  ) {}

  run(options: RunKernelOptions): void {
    const commandEncoder = this.device.createCommandEncoder({
      label: options.label ? `${options.label} encoder` : undefined
    });
    const pass = commandEncoder.beginComputePass({
      label: options.label
    });

    const bindGroup = this.device.createBindGroup({
      label: options.label ? `${options.label} bind group` : undefined,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: options.bindings.map((resource, binding) => ({ binding, resource }))
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      options.workgroups[0],
      options.workgroups[1] ?? 1,
      options.workgroups[2] ?? 1
    );
    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }
}

function alignTo(value: number, multiple: number): number {
  return Math.ceil(value / multiple) * multiple;
}
