tinygpgpuclass c extends Error {
  constructor(e = "WebGPU is not available in this browser context.") {
    super(e), this.name = "WebGPUUnavailableError";
  }
}
async function l(i = {}) {
  if (!navigator.gpu)
    throw new c(
      "WebGPU is not available. Use a secure context and a browser with WebGPU support."
    );
  const e = await navigator.gpu.requestAdapter({
    powerPreference: i.powerPreference
  });
  if (!e)
    throw new c("No compatible WebGPU adapter was found.");
  const r = await e.requestDevice({
    requiredFeatures: i.requiredFeatures,
    requiredLimits: i.requiredLimits
  });
  return new d(e, r);
}
class d {
  constructor(e, r) {
    this.adapter = e, this.device = r;
  }
  createBuffer(e, r = {}) {
    if (!Number.isInteger(e) || e <= 0)
      throw new Error("byteLength must be a positive integer.");
    return this.device.createBuffer({
      label: r.label,
      size: f(e, 4),
      usage: r.usage ?? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
  }
  createBufferFromData(e, r = {}) {
    const t = this.createBuffer(e.byteLength, r);
    return this.device.queue.writeBuffer(t, 0, e), t;
  }
  createVector(e, r, t = {}) {
    if (!Number.isInteger(e) || e <= 0)
      throw new Error("length must be a positive integer.");
    const a = e * r.BYTES_PER_ELEMENT, s = this.createBuffer(a, t);
    return new o(this, s, e, a, r);
  }
  createVectorFromData(e, r, t = {}) {
    const a = this.createBufferFromData(e, t);
    return new o(this, a, e.length, e.byteLength, r);
  }
  createReadbackBuffer(e, r) {
    return this.createBuffer(e, {
      label: r,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
  }
  createKernel(e) {
    const r = this.device.createShaderModule({
      label: e.label ? `${e.label} shader` : void 0,
      code: e.code
    }), t = this.device.createComputePipeline({
      label: e.label,
      layout: e.bindGroupLayout ? this.device.createPipelineLayout({ bindGroupLayouts: [e.bindGroupLayout] }) : "auto",
      compute: {
        module: r,
        entryPoint: e.entryPoint ?? "main",
        constants: e.constants
      }
    });
    return new b(this.device, t);
  }
  async readBuffer(e, r, t) {
    const a = this.createReadbackBuffer(r, "readback"), s = this.device.createCommandEncoder();
    s.copyBufferToBuffer(e, 0, a, 0, f(r, 4)), this.device.queue.submit([s.finish()]), await a.mapAsync(GPUMapMode.READ);
    const n = a.getMappedRange().slice(0, r);
    return a.unmap(), a.destroy(), new t(n);
  }
  destroy() {
    this.device.destroy();
  }
}
class o {
  constructor(e, r, t, a, s) {
    this.gpu = e, this.buffer = r, this.length = t, this.byteLength = a, this.ArrayType = s;
  }
  binding() {
    return { buffer: this.buffer };
  }
  read() {
    return this.gpu.readBuffer(this.buffer, this.byteLength, this.ArrayType);
  }
  destroy() {
    this.buffer.destroy();
  }
}
class b {
  constructor(e, r) {
    this.device = e, this.pipeline = r;
  }
  run(e) {
    const r = this.device.createCommandEncoder({
      label: e.label ? `${e.label} encoder` : void 0
    }), t = r.beginComputePass({
      label: e.label
    }), a = this.device.createBindGroup({
      label: e.label ? `${e.label} bind group` : void 0,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: e.bindings.map((s, n) => ({ binding: n, resource: s }))
    });
    t.setPipeline(this.pipeline), t.setBindGroup(0, a), t.dispatchWorkgroups(
      e.workgroups[0],
      e.workgroups[1] ?? 1,
      e.workgroups[2] ?? 1
    ), t.end(), this.device.queue.submit([r.finish()]);
  }
}
function f(i, e) {
  return Math.ceil(i / e) * e;
}
function h(i, e) {
  return u(i, "itemCount"), u(e, "workgroupSize"), Math.ceil(i / e);
}
function p(i, e, r, t) {
  return u(i, "width"), u(e, "height"), u(r, "workgroupWidth"), u(t, "workgroupHeight"), [Math.ceil(i / r), Math.ceil(e / t)];
}
function u(i, e) {
  if (!Number.isInteger(i) || i <= 0)
    throw new Error(`${e} must be a positive integer.`);
}
export {
  o as GpuVector,
  d as WebGPUCompute,
  b as WebGPUKernel,
  c as WebGPUUnavailableError,
  l as initWebGPU,
  h as workgroupCount,
  p as workgroupCounts2D
};
