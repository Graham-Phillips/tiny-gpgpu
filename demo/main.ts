import { initWebGPU } from "@tiny-webgpu/compute";
import {
  burstDensityAnomalies,
  createBurstDensityDetector,
  createRollingZScoreDetector,
  rollingZScoreAnomalies
} from "@tiny-webgpu/network-anomaly";
import "./style.css";

type BenchmarkSize = 100_000 | 500_000 | 1_000_000 | 5_000_000;
type BenchmarkMode = "rolling" | "multipass" | "flags-only" | "burst";

interface ScoreResult {
  scores: Float32Array;
  flags: Uint32Array;
}

interface BurstResult {
  counts: Uint32Array;
  flags: Uint32Array;
}

interface BenchmarkSummary {
  mode: BenchmarkMode;
  itemCount: number;
  anomalyCount: number;
  jsMs: number;
  gpuOneShotMs?: number;
  gpuResidentMs: number;
  oneShotSpeedup?: number;
  residentSpeedup: number;
  mismatches: number;
  maxScoreDelta?: number;
  jsFlags: number;
  gpuFlags: number;
  notes: string[];
}

const app = document.querySelector<HTMLDivElement>("#app");

if (!app) {
  throw new Error("Missing app root.");
}

app.innerHTML = `
  <section class="shell">
    <header class="masthead">
      <p class="eyebrow">Tiny WebGPU Bench</p>
      <h1>Network anomaly benchmark</h1>
    </header>

    <section class="controls" aria-label="Benchmark controls">
      <label>
        Test
        <select id="mode">
          <option value="rolling">Rolling z-score</option>
          <option value="multipass">Resident multi-pass</option>
          <option value="flags-only">Resident flags-only</option>
          <option value="burst">Burst density</option>
        </select>
      </label>
      <label>
        Events
        <select id="size">
          <option value="100000">100,000</option>
          <option value="500000" selected>500,000</option>
          <option value="1000000">1,000,000</option>
          <option value="5000000">5,000,000</option>
        </select>
      </label>
      <label>
        Window
        <input id="window-size" type="number" min="8" max="512" step="8" value="64" />
      </label>
      <label>
        Threshold
        <input id="threshold" type="number" min="1" max="12" step="0.25" value="3" />
      </label>
      <button id="run" type="button">Run benchmark</button>
    </section>

    <section class="metrics" aria-label="Benchmark metrics">
      <article>
        <span>JavaScript</span>
        <strong id="js-time">-</strong>
      </article>
      <article>
        <span>WebGPU one-shot</span>
        <strong id="gpu-time">-</strong>
      </article>
      <article>
        <span>Resident</span>
        <strong id="gpu-resident-time">-</strong>
      </article>
      <article>
        <span>One-shot speedup</span>
        <strong id="speedup">-</strong>
      </article>
      <article>
        <span>Resident speedup</span>
        <strong id="resident-speedup">-</strong>
      </article>
      <article>
        <span>Validation</span>
        <strong id="validation">-</strong>
      </article>
    </section>

    <pre id="output">Ready. Pick a benchmark mode and run it in a WebGPU-capable browser.</pre>
  </section>
`;

const modeSelect = requireElement<HTMLSelectElement>("#mode");
const sizeSelect = requireElement<HTMLSelectElement>("#size");
const windowInput = requireElement<HTMLInputElement>("#window-size");
const thresholdInput = requireElement<HTMLInputElement>("#threshold");
const runButton = requireElement<HTMLButtonElement>("#run");
const output = requireElement<HTMLPreElement>("#output");
const jsTime = requireElement<HTMLElement>("#js-time");
const gpuTime = requireElement<HTMLElement>("#gpu-time");
const gpuResidentTime = requireElement<HTMLElement>("#gpu-resident-time");
const oneShotSpeedup = requireElement<HTMLElement>("#speedup");
const residentSpeedup = requireElement<HTMLElement>("#resident-speedup");
const validation = requireElement<HTMLElement>("#validation");

runButton.addEventListener("click", () => {
  runBenchmark().catch((error: unknown) => {
    output.textContent = error instanceof Error ? error.message : String(error);
    runButton.disabled = false;
  });
});

async function runBenchmark(): Promise<void> {
  runButton.disabled = true;
  resetMetrics();

  const mode = modeSelect.value as BenchmarkMode;
  const itemCount = Number(sizeSelect.value) as BenchmarkSize;
  const windowSize = Number(windowInput.value);
  const threshold = Number(thresholdInput.value);

  try {
    const summary = await runSelectedBenchmark(mode, itemCount, windowSize, threshold);
    renderSummary(summary, windowSize, threshold);
  } finally {
    runButton.disabled = false;
  }
}

async function runSelectedBenchmark(
  mode: BenchmarkMode,
  itemCount: number,
  windowSize: number,
  threshold: number
): Promise<BenchmarkSummary> {
  output.textContent = `Generating ${formatInteger(itemCount)} synthetic network events...`;
  await yieldToBrowser();

  const data = generateNetworkData(itemCount);

  switch (mode) {
    case "rolling":
      return runRollingBenchmark(data.interArrivalMs, data.injectedAnomalies, windowSize, threshold);
    case "multipass":
      return runMultiPassBenchmark(data.interArrivalMs, data.injectedAnomalies, windowSize, threshold);
    case "flags-only":
      return runFlagsOnlyBenchmark(data.interArrivalMs, data.injectedAnomalies, windowSize, threshold);
    case "burst":
      return runBurstBenchmark(data.timestampsMs, data.burstEvents, windowSize);
  }
}

async function runRollingBenchmark(
  values: Float32Array,
  anomalyCount: number,
  windowSize: number,
  threshold: number
): Promise<BenchmarkSummary> {
  output.textContent = "Running JavaScript rolling z-score reference...";
  await yieldToBrowser();
  const jsStart = performance.now();
  const jsResult = rollingZScoreCpu(values, windowSize, threshold);
  const jsMs = performance.now() - jsStart;
  jsTime.textContent = formatMs(jsMs);

  output.textContent = "Running WebGPU one-shot rolling z-score...";
  await yieldToBrowser();
  const gpu = await initWebGPU();
  const gpuStart = performance.now();
  const gpuResult = await rollingZScoreAnomalies(gpu, values, { windowSize, threshold });
  const gpuOneShotMs = performance.now() - gpuStart;

  output.textContent = "Running WebGPU resident rolling z-score...";
  await yieldToBrowser();
  const residentValues = gpu.createVectorFromData(values, Float32Array, {
    label: "resident rolling values"
  });
  const detector = createRollingZScoreDetector(gpu, residentValues);
  const residentStart = performance.now();
  const residentResult = await detector.run({ windowSize, threshold });
  const [residentScores, residentFlags] = await Promise.all([
    residentResult.scores.read(),
    residentResult.flags.read()
  ]);
  const gpuResidentMs = performance.now() - residentStart;

  residentResult.scores.destroy();
  residentResult.flags.destroy();
  residentValues.destroy();
  gpu.destroy();

  const oneShotComparison = compareScoreResults(jsResult, gpuResult);
  const residentComparison = compareScoreResults(jsResult, {
    scores: residentScores,
    flags: residentFlags
  });

  return {
    mode: "rolling",
    itemCount: values.length,
    anomalyCount,
    jsMs,
    gpuOneShotMs,
    gpuResidentMs,
    oneShotSpeedup: jsMs / gpuOneShotMs,
    residentSpeedup: jsMs / gpuResidentMs,
    mismatches: oneShotComparison.mismatches + residentComparison.mismatches,
    maxScoreDelta: Math.max(oneShotComparison.maxScoreDelta, residentComparison.maxScoreDelta),
    jsFlags: countFlags(jsResult.flags),
    gpuFlags: countFlags(residentFlags),
    notes: [
      "One-shot includes upload, detector setup, dispatch, synchronization, and score+flag readback.",
      "Resident keeps input data and detector setup on the GPU, then reads back scores and flags."
    ]
  };
}

async function runMultiPassBenchmark(
  values: Float32Array,
  anomalyCount: number,
  windowSize: number,
  threshold: number
): Promise<BenchmarkSummary> {
  const settings = [
    { windowSize, threshold },
    { windowSize: windowSize * 2, threshold },
    { windowSize, threshold: threshold + 0.5 },
    { windowSize: Math.max(8, Math.floor(windowSize / 2)), threshold: threshold + 1 },
    { windowSize: windowSize * 3, threshold: threshold + 0.25 }
  ];

  output.textContent = `Running JavaScript ${settings.length}-pass rolling z-score reference...`;
  await yieldToBrowser();
  const jsStart = performance.now();
  let jsFlags: Uint32Array<ArrayBufferLike> = new Uint32Array(values.length);

  for (const setting of settings) {
    jsFlags = rollingZScoreCpu(values, setting.windowSize, setting.threshold).flags;
  }

  const jsMs = performance.now() - jsStart;
  jsTime.textContent = formatMs(jsMs);

  output.textContent = `Running WebGPU ${settings.length}-pass resident rolling z-score...`;
  await yieldToBrowser();
  const gpu = await initWebGPU();
  const residentValues = gpu.createVectorFromData(values, Float32Array, {
    label: "resident multi-pass values"
  });
  const detector = createRollingZScoreDetector(gpu, residentValues);
  const residentStart = performance.now();
  let gpuFlags: Uint32Array<ArrayBufferLike> = new Uint32Array(values.length);

  for (const setting of settings) {
    const result = await detector.runFlags(setting);
    gpuFlags = await result.flags.read();
    result.flags.destroy();
  }

  const gpuResidentMs = performance.now() - residentStart;
  residentValues.destroy();
  gpu.destroy();

  return {
    mode: "multipass",
    itemCount: values.length,
    anomalyCount,
    jsMs,
    gpuResidentMs,
    residentSpeedup: jsMs / gpuResidentMs,
    mismatches: countMismatchedFlags(jsFlags, gpuFlags),
    jsFlags: countFlags(jsFlags),
    gpuFlags: countFlags(gpuFlags),
    notes: [
      `${settings.length} resident passes with different windows/thresholds.`,
      "Only flags are read back from each pass to model interactive threshold exploration."
    ]
  };
}

async function runFlagsOnlyBenchmark(
  values: Float32Array,
  anomalyCount: number,
  windowSize: number,
  threshold: number
): Promise<BenchmarkSummary> {
  output.textContent = "Running JavaScript rolling z-score flags-only reference...";
  await yieldToBrowser();
  const jsStart = performance.now();
  const jsFlags = rollingZScoreFlagsCpu(values, windowSize, threshold);
  const jsMs = performance.now() - jsStart;
  jsTime.textContent = formatMs(jsMs);

  output.textContent = "Running WebGPU resident flags-only rolling z-score...";
  await yieldToBrowser();
  const gpu = await initWebGPU();
  const residentValues = gpu.createVectorFromData(values, Float32Array, {
    label: "resident flags-only values"
  });
  const detector = createRollingZScoreDetector(gpu, residentValues);
  const residentStart = performance.now();
  const result = await detector.runFlags({ windowSize, threshold });
  const gpuFlags = await result.flags.read();
  const gpuResidentMs = performance.now() - residentStart;

  result.flags.destroy();
  residentValues.destroy();
  gpu.destroy();

  return {
    mode: "flags-only",
    itemCount: values.length,
    anomalyCount,
    jsMs,
    gpuResidentMs,
    residentSpeedup: jsMs / gpuResidentMs,
    mismatches: countMismatchedFlags(jsFlags, gpuFlags),
    jsFlags: countFlags(jsFlags),
    gpuFlags: countFlags(gpuFlags),
    notes: [
      "This avoids generating or reading back z-score arrays.",
      "It is closer to a visualization workflow that only needs markers or masks."
    ]
  };
}

async function runBurstBenchmark(
  timestampsMs: Float32Array,
  burstEvents: number,
  windowSize: number
): Promise<BenchmarkSummary> {
  const minCount = Math.max(8, Math.floor(windowSize / 2));

  output.textContent = "Running JavaScript burst-density reference...";
  await yieldToBrowser();
  const jsStart = performance.now();
  const jsResult = burstDensityCpu(timestampsMs, windowSize, minCount);
  const jsMs = performance.now() - jsStart;
  jsTime.textContent = formatMs(jsMs);

  output.textContent = "Running WebGPU one-shot burst-density...";
  await yieldToBrowser();
  const gpu = await initWebGPU();
  const gpuStart = performance.now();
  const gpuResult = await burstDensityAnomalies(gpu, timestampsMs, { windowSize, minCount });
  const gpuOneShotMs = performance.now() - gpuStart;

  output.textContent = "Running WebGPU resident burst-density...";
  await yieldToBrowser();
  const residentTimestamps = gpu.createVectorFromData(timestampsMs, Float32Array, {
    label: "resident burst timestamps"
  });
  const detector = createBurstDensityDetector(gpu, residentTimestamps);
  const residentStart = performance.now();
  const residentResult = await detector.run({ windowSize, minCount });
  const [residentCounts, residentFlags] = await Promise.all([
    residentResult.counts.read(),
    residentResult.flags.read()
  ]);
  const gpuResidentMs = performance.now() - residentStart;

  residentResult.counts.destroy();
  residentResult.flags.destroy();
  residentTimestamps.destroy();
  gpu.destroy();

  return {
    mode: "burst",
    itemCount: timestampsMs.length,
    anomalyCount: burstEvents,
    jsMs,
    gpuOneShotMs,
    gpuResidentMs,
    oneShotSpeedup: jsMs / gpuOneShotMs,
    residentSpeedup: jsMs / gpuResidentMs,
    mismatches:
      countMismatchedFlags(jsResult.flags, gpuResult.flags) +
      countMismatchedFlags(jsResult.flags, residentFlags) +
      countMismatchedFlags(jsResult.counts, residentCounts),
    jsFlags: countFlags(jsResult.flags),
    gpuFlags: countFlags(residentFlags),
    notes: [
      `Burst minCount is derived as ${minCount} events within the selected window.`,
      "This benchmark compares dense event clusters over sorted timestamps."
    ]
  };
}

function rollingZScoreCpu(
  values: Float32Array,
  windowSize: number,
  threshold: number,
  minStdDev = 0.000001
): ScoreResult {
  const scores = new Float32Array(values.length);
  const flags = new Uint32Array(values.length);

  for (let index = 0; index < values.length; index += 1) {
    const score = rollingZScoreAt(values, index, windowSize, minStdDev);
    scores[index] = score;
    flags[index] = score >= threshold ? 1 : 0;
  }

  return { scores, flags };
}

function rollingZScoreFlagsCpu(
  values: Float32Array,
  windowSize: number,
  threshold: number,
  minStdDev = 0.000001
): Uint32Array {
  const flags = new Uint32Array(values.length);

  for (let index = 0; index < values.length; index += 1) {
    flags[index] = rollingZScoreAt(values, index, windowSize, minStdDev) >= threshold ? 1 : 0;
  }

  return flags;
}

function rollingZScoreAt(
  values: Float32Array,
  index: number,
  windowSize: number,
  minStdDev: number
): number {
  const start = Math.max(0, index - windowSize);
  const count = index - start;

  if (count < 2) {
    return 0;
  }

  let sum = 0;

  for (let cursor = start; cursor < index; cursor += 1) {
    sum += values[cursor];
  }

  const mean = sum / count;
  let varianceSum = 0;

  for (let cursor = start; cursor < index; cursor += 1) {
    const delta = values[cursor] - mean;
    varianceSum += delta * delta;
  }

  const stdDev = Math.max(Math.sqrt(varianceSum / count), minStdDev);
  return Math.abs(values[index] - mean) / stdDev;
}

function burstDensityCpu(timestampsMs: Float32Array, windowSize: number, minCount: number): BurstResult {
  const counts = new Uint32Array(timestampsMs.length);
  const flags = new Uint32Array(timestampsMs.length);

  for (let index = 0; index < timestampsMs.length; index += 1) {
    const current = timestampsMs[index];
    let count = 0;

    for (let cursor = index; cursor >= 0; cursor -= 1) {
      if (current - timestampsMs[cursor] > windowSize) {
        break;
      }

      count += 1;
    }

    counts[index] = count;
    flags[index] = count >= minCount ? 1 : 0;
  }

  return { counts, flags };
}

function generateNetworkData(itemCount: number): {
  timestampsMs: Float32Array;
  interArrivalMs: Float32Array;
  injectedAnomalies: number;
  burstEvents: number;
} {
  const timestampsMs = new Float32Array(itemCount);
  const interArrivalMs = new Float32Array(itemCount);
  let seed = 0x12345678;
  let timestamp = 0;
  let injectedAnomalies = 0;
  let burstEvents = 0;

  for (let index = 0; index < itemCount; index += 1) {
    seed = nextRandom(seed);
    const jitter = ((seed >>> 8) / 0x00ffffff - 0.5) * 0.5;
    const slowWave = Math.sin(index / 4096) * 0.2;
    let delta = 2 + jitter + slowWave;

    if (index > 256 && index % 18_001 === 0) {
      delta = 18 + (index % 7);
      injectedAnomalies += 1;
    }

    if (index > 256 && index % 43_019 === 0) {
      delta = 0.05;
      injectedAnomalies += 1;
    }

    if (index > 256 && index % 61_003 < 36) {
      delta = 0.08;
      burstEvents += 1;
    }

    timestamp += delta;
    timestampsMs[index] = timestamp;
    interArrivalMs[index] = delta;
  }

  return { timestampsMs, interArrivalMs, injectedAnomalies, burstEvents };
}

function compareScoreResults(
  jsResult: ScoreResult,
  gpuResult: ScoreResult
): { mismatches: number; maxScoreDelta: number } {
  let mismatches = 0;
  let maxScoreDelta = 0;

  for (let index = 0; index < jsResult.flags.length; index += 1) {
    if (jsResult.flags[index] !== gpuResult.flags[index]) {
      mismatches += 1;
    }

    const delta = Math.abs(jsResult.scores[index] - gpuResult.scores[index]);

    if (delta > maxScoreDelta) {
      maxScoreDelta = delta;
    }
  }

  return { mismatches, maxScoreDelta };
}

function countMismatchedFlags(a: Uint32Array, b: Uint32Array): number {
  let mismatches = 0;

  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) {
      mismatches += 1;
    }
  }

  return mismatches;
}

function renderSummary(result: BenchmarkSummary, windowSize: number, threshold: number): void {
  jsTime.textContent = formatMs(result.jsMs);
  gpuTime.textContent = result.gpuOneShotMs === undefined ? "-" : formatMs(result.gpuOneShotMs);
  gpuResidentTime.textContent = formatMs(result.gpuResidentMs);
  oneShotSpeedup.textContent =
    result.oneShotSpeedup === undefined ? "-" : `${result.oneShotSpeedup.toFixed(2)}x`;
  residentSpeedup.textContent = `${result.residentSpeedup.toFixed(2)}x`;
  validation.textContent = result.mismatches === 0 ? "match" : `${result.mismatches} diffs`;

  output.textContent = [
    `Mode:                ${modeLabel(result.mode)}`,
    `Events:              ${formatInteger(result.itemCount)}`,
    `Injected anomalies:  ${formatInteger(result.anomalyCount)}`,
    `Window size:         ${windowSize}`,
    `Threshold:           ${threshold}`,
    "",
    `JavaScript time:     ${formatMs(result.jsMs)}`,
    `WebGPU one-shot:     ${result.gpuOneShotMs === undefined ? "-" : formatMs(result.gpuOneShotMs)}`,
    `Resident time:       ${formatMs(result.gpuResidentMs)}`,
    `One-shot speedup:    ${result.oneShotSpeedup === undefined ? "-" : `${result.oneShotSpeedup.toFixed(2)}x`}`,
    `Resident speedup:    ${result.residentSpeedup.toFixed(2)}x`,
    "",
    `JavaScript flags:    ${formatInteger(result.jsFlags)}`,
    `WebGPU flags:        ${formatInteger(result.gpuFlags)}`,
    `Validation diffs:    ${formatInteger(result.mismatches)}`,
    result.maxScoreDelta === undefined ? "" : `Max score delta:     ${result.maxScoreDelta.toExponential(3)}`,
    "",
    ...result.notes
  ]
    .filter((line) => line !== "")
    .join("\n");
}

function modeLabel(mode: BenchmarkMode): string {
  switch (mode) {
    case "rolling":
      return "Rolling z-score";
    case "multipass":
      return "Resident multi-pass";
    case "flags-only":
      return "Resident flags-only";
    case "burst":
      return "Burst density";
  }
}

function countFlags(flags: Uint32Array): number {
  let count = 0;

  for (const flag of flags) {
    count += flag;
  }

  return count;
}

function resetMetrics(): void {
  jsTime.textContent = "-";
  gpuTime.textContent = "-";
  gpuResidentTime.textContent = "-";
  oneShotSpeedup.textContent = "-";
  residentSpeedup.textContent = "-";
  validation.textContent = "-";
}

function requireElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);

  if (!element) {
    throw new Error(`Missing element: ${selector}`);
  }

  return element;
}

function nextRandom(seed: number): number {
  return (1664525 * seed + 1013904223) >>> 0;
}

function formatMs(value: number): string {
  return `${value.toFixed(1)} ms`;
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, 0);
  });
}
