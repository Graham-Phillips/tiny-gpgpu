export type AnomalyKind = "timing" | "burst" | "density" | "protocol" | "sequence" | "shape" | "error-rate";
export type Severity = "low" | "medium" | "high";
export interface NumericFeatureSet {
    timestampsMs?: Float32Array;
    interArrivalMs?: Float32Array;
    messageSizes?: Float32Array;
    errorCounts?: Uint32Array;
}
export interface AnomalyEvent {
    index: number;
    timestampMs?: number;
    score: number;
    kind: AnomalyKind;
    severity: Severity;
    feature?: string;
    explanation?: string;
}
export interface DenseAnomalyLayer {
    id: string;
    kind: AnomalyKind;
    length: number;
    scores?: Float32Array;
    flags: Uint32Array;
}
export interface BucketedAnomalyLayer {
    id: string;
    kind: AnomalyKind;
    bucketSizeMs: number;
    startTimeMs: number;
    counts: Uint32Array;
    maxScores?: Float32Array;
}
export interface SparseAnomalyLayer {
    id: string;
    kind: AnomalyKind;
    events: AnomalyEvent[];
}
export type AnomalyLayer = DenseAnomalyLayer | BucketedAnomalyLayer | SparseAnomalyLayer;
export interface AnomalyRunMetadata {
    backend: "webgpu" | "wasm" | "js";
    mode: "one-shot" | "resident";
    itemCount: number;
    elapsedMs?: number;
}
export interface AnomalyRunResult {
    layer: AnomalyLayer;
    metadata: AnomalyRunMetadata;
}
