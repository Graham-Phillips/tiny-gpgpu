import type { AnomalyEvent, AnomalyKind, DenseAnomalyLayer, Severity, SparseAnomalyLayer } from "./model";
export interface FlagsToEventsOptions {
    kind: AnomalyKind;
    feature?: string;
    timestampsMs?: Float32Array;
    scores?: Float32Array;
    lowThreshold?: number;
    highThreshold?: number;
}
export declare function createDenseLayer(id: string, kind: AnomalyKind, flags: Uint32Array, scores?: Float32Array): DenseAnomalyLayer;
export declare function flagsToEvents(flags: Uint32Array, options: FlagsToEventsOptions): AnomalyEvent[];
export declare function createSparseLayer(id: string, kind: AnomalyKind, events: AnomalyEvent[]): SparseAnomalyLayer;
export declare function severityFromScore(score: number, lowThreshold?: number, highThreshold?: number): Severity;
