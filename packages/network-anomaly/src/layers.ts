import type { AnomalyEvent, AnomalyKind, DenseAnomalyLayer, Severity, SparseAnomalyLayer } from "./model";

export interface FlagsToEventsOptions {
  kind: AnomalyKind;
  feature?: string;
  timestampsMs?: Float32Array;
  scores?: Float32Array;
  lowThreshold?: number;
  highThreshold?: number;
}

export function createDenseLayer(
  id: string,
  kind: AnomalyKind,
  flags: Uint32Array,
  scores?: Float32Array
): DenseAnomalyLayer {
  return {
    id,
    kind,
    length: flags.length,
    scores,
    flags
  };
}

export function flagsToEvents(flags: Uint32Array, options: FlagsToEventsOptions): AnomalyEvent[] {
  const events: AnomalyEvent[] = [];

  for (let index = 0; index < flags.length; index += 1) {
    if (flags[index] === 0) {
      continue;
    }

    const score = options.scores?.[index] ?? flags[index];
    events.push({
      index,
      timestampMs: options.timestampsMs?.[index],
      score,
      kind: options.kind,
      severity: severityFromScore(score, options.lowThreshold, options.highThreshold),
      feature: options.feature
    });
  }

  return events;
}

export function createSparseLayer(
  id: string,
  kind: AnomalyKind,
  events: AnomalyEvent[]
): SparseAnomalyLayer {
  return {
    id,
    kind,
    events
  };
}

export function severityFromScore(
  score: number,
  lowThreshold = 3,
  highThreshold = 6
): Severity {
  if (score >= highThreshold) {
    return "high";
  }

  if (score >= lowThreshold) {
    return "medium";
  }

  return "low";
}
