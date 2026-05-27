import { describe, expect, it } from "vitest";
import { createDenseLayer, flagsToEvents, severityFromScore } from "../src";

describe("anomaly layers", () => {
  it("creates dense layers", () => {
    const flags = new Uint32Array([0, 1, 0]);
    const layer = createDenseLayer("timing", "timing", flags);

    expect(layer).toEqual({
      id: "timing",
      kind: "timing",
      length: 3,
      scores: undefined,
      flags
    });
  });

  it("converts flags to sparse events", () => {
    const events = flagsToEvents(new Uint32Array([0, 1, 1]), {
      kind: "timing",
      feature: "interArrivalMs",
      timestampsMs: new Float32Array([10, 20, 30]),
      scores: new Float32Array([0, 3.5, 8])
    });

    expect(events).toEqual([
      {
        index: 1,
        timestampMs: 20,
        score: 3.5,
        kind: "timing",
        severity: "medium",
        feature: "interArrivalMs"
      },
      {
        index: 2,
        timestampMs: 30,
        score: 8,
        kind: "timing",
        severity: "high",
        feature: "interArrivalMs"
      }
    ]);
  });

  it("maps scores to severity bands", () => {
    expect(severityFromScore(2)).toBe("low");
    expect(severityFromScore(3)).toBe("medium");
    expect(severityFromScore(6)).toBe("high");
  });
});
