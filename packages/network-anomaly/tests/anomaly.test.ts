import { describe, expect, it } from "vitest";
import { WebGpuResidentDataset, isSortedAscending, packetInterArrivalTimes } from "../src";

describe("anomaly feature helpers", () => {
  it("calculates packet inter-arrival times", () => {
    expect(Array.from(packetInterArrivalTimes(new Float32Array([1, 1.5, 3.25])))).toEqual([
      0.5, 1.75
    ]);
  });

  it("checks ascending timestamp order", () => {
    expect(isSortedAscending(new Float32Array([1, 1, 2]))).toBe(true);
    expect(isSortedAscending(new Float32Array([1, 0.5, 2]))).toBe(false);
  });

  it("requires at least one feature for resident datasets", () => {
    expect(() => new WebGpuResidentDataset({} as never, {})).toThrow(
      "At least one feature array is required."
    );
  });
});
