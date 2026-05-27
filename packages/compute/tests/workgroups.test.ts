import { describe, expect, it } from "vitest";
import { workgroupCount, workgroupCounts2D } from "../src";

describe("workgroup helpers", () => {
  it("rounds up 1D dispatch counts", () => {
    expect(workgroupCount(128, 64)).toBe(2);
    expect(workgroupCount(129, 64)).toBe(3);
  });

  it("rounds up 2D dispatch counts", () => {
    expect(workgroupCounts2D(1920, 1080, 16, 16)).toEqual([120, 68]);
  });

  it("rejects invalid values", () => {
    expect(() => workgroupCount(0, 64)).toThrow("itemCount");
    expect(() => workgroupCount(1, 0)).toThrow("workgroupSize");
  });
});
