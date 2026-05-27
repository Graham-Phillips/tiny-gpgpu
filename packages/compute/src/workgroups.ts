export function workgroupCount(itemCount: number, workgroupSize: number): number {
  assertPositiveInteger(itemCount, "itemCount");
  assertPositiveInteger(workgroupSize, "workgroupSize");
  return Math.ceil(itemCount / workgroupSize);
}

export function workgroupCounts2D(
  width: number,
  height: number,
  workgroupWidth: number,
  workgroupHeight: number
): [number, number] {
  assertPositiveInteger(width, "width");
  assertPositiveInteger(height, "height");
  assertPositiveInteger(workgroupWidth, "workgroupWidth");
  assertPositiveInteger(workgroupHeight, "workgroupHeight");
  return [Math.ceil(width / workgroupWidth), Math.ceil(height / workgroupHeight)];
}

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}
