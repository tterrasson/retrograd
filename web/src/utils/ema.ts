/**
 * Exponential moving average with bias correction (as TensorBoard draws it),
 * skipping gaps. `weight` 0 returns the input.
 */
export function ema(values: readonly (number | null)[], weight: number): (number | null)[] {
  if (weight <= 0) return [...values]
  const out: (number | null)[] = []
  let last = 0
  let steps = 0
  for (const value of values) {
    if (value === null || !Number.isFinite(value)) {
      out.push(null)
      continue
    }
    last = last * weight + (1 - weight) * value
    steps++
    out.push(last / (1 - weight ** steps))
  }
  return out
}
