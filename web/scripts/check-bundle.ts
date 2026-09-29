// Fails when the gzipped JavaScript and CSS of the build, charts excepted, are
// over budget. Run after `bun run build`.
import { readdir, readFile } from 'node:fs/promises'
import { gzipSync } from 'node:zlib'

const BUDGET = 400 * 1000
const dir = new URL('../dist/assets/', import.meta.url)

const files = (await readdir(dir)).filter(
  (name) => /\.(js|css)$/.test(name) && !name.startsWith('echarts-'),
)
if (!files.length) {
  console.error('dist/assets is empty: run `bun run build` first')
  process.exit(1)
}
let total = 0
const sizes: [string, number][] = []
for (const name of files) {
  const size = gzipSync(await readFile(new URL(name, dir)), { level: 9 }).length
  sizes.push([name, size])
  total += size
}
sizes.sort((a, b) => b[1] - a[1])
for (const [name, size] of sizes.slice(0, 8))
  console.log(`${(size / 1000).toFixed(1).padStart(8)} kB  ${name}`)
console.log(
  `${(total / 1000).toFixed(1).padStart(8)} kB  total gzipped, charts excepted (budget ${BUDGET / 1000} kB)`,
)
if (total > BUDGET) {
  console.error('over budget')
  process.exit(1)
}
