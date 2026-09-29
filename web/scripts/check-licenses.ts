// Fails when a production dependency, direct or transitive, carries a license
// outside the allowed list. An `OR` expression passes when one side does.
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

const ALLOWED = new Set([
  'MIT',
  'Apache-2.0',
  'BSD-2-Clause',
  'BSD-3-Clause',
  'ISC',
  '0BSD',
  'BlueOak-1.0.0',
  'CC0-1.0',
  'Unlicense',
  'MPL-2.0',
])

/**
 * Packages a production dependency declares but whose code never reaches the
 * bundle, each with the reason. Checked against `dist/` below: an exception
 * that turns up in the build fails like any other license.
 */
const NOT_SHIPPED: Record<string, { reason: string; signature: string }> = {
  argparse: {
    reason: "markdown-it's command-line entry point only; the library never imports it",
    signature: 'ArgumentParser',
  },
}

const root = new URL('..', import.meta.url).pathname

interface Manifest {
  name: string
  version: string
  license?: string | { type?: string }
  licenses?: { type?: string }[]
  dependencies?: Record<string, string>
  optionalDependencies?: Record<string, string>
}

async function manifestOf(
  name: string,
  from: string,
): Promise<{ manifest: Manifest; dir: string } | null> {
  // Node's resolution: the nearest node_modules up from the dependent.
  let dir = from
  for (;;) {
    const candidate = join(dir, 'node_modules', name)
    try {
      const manifest = JSON.parse(
        await readFile(join(candidate, 'package.json'), 'utf8'),
      ) as Manifest
      return { manifest, dir: candidate }
    } catch {
      const parent = join(dir, '..')
      if (parent === dir || !parent.startsWith(root.replace(/\/$/, ''))) return null
      dir = parent
    }
  }
}

function licenseOf(manifest: Manifest): string {
  if (typeof manifest.license === 'string') return manifest.license
  if (manifest.license?.type) return manifest.license.type
  return (
    manifest.licenses
      ?.map((item) => item.type)
      .filter(Boolean)
      .join(' OR ') ?? ''
  )
}

function allowed(expression: string): boolean {
  const cleaned = expression.replace(/[()]/g, ' ').trim()
  if (!cleaned) return false
  return cleaned
    .split(/\s+OR\s+/i)
    .some((alternative) =>
      alternative
        .split(/\s+AND\s+/i)
        .every((id) => ALLOWED.has(id.trim()) || ALLOWED.has(id.trim().replace(/\+$/, ''))),
    )
}

const top = JSON.parse(await readFile(join(root, 'package.json'), 'utf8')) as Manifest
const seen = new Set<string>()
const failures: string[] = []
const exempted: string[] = []
const missing: string[] = []
const queue: [string, string][] = Object.keys(top.dependencies ?? {}).map((name) => [name, root])

while (queue.length) {
  const [name, from] = queue.shift() as [string, string]
  const found = await manifestOf(name, from)
  if (!found) {
    missing.push(name)
    continue
  }
  const key = `${found.manifest.name}@${found.manifest.version}`
  if (seen.has(key)) continue
  seen.add(key)
  const license = licenseOf(found.manifest)
  if (!allowed(license)) {
    if (found.manifest.name in NOT_SHIPPED) exempted.push(found.manifest.name)
    else failures.push(`${key}: ${license || 'no license field'}`)
  }
  for (const child of Object.keys(found.manifest.dependencies ?? {})) queue.push([child, found.dir])
  for (const child of Object.keys(found.manifest.optionalDependencies ?? {})) {
    if (await manifestOf(child, found.dir)) queue.push([child, found.dir])
  }
}

console.log(`${seen.size} production packages checked`)
for (const name of exempted) {
  const exemption = NOT_SHIPPED[name]
  if (!exemption) continue
  console.log(`exempted ${name}: ${exemption.reason}`)
  const shipped = await Array.fromAsync(new Bun.Glob('dist/assets/*.js').scan({ cwd: root }))
  for (const file of shipped) {
    if ((await readFile(join(root, file), 'utf8')).includes(exemption.signature)) {
      failures.push(`${name} is exempted as not shipped, but ${file} contains it`)
    }
  }
}
if (missing.length)
  console.warn(`not installed (optional or platform-specific): ${[...new Set(missing)].join(', ')}`)
if (failures.length) {
  console.error(`licenses outside the allowed list:\n  ${failures.join('\n  ')}`)
  process.exit(1)
}
