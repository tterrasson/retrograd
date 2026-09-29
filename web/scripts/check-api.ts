// Fails when src/api/schema.d.ts is not what openapi.json generates.
import { readFile } from 'node:fs/promises'
import { generate } from './gen-api'

const target = new URL('../src/api/schema.d.ts', import.meta.url)
const before = await readFile(target, 'utf8').catch(() => '')
await generate()
const after = await readFile(target, 'utf8')
if (before !== after) {
  console.error(
    'src/api/schema.d.ts was out of date with openapi.json; it has been regenerated.\n' +
      'Commit the result, or run `bun run gen:api` after refreshing openapi.json with\n' +
      '  cargo run -q -p retrograd-server --bin retrograd-server -- openapi > web/openapi.json',
  )
  process.exit(1)
}
console.log('src/api/schema.d.ts matches openapi.json')
