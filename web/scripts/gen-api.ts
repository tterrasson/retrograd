// Regenerates the typed client schema from the versioned OpenAPI snapshot.
import { $ } from 'bun'

const root = new URL('..', import.meta.url).pathname

export async function generate(): Promise<void> {
  // A field with a default is still one the server may leave out: every
  // `Option` it skips when absent carries `default: null` in the document.
  await $`bunx openapi-typescript openapi.json --default-non-nullable false -o src/api/schema.d.ts`
    .cwd(root)
    .quiet()
  await $`bunx prettier --write src/api/schema.d.ts`.cwd(root).quiet()
}

if (import.meta.main) {
  await generate()
  console.log('src/api/schema.d.ts regenerated from openapi.json')
}
