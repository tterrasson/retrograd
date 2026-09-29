import { readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import pluginVue from 'eslint-plugin-vue'
import importPlugin from 'eslint-plugin-import'
import prettier from 'eslint-config-prettier'

const root = fileURLToPath(new URL('.', import.meta.url))
const features = readdirSync(new URL('./src/features', import.meta.url), { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)

// The dependency rule: pages -> features -> api / composables / utils. A
// feature reaches another only through its index, and only `src/api/` sees
// the generated schema and the HTTP client.
const zones = [
  {
    target: './src/api',
    from: [
      './src/features',
      './src/pages',
      './src/layouts',
      './src/stores',
      './src/composables',
      './src/router',
    ],
    message: 'src/api depends on nothing but src/utils.',
  },
  {
    target: './src/utils',
    from: [
      './src/api',
      './src/features',
      './src/pages',
      './src/layouts',
      './src/stores',
      './src/composables',
      './src/router',
    ],
    message: 'src/utils is a leaf.',
  },
  {
    target: './src/composables',
    from: ['./src/features', './src/pages', './src/layouts', './src/router'],
    message: 'composables sit under the features.',
  },
  {
    target: './src/features',
    from: ['./src/pages', './src/layouts', './src/router'],
    message: 'features do not know the pages.',
  },
  {
    target: './src/!(api)/**/*',
    from: ['./src/api/client.ts', './src/api/schema.d.ts'],
    message: 'only src/api talks HTTP and reads the generated schema.',
  },
  ...features.map((feature) => ({
    target: `./src/features/${feature}`,
    from: './src/features',
    except: [
      `./${feature}`,
      ...features.filter((other) => other !== feature).map((other) => `./${other}/index.ts`),
    ],
    message: 'import another feature through its index.ts.',
  })),
]

export default tseslint.config(
  {
    ignores: [
      'dist/**',
      'node_modules/**',
      'src/api/schema.d.ts',
      'test-results/**',
      'playwright-report/**',
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...pluginVue.configs['flat/recommended'],
  {
    files: ['**/*.vue'],
    languageOptions: { parserOptions: { parser: tseslint.parser, extraFileExtensions: ['.vue'] } },
  },
  {
    files: ['src/**/*.{ts,vue}', 'tests/**/*.ts'],
    plugins: { import: importPlugin },
    settings: {
      'import/resolver': {
        typescript: { project: `${root}tsconfig.json` },
        node: true,
      },
    },
    rules: {
      'import/no-restricted-paths': ['error', { basePath: root, zones }],
    },
  },
  {
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_', destructuredArrayIgnorePattern: '^_' },
      ],
      'vue/multi-word-component-names': 'off',
      'vue/require-default-prop': 'off',
      'vue/no-v-html': 'error',
      // TypeScript checks names; this rule does not know the DOM's.
      'no-undef': 'off',
    },
  },
  {
    files: ['scripts/**/*.ts', 'tests/e2e/**/*.ts', '*.config.ts'],
    languageOptions: { globals: { Bun: 'readonly', process: 'readonly', console: 'readonly' } },
  },
  prettier,
)
