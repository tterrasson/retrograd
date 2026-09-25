import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vitepress'

// The path the site is served under. GitHub Pages serves a project site under
// `/retrograd/`: the deploy workflow passes it in `DOCS_BASE_PATH`, without the
// trailing slash. Unset, as with `docs:dev`, the site is served from the root.
const base = `${(process.env.DOCS_BASE_PATH ?? '').replace(/\/+$/, '')}/`

export default defineConfig({
  title: 'Retrograd',
  description: 'Fine-tune GGUF models with LoRA or full weights: SFT, PPO, GRPO, agentic GRPO and distillation',
  base,
  cleanUrls: true,
  srcExclude: ['**/README.md', '**/CLAUDE.md'],
  // Serve docs/assets as the public directory: the logo there is the one the
  // repository README points at, so the site reuses that file rather than
  // keeping a second, re-encoded copy of it.
  vite: {
    publicDir: fileURLToPath(new URL('../assets', import.meta.url)),
  },
  // VitePress prefixes `base` onto the logo, not onto `head`: these are written
  // with it by hand.
  head: [
    ['link', { rel: 'icon', href: `${base}favicon.ico`, sizes: '48x48' }],
    ['link', { rel: 'icon', type: 'image/png', href: `${base}favicon-96x96.png`, sizes: '96x96' }],
    ['link', { rel: 'apple-touch-icon', href: `${base}apple-touch-icon.png`, sizes: '180x180' }],
    ['meta', { name: 'theme-color', content: '#7a2fd4' }],
  ],
  themeConfig: {
    logo: { src: '/logo.png', alt: 'Retrograd' },
    nav: [
      { text: 'Getting started', link: '/getting-started/quickstart' },
      { text: 'Training', link: '/training/sft' },
      { text: 'Reference', link: '/reference/configuration' },
      { text: 'Engineering', link: '/engineering/' },
      // The rustdoc the deploy workflow copies under `api/`, not a VitePress
      // page: `_self` makes the router hand the click to the browser instead
      // of resolving it to its own 404. Absent from `docs:dev`.
      { text: 'API', link: '/api/retrograd/', target: '_self' },
    ],
    sidebar: {
      '/': [
        {
          text: 'Getting started',
          items: [
            { text: 'Quickstart', link: '/getting-started/quickstart' },
            { text: 'Datasets', link: '/getting-started/datasets' },
          ],
        },
        {
          text: 'Training',
          items: [
            { text: 'SFT', link: '/training/sft' },
            { text: 'PPO', link: '/training/ppo' },
            { text: 'GRPO', link: '/training/grpo' },
            { text: 'Agentic GRPO', link: '/training/agent' },
            { text: 'Tools and toolsets', link: '/training/tools' },
            { text: 'Distillation', link: '/training/distill' },
            { text: 'Observing rollouts', link: '/training/observe' },
          ],
        },
        {
          text: 'Operations',
          items: [
            { text: 'Checkpoints and metrics', link: '/operations/checkpoints' },
            { text: 'Performance and memory', link: '/operations/performance' },
          ],
        },
        {
          text: 'Reference',
          items: [
            { text: 'Configuration', link: '/reference/configuration' },
            { text: 'CLI', link: '/reference/cli' },
            { text: 'Build variants', link: '/reference/builds' },
          ],
        },
      ],
      '/engineering/': [
        {
          text: 'Contributing',
          items: [
            { text: 'Overview', link: '/engineering/' },
            { text: 'Contribution principles', link: '/engineering/contributing' },
            { text: 'Numeric conversions', link: '/engineering/CONVERSIONS' },
            { text: 'Tests and validation', link: '/engineering/tests/notice' },
            { text: 'Test lanes in detail', link: '/engineering/tests/lanes' },
            { text: 'llama.cpp fork workflow', link: '/engineering/LLAMA_CPP_FORK_WORKFLOW' },
            { text: 'Server errors', link: '/engineering/server/ERRORS' },
          ],
        },
        {
          text: 'Backends and RIR',
          items: [
            { text: 'Support matrix', link: '/engineering/SUPPORT' },
            { text: 'CUDA status', link: '/engineering/cuda/STATUS' },
            { text: 'RIR kernel families', link: '/engineering/rir/KERNELS' },
            { text: 'RIR kernel promotion', link: '/engineering/rir/PROMOTION' },
          ],
        },
        {
          text: 'Performance notes',
          items: [
            { text: 'GRPO sampling path', link: '/engineering/optims/SAMPLING' },
            { text: 'Optimizer cost and quality', link: '/engineering/optims/OPTIMIZERS' },
          ],
        },
      ],
    },
    search: { provider: 'local' },
    outline: { level: [2, 3] },
    socialLinks: [{ icon: 'github', link: 'https://github.com/tterrasson/retrograd' }],
    footer: {
      message: 'Retrograd documentation',
      copyright: 'Retrograd',
    },
  },
})
