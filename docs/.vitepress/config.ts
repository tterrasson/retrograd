import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vitepress'

// The path the site is served under. GitHub Pages serves a project site under
// `/retrograd/`: the deploy workflow passes it in `DOCS_BASE_PATH`, without the
// trailing slash. Unset, as with `docs:dev`, the site is served from the root.
const base = `${(process.env.DOCS_BASE_PATH ?? '').replace(/\/+$/, '')}/`

export default defineConfig({
  title: 'Retrograd',
  description: 'User and engineering documentation for Retrograd LoRA training',
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
      { text: 'Operations', link: '/operations/checkpoints' },
      { text: 'Reference', link: '/reference/configuration' },
      { text: 'Engineering', link: '/engineering/' },
      // The rustdoc the deploy workflow copies under `api/`, not a VitePress
      // page: `_self` makes the router hand the click to the browser instead
      // of resolving it to its own 404. Absent from `docs:dev`.
      { text: 'API', link: '/api/retrograd/', target: '_self' },
    ],
    sidebar: {
      '/getting-started/': [
        {
          text: 'Getting started',
          items: [
            {
              text: 'Quickstart',
              link: '/getting-started/quickstart',
              items: [
                { text: 'Build the CLI', link: '/getting-started/quickstart#_1-build-the-cli' },
                { text: 'Create a dataset', link: '/getting-started/quickstart#_2-create-a-dataset' },
                { text: 'Write a configuration', link: '/getting-started/quickstart#_3-write-a-configuration' },
                { text: 'Run training', link: '/getting-started/quickstart#_4-run-training' },
                { text: 'Test the adapter', link: '/getting-started/quickstart#_5-test-the-adapter' },
                { text: 'Choosing an algorithm', link: '/getting-started/quickstart#choosing-an-algorithm' },
              ],
            },
            {
              text: 'Datasets and paths',
              link: '/getting-started/datasets',
              items: [
                { text: 'Plain text', link: '/getting-started/datasets#plain-text' },
                { text: 'Chat JSONL', link: '/getting-started/datasets#chat-jsonl' },
                { text: 'Training and evaluation files', link: '/getting-started/datasets#training-and-evaluation-files' },
              ],
            },
          ],
        },
      ],
      '/training/': [
        {
          text: 'Training algorithms',
          items: [
            {
              text: 'SFT',
              link: '/training/sft',
              items: [
                { text: 'Minimal configuration', link: '/training/sft#minimal-configuration' },
                { text: 'What is trained', link: '/training/sft#what-is-trained' },
                { text: 'Context and batch geometry', link: '/training/sft#context-and-batch-geometry' },
                { text: 'Evaluation and output', link: '/training/sft#evaluation-and-output' },
                { text: 'SFT parameters', link: '/training/sft#sft-parameters' },
              ],
            },
            {
              text: 'PPO',
              link: '/training/ppo',
              items: [
                { text: 'Configuration', link: '/training/ppo#configuration' },
                { text: 'Update sequence', link: '/training/ppo#update-sequence' },
                { text: 'Reward command protocol', link: '/training/ppo#reward-command-protocol' },
                { text: 'Critic settings', link: '/training/ppo#critic-settings' },
                { text: 'Observing rollouts', link: '/training/ppo#observing-rollouts' },
                { text: 'PPO parameters', link: '/training/ppo#ppo-parameters' },
              ],
            },
            {
              text: 'GRPO',
              link: '/training/grpo',
              items: [
                { text: 'Configuration', link: '/training/grpo#configuration' },
                { text: 'On-policy sampling', link: '/training/grpo#on-policy-sampling' },
                { text: 'Update sequence', link: '/training/grpo#update-sequence' },
                { text: 'Judge configuration', link: '/training/grpo#optional-group-judge' },
                { text: 'Optional GRPO controls', link: '/training/grpo#optional-grpo-controls' },
                { text: 'Observing rollouts', link: '/training/grpo#observing-rollouts' },
                { text: 'Container-backed agentic GRPO', link: '/training/grpo#container-backed-agentic-grpo' },
                { text: 'GRPO parameters', link: '/training/grpo#grpo-parameters' },
              ],
            },
            {
              text: 'Distillation',
              link: '/training/distill',
              items: [
                { text: 'What the objective is', link: '/training/distill#what-the-objective-is' },
                { text: 'Configuration', link: '/training/distill#configuration' },
                { text: 'Update sequence', link: '/training/distill#update-sequence' },
                { text: 'What a run reports', link: '/training/distill#what-a-run-reports' },
                { text: 'Held-out evaluation', link: '/training/distill#held-out-evaluation' },
                { text: 'Measuring against a baseline', link: '/training/distill#measuring-against-a-baseline' },
                { text: 'Distillation parameters', link: '/training/distill#distillation-parameters' },
                { text: 'Offline top-k distillation', link: '/training/distill#offline-top-k-distillation' },
              ],
            },
            {
              text: 'Observing rollouts',
              link: '/training/observe',
              items: [
                { text: 'Viewing a run', link: '/training/observe#viewing-a-run' },
                { text: 'Directory layout', link: '/training/observe#directory-layout' },
                { text: 'Record schema', link: '/training/observe#record-schema' },
                { text: 'Resuming', link: '/training/observe#resuming' },
              ],
            },
          ],
        },
      ],
      '/reference/': [
        {
          text: 'Reference',
          items: [
            { text: 'Configuration', link: '/reference/configuration' },
            { text: 'CLI', link: '/reference/cli' },
            { text: 'Build variants', link: '/reference/builds' },
          ],
        },
      ],
      '/operations/': [
        {
          text: 'Operations',
          items: [
            { text: 'Checkpoints and monitoring', link: '/operations/checkpoints' },
            { text: 'Profiling', link: '/operations/profiling' },
          ],
        },
      ],
      '/engineering/': [
        {
          text: 'Engineering documentation',
          items: [
            { text: 'Overview', link: '/engineering/' },
            { text: 'Contributing', link: '/engineering/contributing' },
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
            { text: 'RIR kernel families', link: '/engineering/rir/KERNELS' },
            { text: 'RIR kernel promotion', link: '/engineering/rir/PROMOTION' },
            { text: 'Support matrix', link: '/engineering/SUPPORT' },
            { text: 'CUDA status', link: '/engineering/cuda/STATUS' },
          ],
        },
        {
          text: 'Implementation notes',
          items: [
            { text: 'The configuration document', link: '/engineering/CONFIG' },
            { text: 'PPO', link: '/engineering/PPO' },
            { text: 'GRPO', link: '/engineering/GRPO' },
            { text: 'Agentic GRPO', link: '/engineering/AGENTIC_GRPO' },
            { text: 'GRPO sampling path', link: '/engineering/optims/SAMPLING' },
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
