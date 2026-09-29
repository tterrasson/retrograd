// Responses of the API the end-to-end tests replay.
//
// Two sources. Under `recorded/` are bodies the real `retrograd-server`
// answered - discovery routes from the control plane, trajectories from
// `retrograd-server view` over `observe/observe.jsonl`. Here are the runs and
// their event streams, which need a trained model to record; they are typed
// against the generated contract, so a field the server renames breaks
// `vue-tsc` rather than a test at runtime.
import type {
  ArtifactListing,
  CheckpointListing,
  ConfigDocument,
  PlanSummary,
  Provenance,
  RunEvent,
  RunEventPayload,
  RunSummary,
  RunView,
} from '../../src/api/types'
import grpoPlan from './recorded/plan-grpo.json'
import sftPlan from './recorded/plan-sft.json'
import sftProvenance from './recorded/provenance-sft.json'

export const SFT_RUN = '3a7c9e10-5b2d-4c8f-9e61-0d4b2a1f7c35'
export const AGENT_RUN = '8e2f4b6a-1c3d-4e5f-a7b9-c0d1e2f3a4b5'

const created = 1_790_000_000

const sftConfig: ConfigDocument = {
  run: { algorithm: 'sft' },
  model: { path: '/models/qwen3-0.6b.gguf' },
  output: { path: '/runs/sft/adapter.gguf' },
  lora: { rank: 16, alpha: 32 },
  training: { lr: 0.0002, epochs: 2, ctx: 1024 },
  sft: { data: '/data/chat.jsonl' },
}

const agentConfig: ConfigDocument = {
  run: { algorithm: 'agent_grpo' },
  model: { path: '/models/qwen3-0.6b.gguf' },
  output: { path: '/runs/agent/adapter.gguf' },
  training: { lr: 0.00001, ctx: 2048 },
  observe: { directory: '/state/agent/observe', every: 2 },
}

// The resolver's own snapshots (`crates/retrograd-plan/tests/snapshots`). A JSON
// import widens every closed vocabulary to `string`, hence the casts.
function view(
  summary: RunSummary,
  effective_config: ConfigDocument,
  plan: unknown,
  provenance: unknown,
): RunView {
  return {
    ...summary,
    holds_device: false,
    effective_config,
    provenance: provenance as Provenance,
    plan: plan as PlanSummary,
  }
}

export const sftSummary: RunSummary = {
  id: SFT_RUN,
  name: 'chat-sft',
  status: 'completed',
  algorithm: 'sft',
  objective: 'instruction-tuning',
  model: 'qwen3-0.6b.gguf',
  observed: false,
  created_at: created,
  started_at: created + 5,
  finished_at: created + 600,
  progress: {
    iteration: 2,
    iterations: 2,
    global_step: 40,
    train_loss: 0.84,
    tokens_per_second: 1850,
  },
}

export const agentSummary: RunSummary = {
  id: AGENT_RUN,
  name: 'search-agent',
  status: 'completed',
  algorithm: 'agent_grpo',
  model: 'qwen3-0.6b.gguf',
  observed: true,
  created_at: created + 1000,
  started_at: created + 1010,
  finished_at: created + 4000,
  progress: { iteration: 3, iterations: 3, global_step: 3, reward: 0.75 },
}

export const runs: Record<string, RunView> = {
  [SFT_RUN]: view(sftSummary, sftConfig, sftPlan, sftProvenance),
  [AGENT_RUN]: view(agentSummary, agentConfig, grpoPlan, {}),
}

export const listing: RunSummary[] = [agentSummary, sftSummary]

/** The agentic run's whole stream, as `events.jsonl` would replay it. */
const agentPayloads: RunEventPayload[] = [
  { type: 'status', status: 'queued' },
  { type: 'status', status: 'starting' },
  { type: 'log', message: 'model loaded in 1.2 s' },
  { type: 'status', status: 'running' },
  ...[1, 2, 3].flatMap((update): RunEventPayload[] => [
    { type: 'log', message: `update ${update}: 4 trajectories sampled` },
    {
      type: 'metrics',
      iteration: update,
      global_step: update,
      values: { 'reward/mean': 0.25 * update, 'policy/kl': 0.01 * update },
    },
    { type: 'progress', iteration: update, iterations: 3, global_step: update, reward: 0.25 * update },
    { type: 'log', message: `update ${update} trained` },
  ]),
  { type: 'checkpoint', path: '/state/agent/checkpoints/step-000000000003.state' },
  { type: 'status', status: 'completed' },
  { type: 'terminal', status: 'completed' },
]

export const agentEvents: RunEvent[] = agentPayloads.map((payload, index) => ({
  ...payload,
  seq: index + 1,
  at: (created + 1010) * 1000 + index * 1000,
}))

const sftPayloads: RunEventPayload[] = [
  { type: 'status', status: 'running' },
  ...[1, 2].flatMap((epoch): RunEventPayload[] => [
    { type: 'metrics', iteration: epoch, global_step: epoch * 20, values: { 'train/loss': 1.2 / epoch } },
    { type: 'progress', iteration: epoch, iterations: 2, global_step: epoch * 20, train_loss: 1.2 / epoch },
  ]),
  { type: 'status', status: 'completed' },
  { type: 'terminal', status: 'completed' },
]

export const sftEvents: RunEvent[] = sftPayloads.map((payload, index) => ({
  ...payload,
  seq: index + 1,
  at: (created + 5) * 1000 + index * 1000,
}))

export const events: Record<string, RunEvent[]> = {
  [SFT_RUN]: sftEvents,
  [AGENT_RUN]: agentEvents,
}

/** Journal lines a client must end with: everything but metrics and progress. */
export function journalLines(stream: RunEvent[]): number {
  return stream.filter((event) => event.type !== 'metrics' && event.type !== 'progress').length
}

export const checkpoints: CheckpointListing = {
  directory: '/state/agent/checkpoints',
  latest: 'step-000000000003',
  checkpoints: [
    {
      id: 'step-000000000003',
      path: '/state/agent/checkpoints/step-000000000003.state',
      kind: 'step',
      global_step: 3,
      bytes: 12_582_912,
      written_at: created + 3900,
      adapter: '/state/agent/checkpoints/step-000000000003.gguf',
      complete: true,
    },
  ],
}

export const artifacts: ArtifactListing = {
  artifacts: [
    {
      name: 'adapter',
      path: '/runs/agent/adapter.gguf',
      kind: 'file',
      present: true,
      bytes: 4_194_304,
      downloadable: true,
    },
    {
      name: 'checkpoints',
      path: '/state/agent/checkpoints',
      kind: 'directory',
      present: true,
      bytes: 12_582_912,
      downloadable: false,
    },
  ],
}
