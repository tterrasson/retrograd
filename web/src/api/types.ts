// Names for the generated wire types. Everything here is an alias of
// `schema.d.ts`; a view type the UI builds from them lives next to the code that
// builds it, still under `src/api/`.
import type { components } from './schema'

type S = components['schemas']

export type Capabilities = S['Capabilities']
export type Features = S['Features']
export type Budgets = S['Budgets']
export type Budget = S['Budget']
export type Defaults = S['Defaults']
export type DerivedField = S['DerivedField']
export type ActiveDefault = S['ActiveDefault']

export type RunStatus = S['RunStatus']
export type RunSummary = S['RunSummary']
export type RunView = S['RunView']
export type RunListing = S['RunListing']
export type RunProgress = S['RunProgress']
export type RunEvent = S['RunEvent']
export type RunEventPayload = S['RunEventPayload']
export type MetricsPage = S['MetricsPage']
export type CommandAccepted = S['CommandAccepted']
export type CancelRequest = S['CancelRequest']
export type CancelAt = S['CancelAt']
export type PatchRequest = S['PatchRequest']
export type CheckpointListing = S['CheckpointListing']
export type CheckpointEntry = S['CheckpointEntry']
export type EvaluationResult = S['EvaluationResult']
export type GenerateRequest = S['GenerateRequest']
export type GenerationResult = S['GenerationResult']
export type ArtifactEntry = S['ArtifactEntry']
export type ArtifactListing = S['ArtifactListing']
export type DownloadLink = S['DownloadLink']

export type DatasetView = S['DatasetView']
export type DatasetListing = S['DatasetListing']
export type DatasetPreview = S['DatasetPreview']
export type DatasetTokenization = S['DatasetTokenization']
export type DatasetStats = S['DatasetStatsView']

export type Recipe = S['Recipe']
export type Objective = S['Objective']
export type Allow = S['Allow']
export type DataSpec = S['DataSpec']
export type TrainingBudget = S['TrainingBudget']
export type Limits = S['Limits']
export type ForkFrom = S['ForkFrom']
export type PlanRequest = S['PlanRequest']
export type PlanResponse = S['PlanResponse']
export type PlanSummary = S['PlanSummary']
export type PlanWarning = S['PlanWarning']
export type Provenance = S['Provenance']
export type Origin = S['Origin']
export type Source = S['Source']
export type ResourcePost = S['ResourcePost']
export type PhaseResources = S['PhaseResources']
export type ConfigDocument = S['ConfigDocument']
export type PreflightResponse = S['PreflightResponse']

export type RewardEntry = S['RewardEntry']
export type JudgeEntry = S['JudgeEntry']
export type McpServerEntry = S['McpServerEntry']
export type EnvironmentEntry = S['EnvironmentEntry']

export type ModelFile = S['ModelFile']
export type ModelFileListing = S['ModelFileListing']

export type Problem = S['Problem']
export type FieldError = S['FieldError']
export type ErrorCode = S['ErrorCode']

export type TrajectoryOverview = S['TrajectoryOverview']
export type UpdateSummary = S['UpdateSummary']
export type UpdateDetail = S['UpdateDetail']
export type GroupDetail = S['GroupDetail']

/** One payload of the run stream, by its `type`. */
export type EventOf<T extends RunEventPayload['type']> = Extract<RunEvent, { type: T }>

/** An event of `GET /v1/events`: a run event, and the run it came from. */
export type TaggedRunEvent = RunEvent & { run: string }

export const RUN_STATUSES = [
  'queued',
  'resolving',
  'starting',
  'running',
  'paused',
  'pausing',
  'cancelling',
  'completed',
  'failed',
  'cancelled',
  'interrupted',
] as const satisfies readonly RunStatus[]

// Fails to compile when the contract gains a status the list above lacks.
type MissingStatus = Exclude<RunStatus, (typeof RUN_STATUSES)[number]>
export const RUN_STATUSES_COMPLETE: [MissingStatus] extends [never] ? true : never = true

export type Message = S['MessageView']
export type ToolCall = S['ToolCallView']
export type StepReward = S['StepRewardView']
export type PromptView = S['PromptView']
export type MemberSummary = S['MemberSummary']
export type MemberDetail = S['MemberDetail']
export type GroupSummary = S['GroupSummary']
export type TrajectoryConversation = S['Conversation']
export type ModelFileRole = S['ModelFileRole']
export type ServerMode = S['ServerMode']

export const OBJECTIVES = [
  'instruction-tuning',
  'reasoning-rl',
  'preference-rl',
  'agentic',
  'preference-tuning',
] as const satisfies readonly Objective[]
type MissingObjective = Exclude<Objective, (typeof OBJECTIVES)[number]>
export const OBJECTIVES_COMPLETE: [MissingObjective] extends [never] ? true : never = true

export const ALLOWS = [
  'truncate_context',
  'exceed_train_context',
] as const satisfies readonly Allow[]
type MissingAllow = Exclude<Allow, (typeof ALLOWS)[number]>
export const ALLOWS_COMPLETE: [MissingAllow] extends [never] ? true : never = true
