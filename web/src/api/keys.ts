// Every query key of the application, so an invalidation names exactly what a
// query cached.

export interface RunFilters {
  status?: string
  algorithm?: string
  cursor?: string
  limit?: number
}

export const keys = {
  runsAll: () => ['runs'] as const,
  runs: (filters: RunFilters) => ['runs', filters] as const,
  run: (id: string) => ['run', id] as const,
  checkpoints: (id: string) => ['run', id, 'checkpoints'] as const,
  artifacts: (id: string) => ['run', id, 'artifacts'] as const,
  trajectoryUpdates: (id: string) => ['run', id, 'trajectories', 'updates'] as const,
  trajectoryUpdate: (id: string, update: number) => ['run', id, 'trajectories', update] as const,
  trajectoryGroup: (id: string, update: number, group: string) =>
    ['run', id, 'trajectories', update, group] as const,
  datasets: () => ['datasets'] as const,
  datasetPreview: (id: string) => ['dataset', id, 'preview'] as const,
  modelFiles: () => ['model-files'] as const,
  capabilities: () => ['capabilities'] as const,
  defaults: () => ['defaults'] as const,
  configSchema: () => ['config-schema'] as const,
  catalog: (kind: CatalogKind) => ['catalog', kind] as const,
  openaiModels: () => ['openai', 'models'] as const,
}

export type CatalogKind = 'rewards' | 'judges' | 'mcp-servers' | 'environments'
