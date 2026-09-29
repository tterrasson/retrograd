import { computed, toValue, type MaybeRefOrGetter } from 'vue'
import { useInfiniteQuery, useQuery } from '@tanstack/vue-query'
import { client, unwrap } from './client'
import { keys } from './keys'
import type {
  GroupDetail,
  GroupSummary,
  MemberDetail,
  MemberSummary,
  Message,
  PromptView,
  StepReward,
  TrajectoryOverview,
  UpdateDetail,
  UpdateSummary,
} from './types'

/**
 * A member with its content beside its figures: the summary, and the
 * completion or the conversation, flattened for the views that read both.
 */
export interface MemberView extends MemberSummary {
  completion?: string | null
  messages?: Message[]
  prefix?: boolean
  step_rewards?: StepReward[]
  terminal_reward_raw?: number | null
  judge_explanation?: string | null
  metadata?: Record<string, unknown>
}

export function memberView(detail: MemberDetail): MemberView {
  const conversation = detail.conversation
  return {
    ...detail.summary,
    completion: detail.completion,
    ...(conversation
      ? {
          messages: conversation.messages,
          prefix: conversation.prefix,
          step_rewards: conversation.step_rewards,
          terminal_reward_raw: conversation.terminal_reward_raw,
          judge_explanation: conversation.judge_explanation,
          metadata: conversation.metadata as Record<string, unknown>,
        }
      : {}),
  }
}

export interface GroupView {
  update: number
  group: number | null
  prompt: PromptView
  members: MemberView[]
}

export function groupView(detail: GroupDetail): GroupView {
  return {
    update: detail.update,
    group: detail.group ?? null,
    prompt: detail.prompt,
    members: detail.members.map(memberView),
  }
}

/** Groups of an update per page. */
export const GROUP_PAGE = 50
/** Characters of each prompt message in the update view; the group view has them whole. */
export const PREVIEW_CHARS = 400

/** The path segment of a group: PPO has none, and answers under `-`. */
export function groupParam(group: number | null | undefined): string {
  return group === null || group === undefined ? '-' : String(group)
}

export function useTrajectoryOverview(
  id: MaybeRefOrGetter<string>,
  enabled: MaybeRefOrGetter<boolean> = true,
  poll: MaybeRefOrGetter<boolean> = false,
) {
  return useQuery<TrajectoryOverview>({
    queryKey: computed(() => keys.trajectoryUpdates(toValue(id))),
    enabled: computed(() => toValue(enabled)),
    // The standalone viewer has no run-event stream. An unfinished update
    // also needs polling even on the control plane, until its final record lands.
    refetchInterval: (query) =>
      toValue(poll) ||
      (query.state.data as TrajectoryOverview | undefined)?.updates.some((entry) => !entry.status)
        ? 3_000
        : false,
    queryFn: ({ signal }): Promise<TrajectoryOverview> =>
      unwrap(
        client.GET('/v1/runs/{id}/trajectories', {
          params: { path: { id: toValue(id) } },
          signal,
        }),
      ),
  })
}

/**
 * The groups of one update, page by page. Never refetched behind the reader's
 * back: an update is read as it was when it was opened.
 */
export function useUpdateDetail(
  id: MaybeRefOrGetter<string>,
  update: MaybeRefOrGetter<number | null>,
) {
  const query = useInfiniteQuery({
    queryKey: computed(() => keys.trajectoryUpdate(toValue(id), toValue(update) ?? -1)),
    enabled: computed(() => toValue(update) !== null),
    initialPageParam: undefined as string | undefined,
    staleTime: Infinity,
    queryFn: ({ pageParam, signal }): Promise<UpdateDetail> =>
      unwrap(
        client.GET('/v1/runs/{id}/trajectories/updates/{update}', {
          params: {
            path: { id: toValue(id), update: String(toValue(update) ?? 0) },
            query: {
              limit: GROUP_PAGE,
              preview_chars: PREVIEW_CHARS,
              ...(pageParam ? { cursor: pageParam } : {}),
            },
          },
          signal,
        }),
      ),
    getNextPageParam: (last) => last.next_cursor ?? undefined,
  })
  const detail = computed<UpdateSummary | null>(() => query.data.value?.pages[0]?.summary ?? null)
  const groups = computed<GroupSummary[]>(
    () => query.data.value?.pages.flatMap((page) => page.groups) ?? [],
  )
  return { ...query, detail, groups }
}

export function useGroupDetail(
  id: MaybeRefOrGetter<string>,
  update: MaybeRefOrGetter<number | null>,
  group: MaybeRefOrGetter<string | null>,
  member?: MaybeRefOrGetter<number | null>,
) {
  return useQuery({
    queryKey: computed(() => [
      ...keys.trajectoryGroup(toValue(id), toValue(update) ?? -1, toValue(group) ?? ''),
      toValue(member) ?? null,
    ]),
    enabled: computed(() => toValue(update) !== null && toValue(group) !== null),
    staleTime: Infinity,
    select: groupView,
    queryFn: ({ signal }) => {
      const selected = toValue(member)
      return unwrap(
        client.GET('/v1/runs/{id}/trajectories/updates/{update}/groups/{group}', {
          params: {
            path: {
              id: toValue(id),
              update: String(toValue(update) ?? 0),
              group: toValue(group) ?? '-',
            },
            query: selected !== null && selected !== undefined ? { member: selected } : {},
          },
          signal,
        }),
      )
    },
  })
}

/** The metrics the update timeline charts, in order, when an update carries them. */
export const TIMELINE_CHARTS: { key: string; band?: string; title: string }[] = [
  { key: 'reward/mean', band: 'reward/std', title: 'reward/mean ± std' },
  { key: 'batch/trained_fraction', title: 'batch/trained_fraction' },
  { key: 'completions/length_mean', title: 'completions/length_mean' },
  { key: 'policy/kl', title: 'policy/kl' },
  { key: 'agent/turns_per_traj_mean', title: 'agent/turns_per_traj_mean' },
  { key: 'agent/tool_calls_per_traj', title: 'agent/tool_calls_per_traj' },
  { key: 'agent/failed_fraction', title: 'agent/failed_fraction' },
]

export interface TimelineSeries {
  key: string
  title: string
  points: { update: number; value: number; spread: number | null; texts: boolean }[]
}

/** One series per charted metric an update actually carries. */
export function timelineSeries(updates: readonly UpdateSummary[]): TimelineSeries[] {
  const out: TimelineSeries[] = []
  for (const chart of TIMELINE_CHARTS) {
    const points = updates.flatMap((entry) => {
      const value = entry.metrics[chart.key]
      if (value === undefined || !Number.isFinite(value)) return []
      const spread = chart.band ? entry.metrics[chart.band] : undefined
      return [
        {
          update: entry.update,
          value,
          spread: spread !== undefined && Number.isFinite(spread) ? spread : null,
          texts: entry.texts,
        },
      ]
    })
    if (points.length) out.push({ key: chart.key, title: chart.title, points })
  }
  return out
}

/** Where a resumed segment starts in the list, for a marker on the timeline. */
export function segmentStarts(updates: readonly UpdateSummary[]): number[] {
  const starts: number[] = []
  for (let index = 1; index < updates.length; index++) {
    const previous = updates[index - 1]
    const current = updates[index]
    if (previous && current && previous.segment !== current.segment) starts.push(current.update)
  }
  return starts
}

export interface GroupStats {
  mean: number | null
  spread: number | null
}

export function groupStats(group: GroupSummary): GroupStats {
  return { mean: group.reward_mean ?? null, spread: group.reward_std ?? null }
}
