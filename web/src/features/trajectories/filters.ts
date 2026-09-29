import type { MemberSummary } from '@/api/types'

export type GroupSort = 'id' | 'reward' | 'variance'

export interface TrajectoryFilterState {
  rewardMin: number | null
  rewardMax: number | null
  trainedOnly: boolean
  /** Truncated, skipped or ineligible. */
  flaggedOnly: boolean
  skipReason: string | null
  toolErrorsOnly: boolean
  text: string
  tool: string | null
  compact: boolean
  markdown: boolean
}

export function emptyFilters(): TrajectoryFilterState {
  return {
    rewardMin: null,
    rewardMax: null,
    trainedOnly: false,
    flaggedOnly: false,
    skipReason: null,
    toolErrorsOnly: false,
    text: '',
    tool: null,
    compact: false,
    markdown: false,
  }
}

/** The filters that read what the update view already has: no member content. */
export function acceptsSummary(filters: TrajectoryFilterState, member: MemberSummary): boolean {
  if (filters.trainedOnly && member.trained !== true) return false
  if (filters.flaggedOnly && !(member.truncated || member.skip_reason || member.eligible === false))
    return false
  if (filters.skipReason && member.skip_reason !== filters.skipReason) return false
  if (filters.toolErrorsOnly && !((member.tool_errors ?? 0) > 0)) return false
  const reward = member.reward
  if (filters.rewardMin !== null || filters.rewardMax !== null) {
    if (reward === null || reward === undefined || !Number.isFinite(reward)) return false
    if (filters.rewardMin !== null && reward < filters.rewardMin) return false
    if (filters.rewardMax !== null && reward > filters.rewardMax) return false
  }
  return true
}

export function activeFilterCount(filters: TrajectoryFilterState): number {
  return [
    filters.rewardMin !== null,
    filters.rewardMax !== null,
    filters.trainedOnly,
    filters.flaggedOnly,
    !!filters.skipReason,
    filters.toolErrorsOnly,
    !!filters.text.trim(),
    !!filters.tool,
  ].filter(Boolean).length
}

/** Members by reward, highest first; a member without one sinks. */
export function byReward<T extends { reward?: number | null }>(members: readonly T[]): T[] {
  return [...members].sort((a, b) => (b.reward ?? -Infinity) - (a.reward ?? -Infinity))
}

/** A member of an update, as the comparison and the URL name it. */
export interface MemberRef {
  update: number
  /** The group's path segment: its number, or `-` without groups. */
  group: string
  member: number
}

export function sameMember(a: MemberRef, b: MemberRef): boolean {
  return a.update === b.update && a.group === b.group && a.member === b.member
}

/** `12:3:5` → update 12, group 3, member 5. */
export function parseMemberRef(text: string): MemberRef | null {
  const [update, group, member] = text.split(':')
  if (update === undefined || group === undefined || member === undefined) return null
  const u = Number(update)
  const m = Number(member)
  return Number.isInteger(u) && Number.isInteger(m) && group !== ''
    ? { update: u, group, member: m }
    : null
}

export function formatMemberRef(ref: MemberRef): string {
  return `${ref.update}:${ref.group}:${ref.member}`
}
