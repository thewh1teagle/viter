/** Time / interval helpers shared by the waveform, tiers canvas and status bar. */

import type { Interval, Tier } from "@/api"

/** `1:23.456` — minutes:seconds with millisecond precision. */
export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0
  const m = Math.floor(seconds / 60)
  const s = seconds - m * 60
  return `${m}:${s.toFixed(3).padStart(6, "0")}`
}

/** `1:23` — compact form for the file list. */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds - m * 60)
  return `${m}:${String(s).padStart(2, "0")}`
}

/** `340 ms` / `1.24 s` */
export function formatSpan(seconds: number): string {
  const ms = seconds * 1000
  return ms < 1000 ? `${Math.round(ms)} ms` : `${seconds.toFixed(2)} s`
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

/** Praat treats empty / `sil` / `sp` intervals as silence; we render them dimmed. */
export function isSilence(text: string): boolean {
  const t = text.trim().toLowerCase()
  return t === "" || t === "sil" || t === "sp" || t === "spn" || t === "<eps>"
}

/** Index of the interval containing `time`, or -1. Intervals are sorted & contiguous. */
export function intervalAt(intervals: Interval[], time: number): number {
  let lo = 0
  let hi = intervals.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const iv = intervals[mid]
    if (time < iv.xmin) hi = mid - 1
    else if (time >= iv.xmax) lo = mid + 1
    else return mid
  }
  return -1
}

/**
 * Neighbour interval for ←/→ navigation. Skips silence so arrow keys walk
 * real phones/words. Returns -1 when there is nothing further in that direction.
 */
export function neighbourInterval(
  intervals: Interval[],
  time: number,
  dir: 1 | -1
): number {
  if (intervals.length === 0) return -1
  const eps = 1e-4
  if (dir === 1) {
    for (let i = 0; i < intervals.length; i++) {
      if (intervals[i].xmin > time + eps && !isSilence(intervals[i].text)) return i
    }
    for (let i = 0; i < intervals.length; i++) {
      if (intervals[i].xmin > time + eps) return i
    }
    return -1
  }
  for (let i = intervals.length - 1; i >= 0; i--) {
    if (intervals[i].xmin < time - eps && !isSilence(intervals[i].text)) return i
  }
  for (let i = intervals.length - 1; i >= 0; i--) {
    if (intervals[i].xmin < time - eps) return i
  }
  return -1
}

/** Preferred navigation tier: the last clicked one, else `phones`, else the last tier. */
export function navTierIndex(tiers: Tier[], preferred: number | null): number {
  if (preferred !== null && preferred >= 0 && preferred < tiers.length) return preferred
  const phones = tiers.findIndex((t) => /phone/i.test(t.name))
  if (phones >= 0) return phones
  return tiers.length - 1
}

/** Case-insensitive subsequence-free substring match used by the sidebar search. */
export function matchesQuery(id: string, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (q === "") return true
  const hay = id.toLowerCase()
  return q.split(/\s+/).every((part) => hay.includes(part))
}

/**
 * Indices of the entries matching `query`, scanning a haystack of ids that were
 * lowercased once when the list was loaded. Returning indices (rather than
 * sliced `FileEntry` objects) keeps the filter allocation-free for the common
 * empty query, which is what the virtualised list wants.
 */
export function filterIndices(haystack: string[], query: string): number[] | null {
  const q = query.trim().toLowerCase()
  if (q === "") return null
  const parts = q.split(/\s+/)
  const out: number[] = []
  for (let i = 0; i < haystack.length; i++) {
    const hay = haystack[i]
    let ok = true
    for (const part of parts) {
      if (!hay.includes(part)) {
        ok = false
        break
      }
    }
    if (ok) out.push(i)
  }
  return out
}
