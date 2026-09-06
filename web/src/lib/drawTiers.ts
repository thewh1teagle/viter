/** Canvas renderer for the TextGrid tiers, pixel-synced to the waveform. */

import type { Tier } from "@/api"
import type { Viewport } from "@/lib/player"
import { isSilence } from "@/lib/time"

/** Canvas `font` accepts no CSS custom properties, so the stack is literal. */
const FONT = "'Geist Variable', ui-sans-serif, system-ui, sans-serif"

export const TIER_HEIGHT = 46
export const TIER_GAP = 4
export const LABEL_PAD = 6

export interface HoverTarget {
  tier: number
  index: number
}

export interface DrawTheme {
  bg: string
  box: string
  boxSilence: string
  boxHover: string
  boxSelected: string
  border: string
  label: string
  labelMuted: string
  labelSelected: string
  playhead: string
  tierName: string
}

export function readTheme(el: HTMLElement): DrawTheme {
  const cs = getComputedStyle(el)
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback
  return {
    bg: v("--card", "#1a1a1a"),
    box: v("--secondary", "#2a2a2a"),
    boxSilence: v("--muted", "#222"),
    boxHover: v("--accent", "#333"),
    boxSelected: v("--primary", "#e5e5e5"),
    border: v("--border", "#3a3a3a"),
    label: v("--foreground", "#eee"),
    labelMuted: v("--muted-foreground", "#888"),
    labelSelected: v("--primary-foreground", "#111"),
    playhead: v("--destructive", "#ef4444"),
    tierName: v("--muted-foreground", "#888"),
  }
}

export function tiersCanvasHeight(tierCount: number): number {
  return Math.max(1, tierCount) * (TIER_HEIGHT + TIER_GAP) + TIER_GAP
}

export function tierTop(i: number): number {
  return TIER_GAP + i * (TIER_HEIGHT + TIER_GAP)
}

/** Which interval is under (x, y)? x is in *content* pixels, y in canvas pixels. */
export function hitTest(
  tiers: Tier[],
  contentX: number,
  y: number,
  pxPerSec: number
): HoverTarget | null {
  if (pxPerSec <= 0) return null
  const time = contentX / pxPerSec
  for (let t = 0; t < tiers.length; t++) {
    const top = tierTop(t)
    if (y < top || y > top + TIER_HEIGHT) continue
    const intervals = tiers[t].intervals
    let lo = 0
    let hi = intervals.length - 1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      const iv = intervals[mid]
      if (time < iv.xmin) hi = mid - 1
      else if (time >= iv.xmax) lo = mid + 1
      else return { tier: t, index: mid }
    }
    return null
  }
  return null
}

export interface DrawArgs {
  ctx: CanvasRenderingContext2D
  tiers: Tier[]
  viewport: Viewport
  playhead: number
  hover: HoverTarget | null
  selected: { tier: number; index: number } | null
  theme: DrawTheme
  /** CSS-pixel width of the canvas (the visible window). */
  width: number
  height: number
}

export function drawTiers({
  ctx,
  tiers,
  viewport,
  playhead,
  hover,
  selected,
  theme,
  width,
  height,
}: DrawArgs) {
  const { pxPerSec, scrollLeft } = viewport
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = theme.bg
  ctx.fillRect(0, 0, width, height)

  if (pxPerSec <= 0) return

  const t0 = scrollLeft / pxPerSec
  const t1 = (scrollLeft + width) / pxPerSec

  ctx.textBaseline = "middle"

  for (let t = 0; t < tiers.length; t++) {
    const tier = tiers[t]
    const top = tierTop(t)
    const intervals = tier.intervals

    // Tier lane background.
    ctx.fillStyle = theme.box
    ctx.globalAlpha = 0.25
    ctx.fillRect(0, top, width, TIER_HEIGHT)
    ctx.globalAlpha = 1

    // Binary-search the first visible interval, then walk forward.
    let start = 0
    {
      let lo = 0
      let hi = intervals.length - 1
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        if (intervals[mid].xmax <= t0) lo = mid + 1
        else hi = mid - 1
      }
      start = lo
    }

    for (let i = start; i < intervals.length; i++) {
      const iv = intervals[i]
      if (iv.xmin > t1) break

      const x = iv.xmin * pxPerSec - scrollLeft
      const w = Math.max(0, (iv.xmax - iv.xmin) * pxPerSec)
      const silent = isSilence(iv.text)
      const isHover = hover?.tier === t && hover.index === i
      const isSel = selected?.tier === t && selected.index === i

      // Box fill.
      ctx.fillStyle = isSel
        ? theme.boxSelected
        : isHover
          ? theme.boxHover
          : silent
            ? theme.boxSilence
            : theme.box
      ctx.globalAlpha = isSel ? 0.9 : silent && !isHover ? 0.4 : 1
      ctx.fillRect(x, top, w, TIER_HEIGHT)
      ctx.globalAlpha = 1

      // Boundary line at the interval start.
      ctx.fillStyle = theme.border
      ctx.fillRect(Math.round(x) + 0.0, top, 1, TIER_HEIGHT)

      // Label, centered, hidden when the box is too narrow to hold it.
      const text = iv.text.trim()
      if (text !== "" && w > 14) {
        ctx.font = `${isSel || isHover ? 600 : 400} 12px ${FONT}`
        const metrics = ctx.measureText(text)
        if (metrics.width + LABEL_PAD * 2 <= w) {
          ctx.fillStyle = isSel ? theme.labelSelected : silent ? theme.labelMuted : theme.label
          // Center within the *visible* portion so labels stay readable while
          // a long interval is partially scrolled off screen.
          const visLeft = Math.max(x, 0)
          const visRight = Math.min(x + w, width)
          const cx = (visLeft + visRight) / 2
          const clamped = Math.min(Math.max(cx, x + metrics.width / 2 + LABEL_PAD), x + w - metrics.width / 2 - LABEL_PAD)
          ctx.textAlign = "center"
          ctx.fillText(text, clamped, top + TIER_HEIGHT / 2)
        }
      }
    }

    // Final boundary of the tier.
    if (intervals.length > 0) {
      const last = intervals[intervals.length - 1]
      const x = last.xmax * pxPerSec - scrollLeft
      if (x >= 0 && x <= width) {
        ctx.fillStyle = theme.border
        ctx.fillRect(Math.round(x), top, 1, TIER_HEIGHT)
      }
    }

    // Lane outline + tier name badge pinned to the left edge.
    ctx.strokeStyle = theme.border
    ctx.lineWidth = 1
    ctx.strokeRect(0.5, top + 0.5, width - 1, TIER_HEIGHT - 1)

    ctx.font = `500 10px ${FONT}`
    ctx.textAlign = "left"
    const name = tier.name
    const nameW = ctx.measureText(name).width
    ctx.fillStyle = theme.bg
    ctx.globalAlpha = 0.85
    ctx.fillRect(4, top + 3, nameW + 8, 14)
    ctx.globalAlpha = 1
    ctx.fillStyle = theme.tierName
    ctx.fillText(name, 8, top + 10)
  }

  // Playhead across all lanes.
  const px = playhead * pxPerSec - scrollLeft
  if (px >= -1 && px <= width + 1) {
    ctx.fillStyle = theme.playhead
    ctx.fillRect(Math.round(px), 0, 1, height)
  }
}
