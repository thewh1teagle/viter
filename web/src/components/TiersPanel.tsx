import { useCallback, useEffect, useRef, useState } from "react"
import type { TextGrid } from "@/api"
import { EMPTY_VIEWPORT, type PlayerRef, type Viewport } from "@/lib/player"
import {
  drawTiers,
  hitTest,
  readTheme,
  tiersCanvasHeight,
  type HoverTarget,
} from "@/lib/drawTiers"
import { formatSpan, formatTime } from "@/lib/time"
import { useStore } from "@/store"

interface TipState {
  x: number
  y: number
  label: string
  xmin: number
  xmax: number
}

export function TiersPanel({ textgrid, player }: { textgrid: TextGrid; player: PlayerRef }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<Viewport>(EMPTY_VIEWPORT)
  const hoverRef = useRef<HoverTarget | null>(null)
  const rafRef = useRef(0)
  const [tip, setTip] = useState<TipState | null>(null)

  const theme = useStore((s) => s.theme)
  const playhead = useStore((s) => s.playhead)
  const selection = useStore((s) => s.selection)
  const setSelection = useStore((s) => s.setSelection)

  const tiers = textgrid.tiers
  const height = tiersCanvasHeight(tiers.length)

  // Latest values for the imperative draw loop, without re-binding listeners.
  const stateRef = useRef({ playhead, selection, tiers })

  const paint = useCallback(() => {
    rafRef.current = 0
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const cssW = wrap.clientWidth
    const cssH = tiersCanvasHeight(stateRef.current.tiers.length)
    const dpr = window.devicePixelRatio || 1
    const needW = Math.round(cssW * dpr)
    const needH = Math.round(cssH * dpr)
    if (canvas.width !== needW || canvas.height !== needH) {
      canvas.width = needW
      canvas.height = needH
      canvas.style.width = `${cssW}px`
      canvas.style.height = `${cssH}px`
    }
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const { playhead: ph, selection: sel, tiers: tr } = stateRef.current
    drawTiers({
      ctx,
      tiers: tr,
      viewport: viewportRef.current,
      playhead: ph,
      hover: hoverRef.current,
      selected: sel ? { tier: sel.tier, index: sel.index } : null,
      theme: readTheme(canvas),
      width: cssW,
      height: cssH,
    })
  }, [])

  const schedule = useCallback(() => {
    if (rafRef.current === 0) rafRef.current = requestAnimationFrame(paint)
  }, [paint])

  // Mirror the waveform viewport (scroll + zoom) and repaint.
  useEffect(() => {
    viewportRef.current = player.getViewport()
    schedule()
    const unsub = player.subscribe((v) => {
      viewportRef.current = v
      schedule()
    })
    const ro = new ResizeObserver(schedule)
    if (wrapRef.current) ro.observe(wrapRef.current)
    return () => {
      unsub()
      ro.disconnect()
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = 0
    }
  }, [player, schedule])

  // Publish the latest reactive values to the draw loop, then repaint.
  useEffect(() => {
    stateRef.current = { playhead, selection, tiers }
    schedule()
  }, [schedule, playhead, selection, tiers, theme])

  const targetAt = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): HoverTarget | null => {
      const canvas = canvasRef.current
      if (!canvas) return null
      const rect = canvas.getBoundingClientRect()
      const v = viewportRef.current
      return hitTest(
        stateRef.current.tiers,
        e.clientX - rect.left + v.scrollLeft,
        e.clientY - rect.top,
        v.pxPerSec
      )
    },
    []
  )

  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const target = targetAt(e)
      const prev = hoverRef.current
      if (prev?.tier !== target?.tier || prev?.index !== target?.index) {
        hoverRef.current = target
        schedule()
      }
      if (target) {
        const iv = stateRef.current.tiers[target.tier].intervals[target.index]
        setTip({
          x: e.clientX,
          y: e.clientY,
          label: iv.text.trim() || "∅",
          xmin: iv.xmin,
          xmax: iv.xmax,
        })
      } else if (tip) {
        setTip(null)
      }
    },
    [targetAt, schedule, tip]
  )

  const onMouseLeave = useCallback(() => {
    if (hoverRef.current) {
      hoverRef.current = null
      schedule()
    }
    setTip(null)
  }, [schedule])

  const onClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const target = targetAt(e)
      if (!target) return
      const iv = stateRef.current.tiers[target.tier].intervals[target.index]
      setSelection({
        tier: target.tier,
        index: target.index,
        xmin: iv.xmin,
        xmax: iv.xmax,
        text: iv.text,
      })
      player.playRegion(iv.xmin, iv.xmax)
    },
    [targetAt, setSelection, player]
  )

  // Ctrl/Cmd+wheel zooms around the cursor; plain wheel scrolls horizontally.
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const onWheel = (e: WheelEvent) => {
      const v = viewportRef.current
      if (v.pxPerSec <= 0) return
      e.preventDefault()
      const rect = wrap.getBoundingClientRect()
      if (e.ctrlKey || e.metaKey) {
        const anchorTime = (e.clientX - rect.left + v.scrollLeft) / v.pxPerSec
        const factor = Math.exp(-e.deltaY * 0.002)
        player.zoomAt(v.pxPerSec * factor, anchorTime, e.clientX)
      } else {
        const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY
        player.scrollBy(dx)
      }
    }
    wrap.addEventListener("wheel", onWheel, { passive: false })
    return () => wrap.removeEventListener("wheel", onWheel)
  }, [player])

  return (
    <div ref={wrapRef} className="relative w-full overflow-hidden" style={{ height }}>
      <canvas
        ref={canvasRef}
        className="block cursor-pointer"
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
        onClick={onClick}
      />
      {tip && (
        <div
          className="pointer-events-none fixed z-50 rounded-md border border-border bg-popover px-2.5 py-1.5 text-xs shadow-md"
          style={{
            left: Math.min(tip.x + 14, window.innerWidth - 190),
            top: Math.max(tip.y - 62, 8),
          }}
        >
          <div className="font-medium text-popover-foreground">{tip.label}</div>
          <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">
            {formatTime(tip.xmin)} → {formatTime(tip.xmax)}
          </div>
          <div className="font-mono text-[11px] text-muted-foreground">
            {formatSpan(tip.xmax - tip.xmin)}
          </div>
        </div>
      )}
    </div>
  )
}
