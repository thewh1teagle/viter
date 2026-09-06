import { useEffect, useRef } from "react"
import WaveSurfer from "wavesurfer.js"
import { toast } from "sonner"
import { audioUrl, fetchPeaks, peaksToWaveSurfer, type FileEntry } from "@/api"
import { MAX_PX_PER_SEC, MIN_PX_PER_SEC, useStore } from "@/store"
import { clamp } from "@/lib/time"
import type { PlayerRef, Viewport } from "@/lib/player"

/** Columns requested from /api/peaks — plenty for a wide screen, cheap to send. */
const PEAK_COLUMNS = 4000
const WAVE_HEIGHT = 132

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

/**
 * WaveSurfer renders into an open shadow root on a wrapper div it appends to
 * our container, so the horizontal scroller (`.scroll`) is not reachable with a
 * plain descendant query — we have to hop through `shadowRoot`.
 */
function findScroller(container: HTMLElement): HTMLElement | null {
  for (const child of Array.from(container.children)) {
    const root = (child as HTMLElement).shadowRoot
    const scroll = root?.querySelector<HTMLElement>(".scroll")
    if (scroll) return scroll
  }
  return null
}

export function Waveform({ file, player }: { file: FileEntry; player: PlayerRef }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const stopAtRef = useRef<number | null>(null)
  const regionStartRef = useRef(0)
  const theme = useStore((s) => s.theme)

  // Keep the latest zoom without re-creating the instance on every zoom change.
  const zoomRef = useRef(useStore.getState().zoom)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let ws: WaveSurfer | null = null
    let disposed = false
    const abort = new AbortController()
    const getScroller = () => findScroller(container)

    /**
     * Read the viewport from the *rendered* geometry rather than from
     * `zoom * duration`: with `fillParent` the waveform is stretched to the
     * container at low zoom levels, so the effective px/sec is wider than the
     * requested minPxPerSec. The tiers canvas must use the same effective
     * scale or it drifts out of sync with the waveform.
     */
    const measure = (): Viewport => {
      const scroller = getScroller()
      const dur = ws?.getDuration() || file.duration
      const clientWidth = scroller?.clientWidth ?? container.clientWidth
      const contentWidth = Math.max(scroller?.scrollWidth ?? 0, clientWidth)
      const pxPerSec = dur > 0 ? contentWidth / dur : zoomRef.current
      return {
        contentWidth,
        scrollLeft: scroller?.scrollLeft ?? 0,
        clientWidth,
        pxPerSec,
        duration: dur,
      }
    }

    const emit = () => player.emit(measure())

    const create = (peaks?: Array<Float32Array | number[]>, duration?: number) => {
      if (disposed) return
      ws = WaveSurfer.create({
        container,
        height: WAVE_HEIGHT,
        url: audioUrl(file.id),
        peaks,
        duration,
        backend: "MediaElement",
        waveColor: cssVar("--chart-2", "#8a8a8a"),
        progressColor: cssVar("--primary", "#e5e5e5"),
        cursorColor: cssVar("--destructive", "#ef4444"),
        cursorWidth: 1,
        minPxPerSec: zoomRef.current,
        fillParent: true,
        autoScroll: true,
        autoCenter: false,
        normalize: true,
        barWidth: 1,
        barGap: 1,
        barRadius: 1,
        interact: true,
        dragToSeek: true,
      })
      wsRef.current = ws

      ws.on("ready", () => {
        const d = ws?.getDuration() ?? file.duration
        useStore.getState().setDuration(d)
        emit()
      })
      ws.on("redraw", emit)
      ws.on("zoom", emit)
      ws.on("scroll", emit)
      ws.on("timeupdate", (t: number) => {
        useStore.getState().setPlayhead(t)
        // Only stop once we are inside the region: a stale timeupdate from
        // the previous position must not cancel a fresh click.
        const stop = stopAtRef.current
        if (stop !== null && t >= stop && t >= regionStartRef.current) {
          stopAtRef.current = null
          ws?.pause()
        }
      })
      ws.on("play", () => useStore.getState().setPlaying(true))
      ws.on("pause", () => useStore.getState().setPlaying(false))
      ws.on("finish", () => {
        stopAtRef.current = null
        useStore.getState().setPlaying(false)
      })
      ws.on("error", (err: Error) => {
        if (disposed) return
        toast.error(`Could not decode ${file.id}`, { description: String(err?.message ?? err) })
      })
    }

    // Peaks first so the waveform paints instantly from the server summary;
    // MediaElement backend then streams the audio without re-decoding.
    fetchPeaks(file.id, PEAK_COLUMNS, abort.signal)
      .then((p) => create(peaksToWaveSurfer(p.peaks), p.duration || file.duration))
      .catch((err) => {
        if (abort.signal.aborted || disposed) return
        toast.warning("Peaks unavailable, decoding in the browser", {
          description: String(err?.message ?? err),
        })
        create(undefined, file.duration)
      })

    const impl = {
      isReady: () => ws !== null,
      getViewport: measure,
      seek: (time: number) => {
        const dur = ws?.getDuration() || file.duration
        if (!ws || dur <= 0) return
        stopAtRef.current = null
        ws.seekTo(clamp(time, 0, dur) / dur)
        useStore.getState().setPlayhead(clamp(time, 0, dur))
      },
      playRegion: (from: number, to: number) => {
        const dur = ws?.getDuration() || file.duration
        if (!ws || dur <= 0) return
        const start = clamp(from, 0, dur)
        regionStartRef.current = start
        stopAtRef.current = Math.min(to, dur)
        ws.setTime(start)
        useStore.getState().setPlayhead(start)
        void ws.play()
      },
      playPause: () => {
        if (!ws) return
        stopAtRef.current = null
        void ws.playPause()
      },
      pause: () => {
        stopAtRef.current = null
        ws?.pause()
      },
      zoomAt: (pxPerSec: number, anchorTime: number, anchorClientX?: number) => {
        const scroller = getScroller()
        if (!ws || !scroller) return
        const next = clamp(pxPerSec, MIN_PX_PER_SEC, MAX_PX_PER_SEC)
        // Keep the anchor under the pointer: its offset from the left edge of
        // the visible window must stay constant across the scale change.
        const rect = scroller.getBoundingClientRect()
        const offset = anchorClientX !== undefined ? anchorClientX - rect.left : scroller.clientWidth / 2
        zoomRef.current = next
        useStore.getState().setZoom(next)
        try {
          ws.zoom(next)
        } catch {
          /* not ready yet; minPxPerSec applies on ready */
        }
        // Re-measure: `fillParent` may render wider than `next` px/sec.
        scroller.scrollLeft = anchorTime * measure().pxPerSec - offset
        emit()
      },
      scrollBy: (dx: number) => {
        const scroller = getScroller()
        if (!scroller) return
        scroller.scrollLeft += dx
        emit()
      },
      revealTime: (time: number) => {
        const scroller = getScroller()
        if (!scroller) return
        const x = time * measure().pxPerSec
        const pad = Math.min(120, scroller.clientWidth * 0.25)
        if (x < scroller.scrollLeft + pad) scroller.scrollLeft = Math.max(0, x - pad)
        else if (x > scroller.scrollLeft + scroller.clientWidth - pad)
          scroller.scrollLeft = x - scroller.clientWidth + pad
        emit()
      },
      subscribe: player.subscribe,
    }
    player.impl = impl

    const ro = new ResizeObserver(emit)
    ro.observe(container)

    return () => {
      disposed = true
      abort.abort()
      ro.disconnect()
      if (player.impl === impl) player.impl = null
      wsRef.current = null
      ws?.destroy()
    }
    // Re-create only when the file changes; zoom/theme are applied imperatively.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file.id, file.duration, player])

  // Re-colour in place when the theme flips.
  useEffect(() => {
    const ws = wsRef.current
    if (!ws) return
    ws.setOptions({
      waveColor: cssVar("--chart-2", "#8a8a8a"),
      progressColor: cssVar("--primary", "#e5e5e5"),
      cursorColor: cssVar("--destructive", "#ef4444"),
    })
  }, [theme])

  return (
    <div
      ref={containerRef}
      className="w-full select-none [&_::part(scroll)]:overscroll-x-contain"
      style={{ height: WAVE_HEIGHT }}
    />
  )
}
