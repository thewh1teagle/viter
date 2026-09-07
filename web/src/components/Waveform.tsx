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
  /**
   * Interval being played after a click on a tier. `armed` flips once a time
   * inside the interval has been observed, so a stale reading from before the
   * seek (the previous position, or the file end) cannot cancel the new play.
   */
  const regionRef = useRef<{ start: number; end: number; armed: boolean } | null>(null)
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

    const create = (blob: Blob | null, peaks?: Array<Float32Array | number[]>, duration?: number) => {
      if (disposed) return
      ws = WaveSurfer.create({
        container,
        height: WAVE_HEIGHT,
        // With the blob in hand the media element plays from memory, so a
        // seek to a spot the browser has not buffered yet (anything earlier
        // than where playback first started) does not stall on a range fetch.
        // Without url/peaks the constructor schedules no load of its own,
        // which would otherwise race with (and revoke) the blob below.
        url: blob ? undefined : audioUrl(file.id),
        peaks: blob ? undefined : peaks,
        duration: blob ? undefined : duration,
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
        const region = regionRef.current
        if (!region) return
        if (!region.armed) {
          if (t >= region.start && t < region.end) region.armed = true
          return
        }
        if (t >= region.end) {
          regionRef.current = null
          ws?.pause()
        }
      })
      ws.on("play", () => useStore.getState().setPlaying(true))
      ws.on("pause", () => useStore.getState().setPlaying(false))
      ws.on("finish", () => {
        regionRef.current = null
        useStore.getState().setPlaying(false)
      })
      ws.on("error", (err: Error) => {
        if (disposed) return
        toast.error(`Could not decode ${file.id}`, { description: String(err?.message ?? err) })
      })
      if (blob) void ws.loadBlob(blob, peaks, duration).catch(() => {})
    }

    // Peaks paint the waveform from the server summary without decoding;
    // the audio itself is fetched whole so playback never waits on the network.
    const peaks = fetchPeaks(file.id, PEAK_COLUMNS, abort.signal).then(
      (p) => ({ peaks: peaksToWaveSurfer(p.peaks), duration: p.duration || file.duration }),
      (err) => {
        if (abort.signal.aborted || disposed) throw err
        toast.warning("Peaks unavailable, decoding in the browser", {
          description: String(err?.message ?? err),
        })
        return { peaks: undefined, duration: file.duration }
      }
    )
    const audio = fetch(audioUrl(file.id), { signal: abort.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`audio ${r.status}`)
        return r.blob()
      })
      .catch(() => null)
    Promise.all([peaks, audio])
      .then(([p, blob]) => create(blob, p.peaks, p.duration))
      .catch(() => {})

    const impl = {
      isReady: () => ws !== null,
      getViewport: measure,
      seek: (time: number) => {
        const dur = ws?.getDuration() || file.duration
        if (!ws || dur <= 0) return
        regionRef.current = null
        ws.seekTo(clamp(time, 0, dur) / dur)
        useStore.getState().setPlayhead(clamp(time, 0, dur))
      },
      playRegion: (from: number, to: number) => {
        const dur = ws?.getDuration() || file.duration
        if (!ws || dur <= 0) return
        const w = ws
        const start = clamp(from, 0, dur)
        const end = Math.min(to, dur)
        const region = { start, end, armed: false }
        regionRef.current = region
        // Every click restarts the interval, even while it is still playing.
        w.pause()
        const begin = () => {
          // A later click superseded this one while we waited for metadata.
          if (disposed || regionRef.current !== region) return
          w.setTime(start)
          useStore.getState().setPlayhead(start)
          w.play().catch((err: unknown) => {
            if (regionRef.current === region) regionRef.current = null
            console.warn("play failed", err)
          })
        }
        // Seeking before the media element knows its duration is silently
        // dropped, so the first click on a freshly opened file did nothing.
        const media = w.getMediaElement()
        if (media.readyState >= HTMLMediaElement.HAVE_METADATA) {
          begin()
        } else {
          media.addEventListener("loadedmetadata", begin, { once: true })
          if (media.networkState !== HTMLMediaElement.NETWORK_LOADING) media.load()
        }
      },
      playPause: () => {
        if (!ws) return
        regionRef.current = null
        void ws.playPause()
      },
      pause: () => {
        regionRef.current = null
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
