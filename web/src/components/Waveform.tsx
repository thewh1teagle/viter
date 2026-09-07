import { useEffect, useRef } from "react"
import WaveSurfer from "wavesurfer.js"
import { toast } from "sonner"
import { audioUrl, fetchPeaks, peaksToWaveSurfer, type FileEntry } from "@/api"
import { useStore } from "@/store"
import type { PlayerRef, Viewport } from "@/lib/player"
import { createWaveformApi } from "@/lib/waveformApi"

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

  // The instance outlives any one file, so everything file-dependent is read
  // through a ref rather than captured in the mount-once effect's closure.
  const fileRef = useRef(file)
  fileRef.current = file

  // Keep the latest zoom without re-creating the instance on every zoom change.
  const zoomRef = useRef(useStore.getState().zoom)

  // ---- Create the WaveSurfer instance once, for the lifetime of the panel. ----
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let disposed = false
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
      const dur = ws.getDuration() || fileRef.current.duration
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

    const ws = WaveSurfer.create({
      container,
      height: WAVE_HEIGHT,
      // No url/peaks here: the constructor would schedule a load of its own,
      // which races with (and revokes) the blob the per-file effect loads.
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
      useStore.getState().setDuration(ws.getDuration() || fileRef.current.duration)
      emit()
    })
    ws.on("redraw", emit)
    ws.on("zoom", emit)
    ws.on("scroll", emit)
    ws.on("timeupdate", (t: number) => {
      useStore.getState().setPlayhead(t)
      const region = regionRef.current
      if (!region || !region.armed) return
      // Stop at the interval end; ignore stale readings from before the seek.
      if (t >= region.end && t >= region.start) {
        regionRef.current = null
        ws.pause()
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
      toast.error(`Could not decode ${fileRef.current.id}`, {
        description: String(err?.message ?? err),
      })
    })

    const impl = createWaveformApi({
      ws,
      player,
      fileRef,
      regionRef,
      zoomRef,
      measure,
      emit,
      getScroller,
      isDisposed: () => disposed,
    })
    player.impl = impl

    const ro = new ResizeObserver(emit)
    ro.observe(container)

    return () => {
      disposed = true
      ro.disconnect()
      if (player.impl === impl) player.impl = null
      wsRef.current = null
      ws.destroy()
    }
  }, [player])

  // ---- Load each newly selected file into the existing instance. ----
  useEffect(() => {
    const ws = wsRef.current
    if (!ws) return
    const abort = new AbortController()
    let cancelled = false
    regionRef.current = null

    // Peaks paint the waveform from the server summary without decoding; the
    // audio is fetched whole so playback never waits on the network. Both go
    // out at once, in parallel with the TextGrid fetch App.tsx starts.
    const peaks = fetchPeaks(file.id, PEAK_COLUMNS, abort.signal).then(
      (p) => ({ peaks: peaksToWaveSurfer(p.peaks), duration: p.duration || file.duration }),
      (err) => {
        if (abort.signal.aborted || cancelled) throw err
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
      .then(([p, blob]) => {
        if (cancelled) return
        if (blob) return ws.loadBlob(blob, p.peaks, p.duration)
        return ws.load(audioUrl(file.id), p.peaks, p.duration)
      })
      .catch(() => {})

    return () => {
      cancelled = true
      // Drop the in-flight peaks/audio requests for a file the user left.
      abort.abort()
    }
  }, [file.id, file.duration])

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
