/**
 * The imperative `PlayerApi` implementation over a single WaveSurfer instance.
 * Extracted from `Waveform.tsx` purely to keep that component under the size
 * budget; the semantics — in particular the `playRegion` seek-then-arm dance —
 * are unchanged.
 */

import type WaveSurfer from "wavesurfer.js"
import type { FileEntry } from "@/api"
import { clamp } from "@/lib/time"
import type { PlayerRef, Viewport } from "@/lib/player"
import { playRegion as playBufferRegion, type RegionPlayback } from "@/lib/regionPlayer"
import { MAX_PX_PER_SEC, MIN_PX_PER_SEC, useStore } from "@/store"

export interface Region {
  start: number
  end: number
  armed: boolean
}

export function createWaveformApi(opts: {
  ws: WaveSurfer
  player: PlayerRef
  fileRef: { current: FileEntry }
  regionRef: { current: Region | null }
  bufferRef: { current: { id: string; buffer: AudioBuffer } | null }
  activeRef: { current: RegionPlayback | null }
  zoomRef: { current: number }
  measure: () => Viewport
  emit: () => void
  getScroller: () => HTMLElement | null
  isDisposed: () => boolean
}) {
  const { ws, player, fileRef, regionRef, bufferRef, activeRef, zoomRef, measure, emit, getScroller } =
    opts
  const disposedNow = opts.isDisposed

  /** Cancel a Web Audio interval in progress, leaving the cursor where it is. */
  const stopActive = () => {
    const active = activeRef.current
    if (!active) return
    activeRef.current = null
    active.stop()
    // Leave the media element where the sound stopped, not where it started.
    ws.setTime(useStore.getState().playhead)
    useStore.getState().setPlaying(false)
  }

  /**
   * Sample-accurate path: the media element stays paused and only mirrors the
   * position, so its `timeupdate` poll can no longer overshoot the boundary.
   */
  const playDecoded = (buffer: AudioBuffer, start: number, end: number) => {
    const renderer = ws.getRenderer()
    const dur = ws.getDuration() || fileRef.current.duration
    const media = ws.getMediaElement()
    media.pause()
    const show = (t: number) => {
      useStore.getState().setPlayhead(t)
      if (dur > 0) renderer.renderProgress(t / dur, true)
    }
    show(start)
    useStore.getState().setPlaying(true)
    const playback: RegionPlayback = playBufferRegion(buffer, start, end, show, () => {
      if (activeRef.current !== playback) return
      activeRef.current = null
      // Park the media element at the boundary so a later Space resumes there.
      ws.setTime(end)
      useStore.getState().setPlaying(false)
    })
    activeRef.current = playback
  }

  return {

      isReady: () => true,
      getViewport: measure,
      seek: (time: number) => {
        const dur = ws.getDuration() || fileRef.current.duration
        if (dur <= 0) return
        regionRef.current = null
        stopActive()
        ws.seekTo(clamp(time, 0, dur) / dur)
        useStore.getState().setPlayhead(clamp(time, 0, dur))
      },
      playRegion: (from: number, to: number) => {
        const dur = ws.getDuration() || fileRef.current.duration
        if (dur <= 0) return
        const w = ws
        const start = clamp(from, 0, dur)
        const end = Math.min(to, dur)
        stopActive()
        const decoded = bufferRef.current
        if (decoded && decoded.id === fileRef.current.id) {
          regionRef.current = null
          playDecoded(decoded.buffer, start, end)
          return
        }
        const region = { start, end, armed: false }
        regionRef.current = region
        const media = w.getMediaElement()

        // Drive the media element directly and arm only once the browser has
        // confirmed the seek ("seeked"), so a click on an interval earlier than
        // the current position never depends on the order of timeupdate events.
        const go = () => {
          if (disposedNow() || regionRef.current !== region) return
          region.armed = true
          useStore.getState().setPlayhead(start)
          media.play().catch((err: unknown) => {
            if (regionRef.current === region) regionRef.current = null
            console.warn("play failed", err)
          })
        }
        const seekThenGo = () => {
          if (disposedNow() || regionRef.current !== region) return
          media.pause()
          const already = Math.abs(media.currentTime - start) < 0.005 && !media.seeking
          if (already) {
            go()
            return
          }
          const onSeeked = () => {
            media.removeEventListener("seeked", onSeeked)
            go()
          }
          media.addEventListener("seeked", onSeeked)
          media.currentTime = start
          // Some engines do not fire "seeked" for a seek to the same buffered
          // position; a short fallback keeps the click from being swallowed.
          window.setTimeout(() => {
            if (regionRef.current === region && !region.armed) {
              media.removeEventListener("seeked", onSeeked)
              go()
            }
          }, 250)
        }
        // Seeking before the media element knows its duration is silently
        // dropped, so the first click on a freshly opened file did nothing.
        if (media.readyState >= HTMLMediaElement.HAVE_METADATA) {
          seekThenGo()
        } else {
          media.addEventListener("loadedmetadata", seekThenGo, { once: true })
          if (media.networkState !== HTMLMediaElement.NETWORK_LOADING) media.load()
        }
      },
      playPause: () => {
        regionRef.current = null
        if (activeRef.current) {
          stopActive()
          return
        }
        void ws.playPause()
      },
      pause: () => {
        regionRef.current = null
        stopActive()
        ws.pause()
      },
      zoomAt: (pxPerSec: number, anchorTime: number, anchorClientX?: number) => {
        const scroller = getScroller()
        if (!scroller) return
        const next = clamp(pxPerSec, MIN_PX_PER_SEC, MAX_PX_PER_SEC)
        // Keep the anchor under the pointer: its offset from the left edge of
        // the visible window must stay constant across the scale change.
        const rect = scroller.getBoundingClientRect()
        const offset =
          anchorClientX !== undefined ? anchorClientX - rect.left : scroller.clientWidth / 2
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
      subscribe: player.subscribe
  }
}
