/**
 * Player context: a tiny event bus around the single WaveSurfer instance so the
 * waveform and the tiers canvas share one time scale and one scroll offset.
 *
 * WaveSurfer owns the horizontal scroller; the tiers panel mirrors its
 * `scrollLeft` and content width so the two are pixel-synced.
 */

import { createContext, useContext } from "react"

export interface Viewport {
  /** Total content width in px = duration * pxPerSec. */
  contentWidth: number
  /** Horizontal scroll offset in px. */
  scrollLeft: number
  /** Visible width of the scroller in px. */
  clientWidth: number
  pxPerSec: number
  duration: number
}

export const EMPTY_VIEWPORT: Viewport = {
  contentWidth: 0,
  scrollLeft: 0,
  clientWidth: 0,
  pxPerSec: 0,
  duration: 0,
}

export interface PlayerApi {
  /** Seek to `time` (seconds) without changing play state. */
  seek: (time: number) => void
  /** Play from `from` and stop automatically at `to`. */
  playRegion: (from: number, to: number) => void
  playPause: () => void
  pause: () => void
  /** Set zoom (px/sec) keeping `anchorTime` at the same screen x. */
  zoomAt: (pxPerSec: number, anchorTime: number, anchorClientX?: number) => void
  scrollBy: (dx: number) => void
  /** Scroll so `time` is comfortably visible. */
  revealTime: (time: number) => void
  getViewport: () => Viewport
  subscribe: (fn: (v: Viewport) => void) => () => void
  isReady: () => boolean
}

const NOOP_API: PlayerApi = {
  seek: () => {},
  playRegion: () => {},
  playPause: () => {},
  pause: () => {},
  zoomAt: () => {},
  scrollBy: () => {},
  revealTime: () => {},
  getViewport: () => EMPTY_VIEWPORT,
  subscribe: () => () => {},
  isReady: () => false,
}

export const PlayerContext = createContext<PlayerApi>(NOOP_API)

export function usePlayer(): PlayerApi {
  return useContext(PlayerContext)
}

/**
 * Mutable holder handed to the context so consumers get a stable identity while
 * the underlying WaveSurfer instance is torn down and rebuilt per file.
 * `impl` is swapped by the Waveform component; `emit` fans a new viewport out
 * to every subscriber (currently the tiers canvas).
 */
export function createPlayerRef() {
  const listeners = new Set<(v: Viewport) => void>()
  const ref = {
    impl: null as PlayerApi | null,
    seek: (t: number) => ref.impl?.seek(t),
    playRegion: (a: number, b: number) => ref.impl?.playRegion(a, b),
    playPause: () => ref.impl?.playPause(),
    pause: () => ref.impl?.pause(),
    zoomAt: (px: number, t: number, x?: number) => ref.impl?.zoomAt(px, t, x),
    scrollBy: (dx: number) => ref.impl?.scrollBy(dx),
    revealTime: (t: number) => ref.impl?.revealTime(t),
    getViewport: () => ref.impl?.getViewport() ?? EMPTY_VIEWPORT,
    isReady: () => ref.impl?.isReady() ?? false,
    subscribe: (fn: (v: Viewport) => void) => {
      listeners.add(fn)
      return () => {
        listeners.delete(fn)
      }
    },
    emit: (v: Viewport) => {
      for (const fn of listeners) fn(v)
    },
  }
  return ref
}

export type PlayerRef = ReturnType<typeof createPlayerRef>
