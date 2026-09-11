/**
 * Sample-accurate interval playback.
 *
 * The media element stops on a `timeupdate` poll, which fires every 15–250 ms
 * depending on the browser, so a click on a word always played some of the
 * next one. Here the file is decoded once into an `AudioBuffer` and each
 * interval is played through an `AudioBufferSourceNode` with an explicit
 * duration, which the audio thread cuts at the exact sample. The media element
 * stays paused meanwhile; the caller mirrors the position onto the waveform.
 */

let ctx: AudioContext | null = null

function context(): AudioContext {
  if (!ctx) ctx = new AudioContext()
  return ctx
}

export async function decodeAudio(blob: Blob, signal?: AbortSignal): Promise<AudioBuffer> {
  const bytes = await blob.arrayBuffer()
  if (signal?.aborted) throw new DOMException("aborted", "AbortError")
  return context().decodeAudioData(bytes)
}

export interface RegionPlayback {
  /** Stop early; `onEnd` is not called. */
  stop: () => void
}

/**
 * Play `[start, end)` of `buffer`. `onTick` is called every animation frame
 * with the current position, `onEnd` once the last sample has played.
 */
export function playRegion(
  buffer: AudioBuffer,
  start: number,
  end: number,
  onTick: (t: number) => void,
  onEnd: () => void
): RegionPlayback {
  const ac = context()
  const source = ac.createBufferSource()
  source.buffer = buffer
  source.connect(ac.destination)

  let raf = 0
  let done = false
  const finish = (ended: boolean) => {
    if (done) return
    done = true
    cancelAnimationFrame(raf)
    source.onended = null
    source.disconnect()
    if (ended) onEnd()
  }

  // `resume()` is async on a suspended context (autoplay policy); schedule
  // against the clock only once it is actually running.
  void ac.resume().then(() => {
    if (done) return
    const dur = Math.max(0, end - start)
    const t0 = ac.currentTime
    source.onended = () => finish(true)
    try {
      source.start(t0, start, dur)
    } catch {
      finish(true)
      return
    }
    const tick = () => {
      if (done) return
      onTick(Math.min(end, start + (ac.currentTime - t0)))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
  })

  return {
    stop: () => {
      if (done) return
      finish(false)
      try {
        source.stop()
      } catch {
        /* never started */
      }
    },
  }
}
