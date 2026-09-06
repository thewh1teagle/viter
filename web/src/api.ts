/** Typed client for the viter_serve JSON API (port 7878, proxied at /api in dev). */

export interface FileEntry {
  /** Relative path without extension. May contain slashes. */
  id: string
  /** Relative path of the audio file. */
  audio: string
  /** Relative path of the TextGrid file. */
  textgrid: string
  /** Duration in seconds. */
  duration: number
}

export interface Interval {
  xmin: number
  xmax: number
  text: string
}

export interface Tier {
  name: string
  intervals: Interval[]
}

export interface TextGrid {
  xmin: number
  xmax: number
  tiers: Tier[]
}

export interface Peaks {
  sample_rate: number
  duration: number
  /** Interleaved [min, max, min, max, ...], one pair per column. */
  peaks: number[]
}

/** Ids may contain slashes; encode each segment but keep the separators. */
export function encodeId(id: string): string {
  return id.split("/").map(encodeURIComponent).join("/")
}

export class ApiError extends Error {
  readonly status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, { signal, headers: { accept: "application/json" } })
  } catch (err) {
    if (signal?.aborted) throw err
    throw new ApiError(`Cannot reach the viter server (${url})`)
  }
  if (!res.ok) throw new ApiError(`${res.status} ${res.statusText} for ${url}`, res.status)
  try {
    return (await res.json()) as T
  } catch {
    throw new ApiError(`Malformed JSON from ${url}`)
  }
}

export function fetchFiles(signal?: AbortSignal): Promise<FileEntry[]> {
  return getJson<FileEntry[]>("/api/files", signal)
}

export function fetchTextGrid(id: string, signal?: AbortSignal): Promise<TextGrid> {
  return getJson<TextGrid>(`/api/textgrid/${encodeId(id)}`, signal)
}

export function fetchPeaks(id: string, px: number, signal?: AbortSignal): Promise<Peaks> {
  return getJson<Peaks>(`/api/peaks/${encodeId(id)}?px=${Math.max(1, Math.round(px))}`, signal)
}

export function audioUrl(id: string): string {
  return `/api/audio/${encodeId(id)}`
}

/**
 * WaveSurfer takes `peaks` as `Array<channel>`, each channel a flat sample
 * array it reduces to min/max per rendered column. The server already sends
 * interleaved [min, max, min, max, ...], which is exactly such a sample
 * sequence at one pair per column — so a single channel passed through
 * verbatim reproduces the server's envelope.
 */
export function peaksToWaveSurfer(peaks: number[]): Float32Array[] {
  return [Float32Array.from(peaks)]
}
