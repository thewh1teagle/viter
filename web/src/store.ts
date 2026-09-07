import { create } from "zustand"
import type { FileEntry, TextGrid } from "@/api"

export type Theme = "dark" | "light"

export interface Selection {
  /** Index into `textgrid.tiers`. */
  tier: number
  /** Index into `tier.intervals`. */
  index: number
  xmin: number
  xmax: number
  text: string
}

export const MIN_PX_PER_SEC = 4
export const MAX_PX_PER_SEC = 4000

interface State {
  files: FileEntry[]
  /** `files[i].id` lowercased once, so search never re-lowercases 13k strings. */
  fileHaystack: string[]
  filesLoading: boolean
  filesError: string | null
  /** Live input value; drives the text field only. */
  query: string
  /** `query` after the debounce; drives filtering. */
  appliedQuery: string

  currentId: string | null
  textgrid: TextGrid | null
  textgridLoading: boolean

  /** Zoom, in pixels per second of audio. */
  zoom: number
  playhead: number
  duration: number
  playing: boolean

  selection: Selection | null
  /** Tier last interacted with, drives ←/→ navigation. */
  navTier: number | null

  theme: Theme

  setFiles: (files: FileEntry[]) => void
  setFilesLoading: (loading: boolean) => void
  setFilesError: (error: string | null) => void
  setQuery: (query: string) => void
  setAppliedQuery: (query: string) => void

  selectFile: (id: string | null) => void
  setTextGrid: (tg: TextGrid | null) => void
  setTextGridLoading: (loading: boolean) => void

  setZoom: (zoom: number) => void
  setPlayhead: (t: number) => void
  setDuration: (d: number) => void
  setPlaying: (p: boolean) => void

  setSelection: (sel: Selection | null) => void
  setNavTier: (tier: number | null) => void

  toggleTheme: () => void
  setTheme: (theme: Theme) => void
}

function initialTheme(): Theme {
  if (typeof window === "undefined") return "dark"
  try {
    const stored = window.localStorage.getItem("viter-theme")
    if (stored === "light" || stored === "dark") return stored
  } catch {
    /* private mode */
  }
  return "dark"
}

export function applyTheme(theme: Theme) {
  const root = document.documentElement
  root.classList.toggle("dark", theme === "dark")
  root.style.colorScheme = theme
  try {
    window.localStorage.setItem("viter-theme", theme)
  } catch {
    /* private mode */
  }
}

export const useStore = create<State>((set, get) => ({
  files: [],
  fileHaystack: [],
  filesLoading: true,
  filesError: null,
  query: "",
  appliedQuery: "",

  currentId: null,
  textgrid: null,
  textgridLoading: false,

  zoom: 120,
  playhead: 0,
  duration: 0,
  playing: false,

  selection: null,
  navTier: null,

  theme: initialTheme(),

  setFiles: (files) => set({ files, fileHaystack: files.map((f) => f.id.toLowerCase()) }),
  setFilesLoading: (filesLoading) => set({ filesLoading }),
  setFilesError: (filesError) => set({ filesError }),
  setQuery: (query) => set({ query }),
  setAppliedQuery: (appliedQuery) => set({ appliedQuery }),

  selectFile: (currentId) => {
    if (get().currentId === currentId) return
    const entry = get().files.find((f) => f.id === currentId)
    set({
      currentId,
      textgrid: null,
      textgridLoading: currentId !== null,
      playhead: 0,
      playing: false,
      selection: null,
      navTier: null,
      duration: entry?.duration ?? 0,
    })
  },
  setTextGrid: (textgrid) => set({ textgrid, textgridLoading: false }),
  setTextGridLoading: (textgridLoading) => set({ textgridLoading }),

  setZoom: (zoom) =>
    set({ zoom: Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, zoom)) }),
  setPlayhead: (playhead) => set({ playhead }),
  setDuration: (duration) => set({ duration }),
  setPlaying: (playing) => set({ playing }),

  setSelection: (selection) =>
    set(selection ? { selection, navTier: selection.tier } : { selection: null }),
  setNavTier: (navTier) => set({ navTier }),

  toggleTheme: () => {
    const theme: Theme = get().theme === "dark" ? "light" : "dark"
    applyTheme(theme)
    set({ theme })
  },
  setTheme: (theme) => {
    applyTheme(theme)
    set({ theme })
  },
}))
