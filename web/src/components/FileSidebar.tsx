import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { FileAudio, Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import { cn } from "cn"
import { filterIndices, formatDuration } from "@/lib/time"
import { useStore } from "@/store"

const ROW_HEIGHT = 48
/** Rows rendered beyond each edge of the viewport, so a fast scroll stays filled. */
const OVERSCAN = 8
/** Search debounce: long enough to skip intermediate keystrokes, short enough to feel live. */
const SEARCH_DEBOUNCE_MS = 80

export function FileSidebar({ searchRef }: { searchRef: React.RefObject<HTMLInputElement | null> }) {
  const files = useStore((s) => s.files)
  const haystack = useStore((s) => s.fileHaystack)
  const query = useStore((s) => s.query)
  const appliedQuery = useStore((s) => s.appliedQuery)
  const setQuery = useStore((s) => s.setQuery)
  const setAppliedQuery = useStore((s) => s.setAppliedQuery)
  const currentId = useStore((s) => s.currentId)
  const selectFile = useStore((s) => s.selectFile)
  const loading = useStore((s) => s.filesLoading)

  const scrollRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportH, setViewportH] = useState(0)

  // Debounce the query that drives filtering; the input itself stays immediate.
  useEffect(() => {
    if (query === appliedQuery) return
    const t = window.setTimeout(() => setAppliedQuery(query), SEARCH_DEBOUNCE_MS)
    return () => window.clearTimeout(t)
  }, [query, appliedQuery, setAppliedQuery])

  // `null` means "everything matches" — the common case allocates nothing.
  const matches = useMemo(
    () => filterIndices(haystack, appliedQuery),
    [haystack, appliedQuery]
  )
  const count = matches ? matches.length : files.length
  const fileAt = useCallback(
    (row: number) => files[matches ? matches[row] : row],
    [files, matches]
  )

  // Track the scroll offset and the visible height to compute the window.
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onScroll = () => setScrollTop(el.scrollTop)
    el.addEventListener("scroll", onScroll, { passive: true })
    const ro = new ResizeObserver(() => setViewportH(el.clientHeight))
    ro.observe(el)
    setViewportH(el.clientHeight)
    return () => {
      el.removeEventListener("scroll", onScroll)
      ro.disconnect()
    }
  }, [])

  const first = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
  const last = Math.min(count, Math.ceil((scrollTop + viewportH) / ROW_HEIGHT) + OVERSCAN)

  // Keep the selected row in view when `[` / `]` moves the selection. The row
  // may not be mounted, so scroll by arithmetic rather than `scrollIntoView`.
  useEffect(() => {
    const el = scrollRef.current
    if (!el || !currentId) return
    let row = -1
    for (let i = 0; i < count; i++) {
      if (fileAt(i)?.id === currentId) {
        row = i
        break
      }
    }
    if (row < 0) return
    const top = row * ROW_HEIGHT
    const bottom = top + ROW_HEIGHT
    if (top < el.scrollTop) el.scrollTop = top
    else if (bottom > el.scrollTop + el.clientHeight) el.scrollTop = bottom - el.clientHeight
  }, [currentId, count, fileAt])

  const rows = []
  for (let i = first; i < last; i++) {
    const f = fileAt(i)
    if (!f) continue
    const active = f.id === currentId
    const slash = f.id.lastIndexOf("/")
    const dir = slash >= 0 ? f.id.slice(0, slash + 1) : ""
    const base = slash >= 0 ? f.id.slice(slash + 1) : f.id
    rows.push(
      <button
        key={f.id}
        data-id={f.id}
        type="button"
        onClick={() => selectFile(f.id)}
        style={{ height: ROW_HEIGHT, top: i * ROW_HEIGHT }}
        className={cn(
          "group absolute inset-x-0 flex items-center gap-2 rounded-md px-2 text-left transition-colors",
          active
            ? "bg-sidebar-accent text-sidebar-accent-foreground"
            : "hover:bg-sidebar-accent/50"
        )}
        aria-current={active ? "true" : undefined}
      >
        <FileAudio
          className={cn("size-3.5 shrink-0", active ? "text-foreground" : "text-muted-foreground")}
        />
        <span className="min-w-0 flex-1">
          {dir && (
            <span className="block truncate text-[10px] leading-tight text-muted-foreground">
              {dir}
            </span>
          )}
          <span
            className={cn("block truncate text-xs leading-tight", active ? "font-medium" : "")}
            title={f.id}
          >
            {base}
          </span>
        </span>
        <span className="shrink-0 font-mono text-[10px] text-muted-foreground tabular-nums">
          {formatDuration(f.duration)}
        </span>
      </button>
    )
  }

  return (
    <div className="flex h-full flex-col bg-sidebar">
      <div className="relative shrink-0 p-2">
        <Search className="pointer-events-none absolute top-1/2 left-4 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          ref={searchRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              setQuery("")
              setAppliedQuery("")
              e.currentTarget.blur()
            }
          }}
          placeholder="Search files…"
          className="h-8 pl-8 text-xs"
          aria-label="Search files"
        />
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
        <div className="px-2 pb-2">
          {loading && (
            <div className="space-y-1.5 pt-1">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="h-10 animate-pulse rounded-md bg-muted/60" />
              ))}
            </div>
          )}

          {!loading && count === 0 && (
            <p className="px-2 py-6 text-center text-xs text-muted-foreground">
              {files.length === 0 ? "No aligned files." : `No match for “${appliedQuery}”.`}
            </p>
          )}

          {/* Spacer sized to the full list; rows are positioned inside it. */}
          <div className="relative" style={{ height: count * ROW_HEIGHT }}>
            {rows}
          </div>
        </div>
      </div>
    </div>
  )
}
