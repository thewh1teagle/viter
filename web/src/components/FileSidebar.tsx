import { useEffect, useMemo, useRef } from "react"
import { FileAudio, Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "cn"
import { formatDuration, matchesQuery } from "@/lib/time"
import { useStore } from "@/store"

const ROW_HEIGHT = 48

export function FileSidebar({ searchRef }: { searchRef: React.RefObject<HTMLInputElement | null> }) {
  const files = useStore((s) => s.files)
  const query = useStore((s) => s.query)
  const setQuery = useStore((s) => s.setQuery)
  const currentId = useStore((s) => s.currentId)
  const selectFile = useStore((s) => s.selectFile)
  const loading = useStore((s) => s.filesLoading)
  const listRef = useRef<HTMLDivElement>(null)

  const visible = useMemo(
    () => files.filter((f) => matchesQuery(f.id, query)),
    [files, query]
  )

  // Keep the selected row in view when `[` / `]` moves the selection.
  useEffect(() => {
    if (!currentId) return
    const el = listRef.current?.querySelector<HTMLElement>(`[data-id="${CSS.escape(currentId)}"]`)
    el?.scrollIntoView({ block: "nearest" })
  }, [currentId])

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
              e.currentTarget.blur()
            }
          }}
          placeholder="Search files…"
          className="h-8 pl-8 text-xs"
          aria-label="Search files"
        />
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div ref={listRef} className="px-2 pb-2">
          {loading && (
            <div className="space-y-1.5 pt-1">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="h-10 animate-pulse rounded-md bg-muted/60" />
              ))}
            </div>
          )}

          {!loading && visible.length === 0 && (
            <p className="px-2 py-6 text-center text-xs text-muted-foreground">
              {files.length === 0 ? "No aligned files." : `No match for “${query}”.`}
            </p>
          )}

          {visible.map((f) => {
            const active = f.id === currentId
            const slash = f.id.lastIndexOf("/")
            const dir = slash >= 0 ? f.id.slice(0, slash + 1) : ""
            const base = slash >= 0 ? f.id.slice(slash + 1) : f.id
            return (
              <button
                key={f.id}
                data-id={f.id}
                type="button"
                onClick={() => selectFile(f.id)}
                style={{ height: ROW_HEIGHT }}
                className={cn(
                  "group flex w-full items-center gap-2 rounded-md px-2 text-left transition-colors",
                  active
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "hover:bg-sidebar-accent/50"
                )}
                aria-current={active ? "true" : undefined}
              >
                <FileAudio
                  className={cn(
                    "size-3.5 shrink-0",
                    active ? "text-foreground" : "text-muted-foreground"
                  )}
                />
                <span className="min-w-0 flex-1">
                  {dir && (
                    <span className="block truncate text-[10px] leading-tight text-muted-foreground">
                      {dir}
                    </span>
                  )}
                  <span
                    className={cn(
                      "block truncate text-xs leading-tight",
                      active ? "font-medium" : ""
                    )}
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
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
