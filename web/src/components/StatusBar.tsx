import { Pause, Play } from "lucide-react"
import { formatSpan, formatTime } from "@/lib/time"
import { useStore } from "@/store"

function Cell({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-[10px] tracking-wider text-muted-foreground uppercase">{label}</span>
      <span className="font-mono text-[11px] tabular-nums">{children}</span>
    </span>
  )
}

export function StatusBar() {
  const playhead = useStore((s) => s.playhead)
  const duration = useStore((s) => s.duration)
  const zoom = useStore((s) => s.zoom)
  const playing = useStore((s) => s.playing)
  const selection = useStore((s) => s.selection)

  return (
    <footer className="flex h-8 shrink-0 items-center gap-4 overflow-hidden border-t border-border bg-card px-3">
      <span className="flex items-center gap-1.5 text-muted-foreground">
        {playing ? <Pause className="size-3" /> : <Play className="size-3" />}
      </span>

      <Cell label="time">
        {formatTime(playhead)}
        <span className="text-muted-foreground"> / {formatTime(duration)}</span>
      </Cell>

      <Cell label="zoom">{Math.round(zoom)} px/s</Cell>

      {selection ? (
        <span className="flex min-w-0 items-baseline gap-1.5">
          <span className="text-[10px] tracking-wider text-muted-foreground uppercase">sel</span>
          <span className="truncate text-[11px] font-medium">
            {selection.text.trim() || "∅"}
          </span>
          <span className="shrink-0 font-mono text-[11px] text-muted-foreground tabular-nums">
            {formatTime(selection.xmin)}–{formatTime(selection.xmax)} ·{" "}
            {formatSpan(selection.xmax - selection.xmin)}
          </span>
        </span>
      ) : (
        <span className="text-[11px] text-muted-foreground">No interval selected</span>
      )}
    </footer>
  )
}
