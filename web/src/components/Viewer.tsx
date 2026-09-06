import { Loader2, Minus, Pause, Play, Plus } from "lucide-react"
import { motion } from "motion/react"
import type { FileEntry, TextGrid } from "@/api"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Waveform } from "@/components/Waveform"
import { TiersPanel } from "@/components/TiersPanel"
import type { PlayerRef } from "@/lib/player"
import { useStore } from "@/store"

export function Viewer({
  file,
  textgrid,
  loading,
  player,
}: {
  file: FileEntry
  textgrid: TextGrid | null
  loading: boolean
  player: PlayerRef
}) {
  const playing = useStore((s) => s.playing)
  const zoom = useStore((s) => s.zoom)
  const playhead = useStore((s) => s.playhead)

  const zoomBy = (factor: number) => player.zoomAt(zoom * factor, playhead)

  return (
    <motion.div
      key={file.id}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
      className="flex h-full min-h-0 flex-col"
    >
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-border px-3">
        <Button
          variant="secondary"
          size="icon"
          className="size-7"
          onClick={() => player.playPause()}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
        </Button>

        <Separator orientation="vertical" className="h-4" />

        <div className="flex items-center gap-0.5">
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            onClick={() => zoomBy(1 / 1.5)}
            aria-label="Zoom out"
          >
            <Minus className="size-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            onClick={() => zoomBy(1.5)}
            aria-label="Zoom in"
          >
            <Plus className="size-3.5" />
          </Button>
        </div>

        <Separator orientation="vertical" className="h-4" />

        <span className="min-w-0 truncate text-xs text-muted-foreground" title={file.id}>
          {file.id}
        </span>

        {loading && <Loader2 className="ml-auto size-3.5 shrink-0 animate-spin text-muted-foreground" />}
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="w-full max-w-full overflow-x-hidden p-3">
          <div className="overflow-hidden rounded-lg border border-border bg-card">
            <Waveform file={file} player={player} />
          </div>

          <div className="mt-2 overflow-hidden rounded-lg border border-border bg-card">
            {textgrid && textgrid.tiers.length > 0 ? (
              <TiersPanel textgrid={textgrid} player={player} />
            ) : (
              <div className="flex h-24 items-center justify-center text-xs text-muted-foreground">
                {loading ? "Loading TextGrid…" : "This TextGrid has no tiers."}
              </div>
            )}
          </div>
        </div>
      </ScrollArea>
    </motion.div>
  )
}
