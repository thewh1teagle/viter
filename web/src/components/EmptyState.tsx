import { useState } from "react"
import { motion } from "motion/react"
import { Check, Copy, FolderOpen, TriangleAlert } from "lucide-react"
import { Button } from "@/components/ui/button"

const COMMAND = "viter align corpus/ model.viter -o out/ && viter serve out/"

function CommandBlock() {
  const [copied, setCopied] = useState(false)
  return (
    <div className="group relative w-full overflow-hidden rounded-lg border border-border bg-muted/40">
      <pre className="overflow-x-auto px-3 py-2.5 pr-10 text-left font-mono text-[11px] leading-relaxed">
        <code>
          <span className="text-muted-foreground">$ </span>
          {COMMAND}
        </code>
      </pre>
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-1.5 right-1.5 size-6"
        aria-label="Copy command"
        onClick={() => {
          void navigator.clipboard?.writeText(COMMAND).then(() => {
            setCopied(true)
            window.setTimeout(() => setCopied(false), 1500)
          })
        }}
      >
        {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      </Button>
    </div>
  )
}

/** Shown when /api/files came back empty, or when the fetch failed. */
export function EmptyState({ error }: { error?: string | null }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className="flex h-full items-center justify-center p-8"
    >
      <div className="w-full max-w-md space-y-4 text-center">
        <div className="mx-auto flex size-11 items-center justify-center rounded-xl border border-border bg-card">
          {error ? (
            <TriangleAlert className="size-5 text-destructive" />
          ) : (
            <FolderOpen className="size-5 text-muted-foreground" />
          )}
        </div>

        <div className="space-y-1.5">
          <h2 className="text-base font-semibold tracking-tight">
            {error ? "Could not load the file list" : "Nothing to show yet"}
          </h2>
          <p className="text-xs leading-relaxed text-muted-foreground">
            {error
              ? error
              : "This directory has no matched TextGrid + audio pairs. Align a corpus, then serve the output directory:"}
          </p>
        </div>

        {!error && <CommandBlock />}

        <p className="text-[11px] leading-relaxed text-muted-foreground">
          {error
            ? "Check that `viter serve` is running on port 7878."
            : "viter pairs each x.TextGrid with x.wav / .flac / .mp3, recursively."}
        </p>
      </div>
    </motion.div>
  )
}
