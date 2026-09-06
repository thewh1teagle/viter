import { Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { Separator } from "@/components/ui/separator"
import { useStore } from "@/store"
import { Wordmark } from "@/components/Logo"

const HINTS: Array<[string[], string]> = [
  [["Space"], "play"],
  [["←", "→"], "interval"],
  [["[", "]"], "file"],
  [["/"], "search"],
  [["⌘", "scroll"], "zoom"],
]

export function Header() {
  const count = useStore((s) => s.files.length)
  const theme = useStore((s) => s.theme)
  const toggleTheme = useStore((s) => s.toggleTheme)

  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border bg-card px-3">
      <div className="flex items-center gap-2">
        <Wordmark size={20} className="text-sm" />
        <span className="text-[10px] font-medium tracking-widest text-muted-foreground uppercase">
          viewer
        </span>
      </div>

      <Badge variant="secondary" className="h-5 px-1.5 font-mono text-[10px] tabular-nums">
        {count} {count === 1 ? "file" : "files"}
      </Badge>

      <Separator orientation="vertical" className="h-4" />

      <div className="hidden min-w-0 flex-1 items-center gap-3 overflow-hidden lg:flex">
        {HINTS.map(([keys, label]) => (
          <span key={label} className="flex shrink-0 items-center gap-1.5">
            <KbdGroup>
              {keys.map((k) => (
                <Kbd key={k}>{k}</Kbd>
              ))}
            </KbdGroup>
            <span className="text-[11px] text-muted-foreground">{label}</span>
          </span>
        ))}
      </div>

      <div className="ml-auto lg:ml-0">
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={toggleTheme}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </Button>
      </div>
    </header>
  )
}
