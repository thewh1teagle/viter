import { useEffect } from "react"
import type { PlayerRef } from "@/lib/player"
import { filterIndices, navTierIndex, neighbourInterval } from "@/lib/time"
import { useStore } from "@/store"

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false
  return (
    el.tagName === "INPUT" ||
    el.tagName === "TEXTAREA" ||
    el.tagName === "SELECT" ||
    el.isContentEditable
  )
}

/** Global shortcuts: Space, ←/→, [/], `/`. */
export function useKeyboard(
  player: PlayerRef,
  searchRef: React.RefObject<HTMLInputElement | null>
) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey) return
      const typing = isTypingTarget(e.target)

      if (e.key === "/" && !typing && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        searchRef.current?.focus()
        searchRef.current?.select()
        return
      }
      if (typing) return
      if (e.ctrlKey || e.metaKey) return

      const s = useStore.getState()

      if (e.code === "Space") {
        e.preventDefault()
        player.playPause()
        return
      }

      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const tg = s.textgrid
        if (!tg || tg.tiers.length === 0) return
        e.preventDefault()
        const ti = navTierIndex(tg.tiers, s.navTier)
        if (ti < 0) return
        const intervals = tg.tiers[ti].intervals
        const dir = e.key === "ArrowRight" ? 1 : -1
        // Navigate relative to the selection when there is one, else the playhead.
        const from =
          s.selection && s.selection.tier === ti
            ? dir === 1
              ? s.selection.xmin
              : s.selection.xmax - 1e-3
            : s.playhead
        const idx = neighbourInterval(intervals, from, dir)
        if (idx < 0) return
        const iv = intervals[idx]
        s.setNavTier(ti)
        s.setSelection({ tier: ti, index: idx, xmin: iv.xmin, xmax: iv.xmax, text: iv.text })
        player.seek(iv.xmin)
        player.revealTime(iv.xmin)
        if (e.shiftKey) player.playRegion(iv.xmin, iv.xmax)
        return
      }

      if (e.key === "[" || e.key === "]") {
        // Same filter the sidebar shows, so [ / ] walks exactly the visible rows.
        const matches = filterIndices(s.fileHaystack, s.appliedQuery)
        const count = matches ? matches.length : s.files.length
        if (count === 0) return
        e.preventDefault()
        const at = (row: number) => s.files[matches ? matches[row] : row]
        let cur = -1
        for (let i = 0; i < count; i++) {
          if (at(i)?.id === s.currentId) {
            cur = i
            break
          }
        }
        const step = e.key === "]" ? 1 : -1
        const next = cur < 0 ? (step === 1 ? 0 : count - 1) : cur + step
        if (next < 0 || next >= count) return
        player.pause()
        s.selectFile(at(next).id)
      }
    }

    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [player, searchRef])
}
