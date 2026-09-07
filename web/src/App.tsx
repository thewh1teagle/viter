import { useEffect, useMemo, useRef } from "react"
import { toast } from "sonner"
import { ApiError, fetchFiles, fetchTextGrid } from "@/api"
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable"
import { Toaster } from "@/components/ui/sonner"
import { EmptyState } from "@/components/EmptyState"
import { FileSidebar } from "@/components/FileSidebar"
import { Header } from "@/components/Header"
import { StatusBar } from "@/components/StatusBar"
import { Viewer } from "@/components/Viewer"
import { createPlayerRef } from "@/lib/player"
import { useKeyboard } from "@/lib/useKeyboard"
import { useStore } from "@/store"

export default function App() {
  const player = useMemo(() => createPlayerRef(), [])
  const searchRef = useRef<HTMLInputElement>(null)

  const files = useStore((s) => s.files)
  const filesLoading = useStore((s) => s.filesLoading)
  const filesError = useStore((s) => s.filesError)
  const currentId = useStore((s) => s.currentId)
  const textgrid = useStore((s) => s.textgrid)
  const textgridLoading = useStore((s) => s.textgridLoading)
  const theme = useStore((s) => s.theme)

  useKeyboard(player, searchRef)

  // Load the file list once, then auto-select the first file.
  useEffect(() => {
    const abort = new AbortController()
    const s = useStore.getState()
    s.setFilesLoading(true)
    fetchFiles(abort.signal)
      .then((list) => {
        if (abort.signal.aborted) return
        s.setFiles(list)
        s.setFilesError(null)
        s.setFilesLoading(false)
        if (list.length > 0) s.selectFile(list[0].id)
      })
      .catch((err) => {
        if (abort.signal.aborted) return
        const msg = err instanceof ApiError ? err.message : String(err?.message ?? err)
        s.setFilesError(msg)
        s.setFilesLoading(false)
        toast.error("Failed to load files", { description: msg })
      })
    return () => abort.abort()
  }, [])

  // Load the TextGrid for the selected file.
  useEffect(() => {
    if (!currentId) return
    const abort = new AbortController()
    const s = useStore.getState()
    s.setTextGridLoading(true)
    fetchTextGrid(currentId, abort.signal)
      .then((tg) => {
        if (abort.signal.aborted) return
        useStore.getState().setTextGrid(tg)
      })
      .catch((err) => {
        if (abort.signal.aborted) return
        useStore.getState().setTextGrid(null)
        const msg = err instanceof ApiError ? err.message : String(err?.message ?? err)
        toast.error(`Failed to load TextGrid for ${currentId}`, { description: msg })
      })
    return () => abort.abort()
  }, [currentId])

  const current = useMemo(
    () => (currentId ? (files.find((f) => f.id === currentId) ?? null) : null),
    [files, currentId]
  )
  const showEmpty = !filesLoading && (files.length === 0 || filesError !== null)

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-background text-foreground">
      <Header />

      <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1">
        <ResizablePanel defaultSize="22" minSize="12" maxSize="40" className="min-w-0">
          <FileSidebar searchRef={searchRef} />
        </ResizablePanel>

        <ResizableHandle withHandle />

        <ResizablePanel defaultSize="78" className="min-w-0">
          <div className="h-full min-h-0">
            {showEmpty ? (
              <EmptyState error={filesError} />
            ) : (
              current && (
                // Deliberately *not* keyed by file id: remounting per file tore
                // down and rebuilt WaveSurfer (and, with `mode="wait"`, only
                // after the exit animation finished). The Viewer now stays
                // mounted and swaps its source in place.
                <Viewer
                  file={current}
                  textgrid={textgrid}
                  loading={textgridLoading}
                  player={player}
                />
              )
            )}
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>

      <StatusBar />
      <Toaster position="bottom-right" theme={theme} />
    </div>
  )
}
