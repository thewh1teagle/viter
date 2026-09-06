# Viewer

`viter serve <dir>` starts an axum server that scans a directory for aligned output and
serves a single-page React app for inspecting it. The app is compiled into the binary, so a
release build needs no node runtime and no separate install.

## What it shows

A file list on the left with search; on the right, a wavesurfer.js waveform with a tiers panel
below it — `words` and `phones` — drawn on the same time scale, so intervals line up with the
audio under them. Zoom with the wheel, click an interval to play just that interval, hover for
its label and duration. Dark mode by default.

`serve` pairs `x.TextGrid` with `x.wav|flac|mp3` recursively; the id of an entry is its path
relative to the served directory, without extension. Files with only audio or only a TextGrid
are not listed.

## API

All JSON except audio. Anything not under `/api` falls through to the embedded SPA, with an
`index.html` fallback so client-side routes work on reload.

| endpoint | returns |
|---|---|
| `GET /api/files` | `[{ id, audio, textgrid, duration }]` — every audio/TextGrid pair found |
| `GET /api/textgrid/{id}` | `{ xmin, xmax, tiers: [{ name, intervals: [{ xmin, xmax, text }] }] }` |
| `GET /api/audio/{id}` | the audio bytes, with the correct content-type and `Range` support so the browser can seek without downloading the whole file |
| `GET /api/peaks/{id}?px=2000` | `{ peaks: [min, max, ...] }` — min/max pairs precomputed server-side at the requested pixel width |

Peaks are computed on the server because decoding a long flac or mp3 in the browser to draw a
waveform is slow and memory-hungry; the client asks for roughly as many buckets as it has
pixels and draws those. Re-request on zoom when the visible width changes materially.

## Keyboard shortcuts

| key | action |
|---|---|
| `space` | play / pause |
| `←` / `→` | jump to the previous / next interval on the active tier |
| `↑` / `↓` | previous / next file in the list |
| `wheel` | zoom the waveform around the cursor |
| `shift` + `wheel` | scroll horizontally |
| `/` | focus the file search box |
| `esc` | clear search / return focus to the waveform |

## Dev workflow

The web app lives in `web/`. pnpm only — never npm or yarn (`AGENTS.md`).

```bash
cd web
pnpm install
pnpm dev            # vite dev server on :5173, proxying /api -> :7878
```

In a second terminal, run the Rust server against real data so the proxy has something to talk
to:

```bash
cargo run -- serve ./aligned --no-open
```

The proxy is configured in `web/vite.config.ts`; `--no-open` keeps the Rust side from opening a
browser at the wrong port. Hot reload works normally — you are editing the vite app, and only
the API comes from Rust.

For a production build:

```bash
cd web && pnpm build      # tsc -b && vite build -> web/dist
cargo build --release     # rust-embed pulls web/dist into the binary
```

`viter_serve` embeds the app with `#[derive(RustEmbed)] #[folder = "../../web/dist"]`. In
debug builds rust-embed reads `dist/` from disk at request time, so a rebuilt frontend shows up
without recompiling Rust; in release the files are baked in, and `web/dist` must therefore be
built *before* `cargo build --release`. A release binary built against a stale `dist/` ships a
stale UI with no warning — build the frontend first.

## Stack and conventions

Vite + React + TypeScript, Tailwind v4, shadcn/ui components in `web/src/components/ui`,
motion for transitions, wavesurfer.js for the waveform, zustand for state. The store holds the
current file, zoom level and playhead — nothing that can be derived from the TextGrid. API
calls live in `web/src/api.ts`, not scattered through components.

Keep components at or under 300 lines; when one grows past that it is doing two jobs, and the
split is by responsibility rather than by line count.
