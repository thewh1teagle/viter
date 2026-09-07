# viter logo

Accent: `#F2694B` (warm rust coral) — the two claw arcs. Never recolor it.
The mark's three waveform bars carry the theme color; the claws stay coral on every background.

## Which file to use

`logo.svg` / `logo-wordmark.svg` use `currentColor` for the bars — use these **only where CSS color is
inherited** (inline SVG, e.g. `web/src/components/Logo.tsx`). In a standalone `<img>` or a README,
`currentColor` falls back to black and the bars disappear on dark backgrounds.

For anything embedded as an image, pick the baked-in pair:

| File | Bars | Use on |
|---|---|---|
| `logo-light.svg`, `logo-wordmark-light.svg` | `#18181B` | light backgrounds |
| `logo-dark.svg`, `logo-wordmark-dark.svg` | `#E6E6E6` | dark backgrounds |

The repo README selects between them with `<picture>` + `media="(prefers-color-scheme: dark)"`, which
GitHub honors on both themes (a `<style>` block inside the SVG would not — camo strips it).

`logo-dark.png` (light strokes, for dark backgrounds) and `logo-light.png` (dark strokes, for light
backgrounds) are 512px with transparent backgrounds, rendered from the matching SVGs.

`web/public/favicon.svg` keeps its own inline `@media (prefers-color-scheme: …)` style block, which
works because browsers load it as a real document.

Clear space: at least the width of one waveform bar on all sides; minimum size 16px — below that use the favicon.
