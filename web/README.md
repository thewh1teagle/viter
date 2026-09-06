# viter web viewer

Vite + React + TypeScript. Tailwind v4, shadcn/ui (base-nova style, Geist font), motion, lucide-react, wavesurfer.js, zustand.

    pnpm dev      # dev server, proxies /api to viter serve on 127.0.0.1:7878
    pnpm build    # -> dist/, embedded into the viter binary by crates/serve via rust-embed

Add shadcn components with `pnpm dlx shadcn@latest add <name>`.
