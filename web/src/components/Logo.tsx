type LogoProps = {
  size?: number
  className?: string
}

export function Logo({ size = 24, className }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      className={className}
      role="img"
      aria-label="viter"
    >
      <g stroke="currentColor" strokeWidth={5} strokeLinecap="round">
        <path d="M24 25v14" />
        <path d="M32 17v30" />
        <path d="M40 25v14" />
      </g>
      <g stroke="#F2694B" strokeWidth={5.5} strokeLinecap="round">
        <path d="M15 19a13.5 13.5 0 0 0 0 26" />
        <path d="M49 19a13.5 13.5 0 0 1 0 26" />
      </g>
    </svg>
  )
}

type WordmarkProps = {
  size?: number
  className?: string
}

export function Wordmark({ size = 24, className }: WordmarkProps) {
  return (
    <span className={`inline-flex items-center gap-2 ${className ?? ""}`}>
      <Logo size={size} />
      <span className="font-semibold tracking-tight">viter</span>
    </span>
  )
}
