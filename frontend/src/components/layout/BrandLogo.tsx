import { cn } from "@/lib/utils";
import { BRAND } from "@/lib/brand";

interface BrandLogoProps {
  /** Logo box size in pixels (square). Default 32. */
  size?: number;
  /** Render a soft primary glow ring behind the logo. */
  glow?: boolean;
  /** Extra classes for the outer wrapper. */
  className?: string;
  /** Extra classes for the <img>. */
  imgClassName?: string;
  alt?: string;
}

/**
 * The single source of truth for rendering the system logo.
 *
 * Swap the actual image in `src/lib/brand.ts` (or replace
 * `src/assets/logo.png`) — every surface that shows the logo uses this
 * component, so there is one place to rebrand.
 */
export function BrandLogo({
  size = 32,
  glow = false,
  className,
  imgClassName,
  alt = BRAND.name,
}: BrandLogoProps) {
  return (
    <span
      className={cn("relative inline-flex items-center justify-center", className)}
      style={{ width: size, height: size }}
    >
      {glow && (
        <span
          aria-hidden
          className="absolute inset-0 -z-10 rounded-2xl bg-primary/30 blur-xl"
        />
      )}
      <img
        src={BRAND.logoSrc}
        alt={alt}
        width={size}
        height={size}
        className={cn("h-full w-full object-contain select-none", imgClassName)}
        draggable={false}
      />
    </span>
  );
}
