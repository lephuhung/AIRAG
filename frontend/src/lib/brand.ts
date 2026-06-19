/**
 * Centralized brand configuration.
 *
 * 👉 To change the system logo, replace `src/assets/logo.png` (keep the same
 *    path) OR point `logoSrc` below at a different imported asset / URL.
 *    Every place in the app renders the logo through <BrandLogo>, which reads
 *    from here — so there is exactly ONE place to update.
 */
import logoSrc from "@/assets/logo.png";

export const BRAND = {
  /** Image shown by <BrandLogo>. Swap this to rebrand the whole app. */
  logoSrc,
  /** Fallback alt text (the display name still comes from i18n `app.name`). */
  name: "HRAG",
} as const;

export type Brand = typeof BRAND;
