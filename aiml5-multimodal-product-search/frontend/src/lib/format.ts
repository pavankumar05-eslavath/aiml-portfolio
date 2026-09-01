/**
 * Small presentation helpers.
 *
 * These live outside `components/ui` deliberately. Vite's fast refresh only
 * applies to modules that export *components exclusively*; mixing plain functions
 * into a component module silently disables hot reloading for everything that
 * imports it.
 */

/** Join class names, dropping falsy values. */
export const cx = (...parts: Array<string | false | null | undefined>): string =>
  parts.filter(Boolean).join(' ')

/**
 * Format a price for display.
 *
 * Falls back to a plain fixed-point string if the runtime cannot format the
 * currency code, so an unusual `currency` value degrades rather than throwing.
 */
export function formatPrice(price: number | null | undefined, currency = 'USD'): string {
  if (price === null || price === undefined) return '—'
  try {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency,
      maximumFractionDigits: 2,
    }).format(price)
  } catch {
    return `${price.toFixed(2)} ${currency}`
  }
}
