/**
 * Presentational primitives shared across pages.
 *
 * This module exports components *only*. `cx` and `formatPrice` live in
 * `@/lib/format` because Vite's fast refresh is disabled for any module that
 * mixes non-component exports with components.
 */

import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cx } from '@/lib/format'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
type ButtonSize = 'sm' | 'md' | 'lg'

const VARIANTS: Record<ButtonVariant, string> = {
  primary:
    'bg-accent-600 text-white hover:bg-accent-700 active:bg-accent-700 disabled:bg-accent-300',
  secondary:
    'bg-white text-ink-700 border border-ink-200 hover:bg-ink-50 hover:border-ink-300 disabled:text-ink-400',
  ghost: 'text-ink-600 hover:bg-ink-100 hover:text-ink-900 disabled:text-ink-300',
  danger: 'bg-red-600 text-white hover:bg-red-700 disabled:bg-red-300',
}

const SIZES: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-xs gap-1.5',
  md: 'h-10 px-4 text-sm gap-2',
  lg: 'h-12 px-6 text-sm gap-2',
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  isLoading?: boolean
  icon?: ReactNode
}

export function Button({
  variant = 'primary',
  size = 'md',
  isLoading = false,
  icon,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || isLoading}
      aria-busy={isLoading || undefined}
      className={cx(
        'inline-flex items-center justify-center rounded-lg font-medium transition-colors',
        'disabled:cursor-not-allowed',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
    >
      {isLoading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  )
}

export function Badge({
  children,
  tone = 'neutral',
  className,
  title,
}: {
  children: ReactNode
  tone?: 'neutral' | 'accent' | 'success' | 'warning' | 'danger'
  className?: string
  title?: string
}) {
  const tones = {
    neutral: 'bg-ink-100 text-ink-600',
    accent: 'bg-accent-100 text-accent-700',
    success: 'bg-emerald-100 text-emerald-700',
    warning: 'bg-amber-100 text-amber-800',
    danger: 'bg-red-100 text-red-700',
  }
  return (
    <span
      title={title}
      className={cx(
        'inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <span role="status" className="text-ink-500 inline-flex items-center gap-2 text-sm">
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}
    </span>
  )
}

export function ErrorBanner({
  title = 'Something went wrong',
  message,
  code,
  onRetry,
}: {
  title?: string
  message: string
  code?: string
  onRetry?: () => void
}) {
  return (
    <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4">
      <div className="flex gap-3">
        <AlertTriangle className="mt-0.5 size-5 shrink-0 text-red-600" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-red-900">{title}</p>
          <p className="mt-1 text-sm text-red-800">{message}</p>
          {code && (
            <p className="mt-1.5 font-mono text-[11px] text-red-600">
              error code: {code}
            </p>
          )}
        </div>
        {onRetry && (
          <Button
            variant="secondary"
            size="sm"
            onClick={onRetry}
            icon={<RefreshCw className="size-3.5" aria-hidden />}
          >
            Retry
          </Button>
        )}
      </div>
    </div>
  )
}

export function EmptyState({
  icon,
  title,
  children,
}: {
  icon?: ReactNode
  title: string
  children?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
      {icon && <div className="text-ink-300 mb-4">{icon}</div>}
      <h3 className="text-ink-800 text-base font-semibold">{title}</h3>
      {children && <div className="text-ink-500 mt-2 max-w-md text-sm">{children}</div>}
    </div>
  )
}

/** Grey placeholder tiles shown while the first page of results loads. */
export function SkeletonGrid({ count = 8 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
      {Array.from({ length: count }, (_, index) => (
        <div key={index} className="card overflow-hidden">
          <div className="bg-ink-100 shimmer aspect-square" />
          <div className="space-y-2 p-3">
            <div className="bg-ink-100 h-3 w-3/4 rounded" />
            <div className="bg-ink-100 h-3 w-1/2 rounded" />
          </div>
        </div>
      ))}
    </div>
  )
}

/** A labelled metric, used on the catalogue dashboard. */
export function Stat({
  label,
  value,
  hint,
  tone = 'neutral',
}: {
  label: string
  value: ReactNode
  hint?: string
  tone?: 'neutral' | 'success' | 'warning' | 'danger'
}) {
  const tones = {
    neutral: 'text-ink-900',
    success: 'text-emerald-600',
    warning: 'text-amber-600',
    danger: 'text-red-600',
  }
  return (
    <div className="card p-4">
      <p className="label">{label}</p>
      <p className={cx('mt-1.5 text-2xl font-semibold tabular-nums', tones[tone])}>{value}</p>
      {hint && <p className="text-ink-400 mt-1 text-xs">{hint}</p>}
    </div>
  )
}
