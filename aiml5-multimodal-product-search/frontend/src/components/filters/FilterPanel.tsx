/**
 * Faceted filter sidebar.
 *
 * Facet values come from `/products/facets`, so the panel reflects the data that
 * is actually indexed rather than a hardcoded list. Selections are pushed into
 * Qdrant's payload filter server-side, not applied to the returned page.
 */

import { SlidersHorizontal, X } from 'lucide-react'
import { useState } from 'react'
import { Badge, Button } from '@/components/ui'
import { cx } from '@/lib/format'
import { countActiveFilters, emptyFilters, type CatalogFacets, type SearchFilters } from '@/types'

interface Props {
  facets: CatalogFacets | null
  filters: SearchFilters
  onChange: (filters: SearchFilters) => void
  onApply: () => void
}

export function FilterPanel({ facets, filters, onChange, onApply }: Props) {
  const activeCount = countActiveFilters(filters)

  const toggle = (key: 'categories' | 'brands' | 'colours' | 'genders', value: string) => {
    const current = filters[key]
    onChange({
      ...filters,
      [key]: current.includes(value)
        ? current.filter((item) => item !== value)
        : [...current, value],
    })
  }

  return (
    <aside className="card sticky top-20 p-4" aria-label="Filters">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-ink-800 inline-flex items-center gap-1.5 text-sm font-semibold">
          <SlidersHorizontal className="size-4" aria-hidden />
          Filters
          {activeCount > 0 && <Badge tone="accent">{activeCount}</Badge>}
        </h2>
        {activeCount > 0 && (
          <button
            type="button"
            onClick={() => {
              onChange(emptyFilters())
              onApply()
            }}
            className="text-ink-400 hover:text-ink-700 inline-flex items-center gap-1 text-xs"
          >
            <X className="size-3" aria-hidden />
            Clear
          </button>
        )}
      </div>

      <div className="space-y-4">
        <PriceRange
          min={filters.min_price}
          max={filters.max_price}
          bounds={{ min: facets?.price_min ?? null, max: facets?.price_max ?? null }}
          onChange={(min, max) => onChange({ ...filters, min_price: min, max_price: max })}
        />

        <FacetGroup
          title="Category"
          values={facets?.categories ?? []}
          selected={filters.categories}
          onToggle={(value) => toggle('categories', value)}
        />

        <FacetGroup
          title="Brand"
          values={facets?.brands ?? []}
          selected={filters.brands}
          onToggle={(value) => toggle('brands', value)}
          maxVisible={8}
        />

        <FacetGroup
          title="Colour"
          values={facets?.colours ?? []}
          selected={filters.colours}
          onToggle={(value) => toggle('colours', value)}
          maxVisible={8}
        />

        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={filters.in_stock_only}
            onChange={(event) => onChange({ ...filters, in_stock_only: event.target.checked })}
            className="accent-accent-600 size-4 rounded"
          />
          <span className="text-ink-700">In stock only</span>
        </label>

        <Button onClick={onApply} className="w-full" size="sm">
          Apply filters
        </Button>
      </div>
    </aside>
  )
}

function FacetGroup({
  title,
  values,
  selected,
  onToggle,
  maxVisible = 6,
}: {
  title: string
  values: Array<{ value: string; count: number }>
  selected: string[]
  onToggle: (value: string) => void
  maxVisible?: number
}) {
  const [showAll, setShowAll] = useState(false)
  if (values.length === 0) return null

  // Keep selected values visible even when they fall outside the truncated head,
  // otherwise a filter can be active with no way to see or remove it.
  const head = values.slice(0, maxVisible)
  const selectedOutside = values.filter(
    (item) => selected.includes(item.value) && !head.includes(item),
  )
  const visible = showAll ? values : [...head, ...selectedOutside]

  return (
    <div>
      <p className="label mb-1.5">{title}</p>
      <div className="scroll-thin max-h-52 space-y-1 overflow-y-auto pr-1">
        {visible.map(({ value, count }) => (
          <label
            key={value}
            className={cx(
              'flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 text-sm transition-colors',
              selected.includes(value) ? 'bg-accent-50' : 'hover:bg-ink-50',
            )}
          >
            <input
              type="checkbox"
              checked={selected.includes(value)}
              onChange={() => onToggle(value)}
              className="accent-accent-600 size-3.5 rounded"
            />
            <span className="text-ink-700 min-w-0 flex-1 truncate" title={value}>
              {value}
            </span>
            <span className="text-ink-400 shrink-0 text-xs tabular-nums">{count}</span>
          </label>
        ))}
      </div>
      {values.length > maxVisible && (
        <button
          type="button"
          onClick={() => setShowAll((open) => !open)}
          className="text-accent-600 hover:text-accent-700 mt-1 text-xs font-medium"
        >
          {showAll ? 'Show fewer' : `Show all ${values.length}`}
        </button>
      )}
    </div>
  )
}

function PriceRange({
  min,
  max,
  bounds,
  onChange,
}: {
  min: number | null
  max: number | null
  bounds: { min: number | null; max: number | null }
  onChange: (min: number | null, max: number | null) => void
}) {
  const parse = (raw: string): number | null => {
    if (raw.trim() === '') return null
    const value = Number(raw)
    return Number.isFinite(value) && value >= 0 ? value : null
  }

  const invalid = min !== null && max !== null && min > max

  return (
    <div>
      <p className="label mb-1.5">Price</p>
      <div className="flex items-center gap-2">
        <input
          type="number"
          min={0}
          inputMode="decimal"
          value={min ?? ''}
          onChange={(event) => onChange(parse(event.target.value), max)}
          placeholder={bounds.min !== null ? bounds.min.toFixed(0) : 'min'}
          aria-label="Minimum price"
          className="border-ink-200 h-9 w-full rounded-lg border px-2 text-sm tabular-nums"
        />
        <span className="text-ink-400 text-xs">to</span>
        <input
          type="number"
          min={0}
          inputMode="decimal"
          value={max ?? ''}
          onChange={(event) => onChange(min, parse(event.target.value))}
          placeholder={bounds.max !== null ? bounds.max.toFixed(0) : 'max'}
          aria-label="Maximum price"
          className="border-ink-200 h-9 w-full rounded-lg border px-2 text-sm tabular-nums"
        />
      </div>
      {invalid && (
        <p role="alert" className="mt-1 text-xs text-red-600">
          Minimum price is above the maximum.
        </p>
      )}
      <p className="text-ink-400 mt-1 text-[11px]">
        Prices in this dataset are synthetic — see the project README.
      </p>
    </div>
  )
}
