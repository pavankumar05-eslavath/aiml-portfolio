/**
 * Results area: query telemetry, sort control, the grid, and load-more.
 *
 * The telemetry strip (channels used, weights, latency split) is deliberately
 * visible rather than tucked away in dev tools - it is how a reader of the project
 * sees that ranking is a real pipeline and not a single cosine call.
 */

import { Clock, Layers, SearchX } from 'lucide-react'
import { ProductCard } from '@/components/results/ProductCard'
import { Badge, Button, EmptyState, ErrorBanner, SkeletonGrid, Spinner } from '@/components/ui'
import type { ApiError } from '@/services/api'
import { channelLabel, type SearchResponse, type SearchResult, type SortOption } from '@/types'

interface Props {
  results: SearchResult[]
  response: SearchResponse | null
  isLoading: boolean
  isLoadingMore: boolean
  error: ApiError | null
  hasSearched: boolean
  canLoadMore: boolean
  sort: SortOption
  onSortChange: (sort: SortOption) => void
  onLoadMore: () => void
  onRetry: () => void
}

export function ResultsSection({
  results,
  response,
  isLoading,
  isLoadingMore,
  error,
  hasSearched,
  canLoadMore,
  sort,
  onSortChange,
  onLoadMore,
  onRetry,
}: Props) {
  if (error) {
    return (
      <ErrorBanner
        title={error.isRetryable ? 'The service is not ready yet' : 'Search failed'}
        message={error.message}
        code={error.code}
        onRetry={onRetry}
      />
    )
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Spinner label="Embedding the query and searching…" />
        <SkeletonGrid count={8} />
      </div>
    )
  }

  if (!hasSearched) {
    return (
      <EmptyState icon={<Layers className="size-10" aria-hidden />} title="Search the catalogue">
        Describe a product, upload a photo, or combine both — for example a picture of a brown
        shoe together with <em>“the same style but in black”</em>.
      </EmptyState>
    )
  }

  if (results.length === 0) {
    return (
      <EmptyState icon={<SearchX className="size-10" aria-hidden />} title="No matching products">
        {response?.warnings?.length
          ? response.warnings.join(' ')
          : 'Try removing some filters, or describing the product differently.'}
      </EmptyState>
    )
  }

  return (
    <div className="space-y-4">
      {response && <QueryTelemetry response={response} shown={results.length} />}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-ink-500 text-sm">
          Showing <span className="text-ink-900 font-semibold">{results.length}</span> of{' '}
          <span className="text-ink-900 font-semibold">{response?.total_candidates ?? 0}</span>{' '}
          candidates
        </p>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-ink-500">Sort</span>
          <select
            value={sort}
            onChange={(event) => onSortChange(event.target.value as SortOption)}
            className="border-ink-200 h-9 rounded-lg border bg-white px-2 text-sm"
          >
            <option value="relevance">Relevance</option>
            <option value="price_asc">Price: low to high</option>
            <option value="price_desc">Price: high to low</option>
            <option value="name_asc">Name: A–Z</option>
          </select>
        </label>
      </div>

      {response?.warnings?.map((warning) => (
        <p
          key={warning}
          className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800"
        >
          {warning}
        </p>
      ))}

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        {results.map((result) => (
          <ProductCard key={result.product.id} result={result} />
        ))}
      </div>

      {canLoadMore && (
        <div className="flex justify-center pt-2">
          <Button variant="secondary" onClick={onLoadMore} isLoading={isLoadingMore}>
            Load more
          </Button>
        </div>
      )}
    </div>
  )
}

function QueryTelemetry({ response, shown }: { response: SearchResponse; shown: number }) {
  const { timings } = response
  return (
    <div className="border-ink-200 bg-ink-50/70 rounded-xl border px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        <span className="text-ink-600 inline-flex items-center gap-1.5 font-medium">
          <Layers className="size-3.5" aria-hidden />
          {response.mode} · {response.strategy}
        </span>

        <span className="text-ink-500 inline-flex flex-wrap items-center gap-1">
          {response.channels_used.map((channel) => (
            <Badge key={channel} tone="neutral" title={`weight ${response.weights[channel] ?? 0}`}>
              {channelLabel[channel]} ×{(response.weights[channel] ?? 0).toFixed(2)}
            </Badge>
          ))}
        </span>

        <span className="text-ink-500 ml-auto inline-flex items-center gap-1.5 font-mono tabular-nums">
          <Clock className="size-3.5" aria-hidden />
          {timings.total_ms.toFixed(0)}ms
          <span className="text-ink-400">
            (embed {timings.embed_ms.toFixed(0)} · retrieve {timings.retrieve_ms.toFixed(0)} ·
            fuse {timings.fuse_ms.toFixed(1)} · hydrate {timings.hydrate_ms.toFixed(0)})
          </span>
        </span>
      </div>
      <p className="text-ink-400 mt-1.5 text-[11px]">
        {shown} of {response.total_candidates} distinct products retrieved across{' '}
        {response.channels_used.length} channel(s) · model{' '}
        <span className="font-mono">{response.model_name}</span>
      </p>
    </div>
  )
}
