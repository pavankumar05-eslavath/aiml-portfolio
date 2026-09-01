/**
 * Product detail page.
 *
 * Also runs a "more like this" search using the product's *stored* image vector,
 * which demonstrates the visual-similarity path without re-encoding an image.
 */

import { ArrowLeft, ImageOff, PackageX } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ProductCard } from '@/components/results/ProductCard'
import { Badge, EmptyState, ErrorBanner, SkeletonGrid, Spinner } from '@/components/ui'
import { formatPrice } from '@/lib/format'
import { ApiError, api, imageUrl } from '@/services/api'
import { emptyFilters, type Product, type SearchResponse } from '@/types'

export function ProductPage() {
  const { id = '' } = useParams<{ id: string }>()
  const [product, setProduct] = useState<Product | null>(null)
  const [similar, setSimilar] = useState<SearchResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingSimilar, setIsLoadingSimilar] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setIsLoading(true)
    setError(null)
    setSimilar(null)

    api
      .getProduct(id, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return
        setProduct(data)
        setIsLoadingSimilar(true)
        return api
          .searchSimilar(
            id,
            {
              top_k: 8,
              offset: 0,
              filters: emptyFilters(),
              sort: 'relevance',
              explain: true,
            },
            controller.signal,
          )
          .then((response) => {
            if (!controller.signal.aborted) setSimilar(response)
          })
          .catch(() => {
            // An unindexed product has no vectors to compare; the detail view is
            // still perfectly useful without recommendations.
          })
          .finally(() => {
            if (!controller.signal.aborted) setIsLoadingSimilar(false)
          })
      })
      .catch((caught) => {
        if (controller.signal.aborted) return
        setError(caught instanceof ApiError ? caught : new ApiError('Failed to load.', 0))
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoading(false)
      })

    return () => controller.abort()
  }, [id])

  if (isLoading) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Spinner label="Loading product…" />
      </div>
    )
  }

  if (error || !product) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <BackLink />
        {error ? (
          <ErrorBanner title="Could not load product" message={error.message} code={error.code} />
        ) : (
          <EmptyState icon={<PackageX className="size-10" aria-hidden />} title="Product not found" />
        )}
      </div>
    )
  }

  const src = imageUrl(product.image_url)

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <BackLink />

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="card bg-swatch flex items-center justify-center overflow-hidden p-6">
          {src ? (
            <img
              src={src}
              alt={product.name}
              className="max-h-[26rem] w-auto object-contain"
            />
          ) : (
            <div className="text-ink-300 flex h-80 items-center justify-center">
              <ImageOff className="size-12" aria-hidden />
            </div>
          )}
        </div>

        <div>
          <div className="flex flex-wrap items-center gap-2">
            {product.brand && <Badge tone="accent">{product.brand}</Badge>}
            {product.category && <Badge>{product.category}</Badge>}
            {product.in_stock ? (
              <Badge tone="success">In stock</Badge>
            ) : (
              <Badge tone="danger">Out of stock</Badge>
            )}
          </div>

          <h1 className="text-ink-900 mt-3 text-2xl font-semibold">{product.name}</h1>

          <p className="text-ink-900 mt-2 text-2xl font-semibold tabular-nums">
            {formatPrice(product.price, product.currency)}
            <span className="text-ink-400 ml-2 align-middle text-xs font-normal">
              synthetic price
            </span>
          </p>

          {product.description && (
            <p className="text-ink-600 mt-4 text-sm leading-relaxed">{product.description}</p>
          )}

          <dl className="border-ink-200 mt-5 grid grid-cols-2 gap-x-6 gap-y-3 border-t pt-5 text-sm">
            <Field label="Article type" value={product.subcategory} />
            <Field label="Colour" value={product.colour} />
            <Field label="Gender" value={product.gender} />
            <Field label="Usage" value={product.usage} />
            <Field label="Season" value={product.season} />
            <Field label="Year" value={product.year?.toString() ?? null} />
            <Field label="External ID" value={product.external_id} mono />
            <Field label="Product ID" value={product.id} mono />
          </dl>

          <IndexingState product={product} />
        </div>
      </div>

      <section className="mt-10">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-ink-900 text-lg font-semibold">Visually similar products</h2>
          {similar && (
            <span className="text-ink-400 text-xs">
              {similar.mode} search · {similar.timings.total_ms.toFixed(0)}ms
            </span>
          )}
        </div>

        {isLoadingSimilar ? (
          <SkeletonGrid count={4} />
        ) : similar && similar.results.length > 0 ? (
          <>
            {similar.warnings.map((warning) => (
              <p
                key={warning}
                className="mb-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800"
              >
                {warning}
              </p>
            ))}
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
              {similar.results.map((result) => (
                <ProductCard key={result.product.id} result={result} />
              ))}
            </div>
          </>
        ) : (
          <p className="text-ink-500 text-sm">
            No recommendations available. This product may not be indexed yet — run catalogue
            indexing from the <Link to="/catalog" className="text-accent-600 underline">catalogue page</Link>.
          </p>
        )}
      </section>
    </div>
  )
}

function BackLink() {
  return (
    <Link
      to="/"
      className="text-ink-500 hover:text-ink-900 mb-4 inline-flex items-center gap-1.5 text-sm"
    >
      <ArrowLeft className="size-4" aria-hidden />
      Back to search
    </Link>
  )
}

function Field({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string | null | undefined
  mono?: boolean
}) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className={mono ? 'text-ink-700 mt-0.5 truncate font-mono text-xs' : 'text-ink-800 mt-0.5'}>
        {value || '—'}
      </dd>
    </div>
  )
}

function IndexingState({ product }: { product: Product }) {
  return (
    <div className="border-ink-200 mt-5 border-t pt-4">
      <p className="label mb-2">Search index state</p>
      <div className="flex flex-wrap gap-2">
        <Badge tone={product.has_image_vector ? 'success' : 'neutral'}>
          image vector {product.has_image_vector ? '✓' : '—'}
        </Badge>
        <Badge tone={product.has_text_vector ? 'success' : 'neutral'}>
          text vector {product.has_text_vector ? '✓' : '—'}
        </Badge>
        {product.indexed_at ? (
          <Badge tone="neutral">indexed {new Date(product.indexed_at).toLocaleString()}</Badge>
        ) : (
          <Badge tone="warning">not indexed</Badge>
        )}
      </div>
      {product.index_error && (
        <p className="mt-2 text-xs text-amber-700">Indexer note: {product.index_error}</p>
      )}
      {product.search_document && (
        <details className="mt-3">
          <summary className="text-ink-500 hover:text-ink-700 cursor-pointer text-xs">
            Show the text that was embedded
          </summary>
          <p className="text-ink-600 bg-ink-50 mt-1.5 rounded-lg p-2.5 font-mono text-[11px] leading-relaxed">
            {product.search_document}
          </p>
        </details>
      )}
    </div>
  )
}
