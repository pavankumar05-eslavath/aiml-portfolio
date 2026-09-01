/**
 * Catalogue administration: index state, product CRUD, and triggering indexing.
 *
 * Indexing runs synchronously on the backend and can take minutes on a large
 * catalogue, so the button explains that and the client uses an extended timeout
 * rather than appearing to hang.
 */

import { Database, Pencil, Plus, RefreshCw, Trash2, X } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Button, ErrorBanner, Spinner, Stat } from '@/components/ui'
import { cx, formatPrice } from '@/lib/format'
import { useCatalogStats, useDebounced, useProducts } from '@/hooks/useCatalog'
import { ApiError, api, imageUrl } from '@/services/api'
import type { IndexReport, Product, ProductInput } from '@/types'

const PAGE_SIZE = 12

export function CatalogPage() {
  const [refreshKey, setRefreshKey] = useState(0)
  const [page, setPage] = useState(0)
  const [search, setSearch] = useState('')
  const debouncedSearch = useDebounced(search)
  const [editing, setEditing] = useState<Product | 'new' | null>(null)
  const [report, setReport] = useState<IndexReport | null>(null)
  const [isIndexing, setIsIndexing] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)

  const { stats, isLoading: statsLoading } = useCatalogStats(refreshKey)
  const { products, total, isLoading } = useProducts(
    { limit: PAGE_SIZE, offset: page * PAGE_SIZE, search: debouncedSearch },
    refreshKey,
  )

  const refresh = () => setRefreshKey((key) => key + 1)

  const runIndexing = async (force: boolean) => {
    setIsIndexing(true)
    setError(null)
    setReport(null)
    try {
      setReport(await api.indexCatalog({ force }))
      refresh()
    } catch (caught) {
      if (caught instanceof ApiError) setError(caught)
    } finally {
      setIsIndexing(false)
    }
  }

  const remove = async (product: Product) => {
    if (!window.confirm(`Delete "${product.name}"? This also removes its vectors.`)) return
    setError(null)
    try {
      await api.deleteProduct(product.id)
      refresh()
    } catch (caught) {
      if (caught instanceof ApiError) setError(caught)
    }
  }

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-ink-900 text-2xl font-semibold">Catalogue</h1>
          <p className="text-ink-500 mt-1 text-sm">
            Manage products and keep the vector index in step with the database.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            onClick={() => void runIndexing(false)}
            isLoading={isIndexing}
            icon={<RefreshCw className="size-4" aria-hidden />}
          >
            Index pending
          </Button>
          <Button
            variant="secondary"
            onClick={() => void runIndexing(true)}
            isLoading={isIndexing}
            title="Re-embed every product. Needed after changing MODEL_NAME."
          >
            Re-index all
          </Button>
          <Button onClick={() => setEditing('new')} icon={<Plus className="size-4" aria-hidden />}>
            Add product
          </Button>
        </div>
      </header>

      {error && (
        <div className="mb-4">
          <ErrorBanner message={error.message} code={error.code} />
        </div>
      )}

      {isIndexing && (
        <div className="border-accent-200 bg-accent-50 mb-4 rounded-xl border p-4">
          <Spinner label="Embedding products… this runs synchronously and may take several minutes on a large catalogue." />
        </div>
      )}

      {report && <IndexReportCard report={report} onDismiss={() => setReport(null)} />}

      <section className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {statsLoading || !stats ? (
          Array.from({ length: 6 }, (_, index) => (
            <div key={index} className="card bg-ink-50 h-20 animate-pulse" />
          ))
        ) : (
          <>
            <Stat label="Products" value={stats.total_products.toLocaleString()} />
            <Stat
              label="Indexed"
              value={stats.indexed_products.toLocaleString()}
              tone={stats.indexed_products === stats.total_products ? 'success' : 'neutral'}
            />
            <Stat
              label="Pending"
              value={stats.pending_products.toLocaleString()}
              tone={stats.pending_products > 0 ? 'warning' : 'neutral'}
              hint={stats.pending_products > 0 ? 'not yet searchable' : undefined}
            />
            <Stat
              label="Vectors"
              value={stats.vector_points < 0 ? 'n/a' : stats.vector_points.toLocaleString()}
              tone={stats.vector_points < 0 ? 'danger' : 'neutral'}
              hint={stats.vector_points < 0 ? 'Qdrant unreachable' : stats.collection}
            />
            <Stat
              label="With image vec"
              value={stats.with_image_vector.toLocaleString()}
              hint={`${stats.with_text_vector.toLocaleString()} with text vec`}
            />
            <Stat
              label="Embedding dim"
              value={stats.embedding_dim ?? '—'}
              hint={stats.model_name.split('/').pop()}
            />
          </>
        )}
      </section>

      <div className="card overflow-hidden">
        <div className="border-ink-200 flex flex-wrap items-center justify-between gap-3 border-b p-3">
          <input
            type="search"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value)
              setPage(0)
            }}
            placeholder="Filter by name, brand or category…"
            className="border-ink-200 h-9 w-full max-w-xs rounded-lg border px-3 text-sm"
          />
          <p className="text-ink-500 text-sm">
            {total.toLocaleString()} product{total === 1 ? '' : 's'}
          </p>
        </div>

        {isLoading ? (
          <div className="p-6">
            <Spinner label="Loading products…" />
          </div>
        ) : products.length === 0 ? (
          <div className="text-ink-500 p-8 text-center text-sm">
            <Database className="text-ink-300 mx-auto mb-3 size-8" aria-hidden />
            No products found.{' '}
            {!search && (
              <>
                Load the sample catalogue with{' '}
                <code className="bg-ink-100 rounded px-1 py-0.5 text-xs">
                  python scripts/index_catalog.py
                </code>
              </>
            )}
          </div>
        ) : (
          <div className="divide-ink-100 divide-y">
            {products.map((product) => (
              <ProductRow
                key={product.id}
                product={product}
                onEdit={() => setEditing(product)}
                onDelete={() => void remove(product)}
              />
            ))}
          </div>
        )}

        {pageCount > 1 && (
          <div className="border-ink-200 flex items-center justify-between border-t p-3">
            <Button
              variant="secondary"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((current) => Math.max(0, current - 1))}
            >
              Previous
            </Button>
            <span className="text-ink-500 text-sm tabular-nums">
              Page {page + 1} of {pageCount}
            </span>
            <Button
              variant="secondary"
              size="sm"
              disabled={page + 1 >= pageCount}
              onClick={() => setPage((current) => current + 1)}
            >
              Next
            </Button>
          </div>
        )}
      </div>

      {editing && (
        <ProductDialog
          product={editing === 'new' ? null : editing}
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null)
            refresh()
          }}
        />
      )}
    </div>
  )
}

function ProductRow({
  product,
  onEdit,
  onDelete,
}: {
  product: Product
  onEdit: () => void
  onDelete: () => void
}) {
  const src = imageUrl(product.image_url)
  return (
    <div className="hover:bg-ink-50/60 flex items-center gap-3 p-3 transition-colors">
      <div className="bg-swatch size-12 shrink-0 overflow-hidden rounded-lg">
        {src && (
          <img src={src} alt="" loading="lazy" className="size-full object-contain p-1" />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <Link
          to={`/product/${product.id}`}
          className="text-ink-900 hover:text-accent-700 block truncate text-sm font-medium"
        >
          {product.name}
        </Link>
        <p className="text-ink-500 truncate text-xs">
          {[product.brand, product.subcategory, product.colour].filter(Boolean).join(' · ') || '—'}
        </p>
      </div>
      <span className="text-ink-800 hidden w-20 shrink-0 text-right text-sm tabular-nums sm:block">
        {formatPrice(product.price, product.currency)}
      </span>
      <div className="hidden w-24 shrink-0 md:block">
        {product.indexed_at ? (
          <Badge tone="success">indexed</Badge>
        ) : (
          <Badge tone="warning">pending</Badge>
        )}
      </div>
      <div className="flex shrink-0 gap-1">
        <button
          type="button"
          onClick={onEdit}
          aria-label={`Edit ${product.name}`}
          className="text-ink-400 hover:bg-ink-100 hover:text-ink-700 rounded-md p-1.5"
        >
          <Pencil className="size-4" aria-hidden />
        </button>
        <button
          type="button"
          onClick={onDelete}
          aria-label={`Delete ${product.name}`}
          className="text-ink-400 rounded-md p-1.5 hover:bg-red-50 hover:text-red-600"
        >
          <Trash2 className="size-4" aria-hidden />
        </button>
      </div>
    </div>
  )
}

function IndexReportCard({
  report,
  onDismiss,
}: {
  report: IndexReport
  onDismiss: () => void
}) {
  return (
    <div className="mb-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-emerald-900">Indexing finished</p>
          <p className="mt-1 text-sm text-emerald-800">
            {report.embedded} embedded · {report.skipped_unchanged} skipped (unchanged) ·{' '}
            {report.failed} failed · {report.image_vectors} image vectors ·{' '}
            {report.text_vectors} text vectors · {(report.duration_ms / 1000).toFixed(1)}s
          </p>
          {report.failures.length > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer text-xs text-emerald-700">
                {report.failures.length} product(s) needed attention
              </summary>
              <ul className="mt-1.5 space-y-0.5 text-xs text-emerald-800">
                {report.failures.slice(0, 10).map((failure) => (
                  <li key={failure.product_id} className="truncate">
                    <span className="font-medium">{failure.name ?? failure.product_id}</span>:{' '}
                    {failure.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className="rounded-md p-1 text-emerald-600 hover:bg-emerald-100"
        >
          <X className="size-4" aria-hidden />
        </button>
      </div>
    </div>
  )
}

const FIELDS: Array<{
  key: keyof ProductInput
  label: string
  type?: 'text' | 'number'
  required?: boolean
}> = [
  { key: 'name', label: 'Name', required: true },
  { key: 'brand', label: 'Brand' },
  { key: 'category', label: 'Category' },
  { key: 'subcategory', label: 'Article type' },
  { key: 'colour', label: 'Colour' },
  { key: 'gender', label: 'Gender' },
  { key: 'usage', label: 'Usage' },
  { key: 'price', label: 'Price', type: 'number' },
  { key: 'image_url', label: 'Image URL or path' },
  { key: 'external_id', label: 'External ID' },
]

function ProductDialog({
  product,
  onClose,
  onSaved,
}: {
  product: Product | null
  onClose: () => void
  onSaved: () => void
}) {
  const [form, setForm] = useState<ProductInput>(() => ({
    name: product?.name ?? '',
    brand: product?.brand ?? '',
    category: product?.category ?? '',
    subcategory: product?.subcategory ?? '',
    colour: product?.colour ?? '',
    gender: product?.gender ?? '',
    usage: product?.usage ?? '',
    price: product?.price ?? null,
    image_url: product?.image_url ?? '',
    external_id: product?.external_id ?? '',
    description: product?.description ?? '',
    in_stock: product?.in_stock ?? true,
  }))
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)

  const save = async () => {
    setIsSaving(true)
    setError(null)
    // Blank strings must become null, not "", so the backend treats them as unset.
    const payload: ProductInput = Object.fromEntries(
      Object.entries(form).map(([key, value]) => [
        key,
        typeof value === 'string' && value.trim() === '' ? null : value,
      ]),
    ) as ProductInput
    payload.name = form.name.trim()

    try {
      if (product) await api.updateProduct(product.id, payload)
      else await api.createProduct(payload)
      onSaved()
    } catch (caught) {
      if (caught instanceof ApiError) setError(caught)
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={product ? 'Edit product' : 'Add product'}
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/40 p-4"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-white p-5 shadow-xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-ink-900 text-lg font-semibold">
            {product ? 'Edit product' : 'Add product'}
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="text-ink-400 hover:bg-ink-100 rounded-md p-1"
          >
            <X className="size-5" aria-hidden />
          </button>
        </div>

        {error && (
          <div className="mb-4">
            <ErrorBanner title="Could not save" message={error.message} code={error.code} />
          </div>
        )}

        <div className="grid gap-3 sm:grid-cols-2">
          {FIELDS.map(({ key, label, type = 'text', required }) => (
            <label key={key} className={cx(key === 'name' && 'sm:col-span-2')}>
              <span className="label mb-1 block">
                {label}
                {required && <span className="text-red-500"> *</span>}
              </span>
              <input
                type={type}
                value={(form[key] as string | number | null) ?? ''}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    [key]:
                      type === 'number'
                        ? event.target.value === ''
                          ? null
                          : Number(event.target.value)
                        : event.target.value,
                  }))
                }
                className="border-ink-200 h-9 w-full rounded-lg border px-2.5 text-sm"
              />
            </label>
          ))}
          <label className="sm:col-span-2">
            <span className="label mb-1 block">Description</span>
            <textarea
              rows={3}
              value={form.description ?? ''}
              onChange={(event) =>
                setForm((current) => ({ ...current, description: event.target.value }))
              }
              className="border-ink-200 w-full rounded-lg border px-2.5 py-2 text-sm"
            />
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={form.in_stock ?? true}
              onChange={(event) =>
                setForm((current) => ({ ...current, in_stock: event.target.checked }))
              }
              className="accent-accent-600 size-4"
            />
            <span className="text-ink-700">In stock</span>
          </label>
        </div>

        <p className="text-ink-400 mt-4 text-xs">
          Saving does not embed the product. Run <strong>Index pending</strong> afterwards to make
          it searchable — editing a field that feeds the embedding marks it for re-indexing
          automatically.
        </p>

        <div className="mt-5 flex justify-end gap-2">
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={() => void save()} isLoading={isSaving} disabled={!form.name.trim()}>
            {product ? 'Save changes' : 'Create product'}
          </Button>
        </div>
      </div>
    </div>
  )
}
