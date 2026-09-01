/**
 * Typed API client.
 *
 * One place that knows how to talk to the backend. Every call funnels through
 * `request`, which unwraps the backend's `{error: {code, message}}` envelope into
 * a typed `ApiError`, so components can show the server's actual message rather
 * than a generic failure. Requests carry an `AbortSignal` where a newer request
 * can supersede an older one - without that, a slow search can land after a fast
 * one and overwrite fresher results.
 */

import type {
  CatalogFacets,
  CatalogStats,
  HealthResponse,
  IndexReport,
  Page,
  Product,
  ProductInput,
  SearchOptions,
  SearchResponse,
  SortOption,
} from '@/types'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/$/, '')

/** Timeout for ordinary requests. Indexing uses its own, much longer, budget. */
const DEFAULT_TIMEOUT_MS = 60_000
const INDEX_TIMEOUT_MS = 30 * 60_000

/**
 * An error carrying the backend's machine-readable code.
 *
 * Fields are declared and assigned explicitly rather than via constructor
 * parameter properties, which `erasableSyntaxOnly` disallows.
 */
export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly details?: unknown

  constructor(message: string, status: number, code = 'unknown', details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }

  /** True when retrying later is plausible (model loading, dependency down). */
  get isRetryable(): boolean {
    return this.status === 503 || this.status === 429
  }
}

interface RequestOptions extends Omit<RequestInit, 'signal'> {
  signal?: AbortSignal
  timeoutMs?: number
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal, ...init } = options

  // Combine the caller's signal with a timeout so either can cancel the request.
  const timeoutController = new AbortController()
  const timer = setTimeout(() => timeoutController.abort(), timeoutMs)
  const signals = [timeoutController.signal, ...(signal ? [signal] : [])]

  try {
    const response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      signal: AbortSignal.any(signals),
    })

    if (response.status === 204) return undefined as T

    const contentType = response.headers.get('content-type') ?? ''
    if (!contentType.includes('application/json')) {
      if (!response.ok) {
        throw new ApiError(
          `Request failed with status ${response.status}.`,
          response.status,
          'non_json_response',
        )
      }
      return undefined as T
    }

    const body = await response.json()

    if (!response.ok) {
      const error = (body as { error?: { code?: string; message?: string; details?: unknown } })
        .error
      throw new ApiError(
        error?.message ?? `Request failed with status ${response.status}.`,
        response.status,
        error?.code ?? 'unknown',
        error?.details,
      )
    }

    return body as T
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error instanceof DOMException && error.name === 'AbortError') {
      // A caller-initiated abort is normal control flow; a timeout is not.
      if (signal?.aborted) throw error
      throw new ApiError(
        `The request timed out after ${Math.round(timeoutMs / 1000)}s.`,
        408,
        'timeout',
      )
    }
    throw new ApiError(
      'Could not reach the API. Is the backend running on port 8000?',
      0,
      'network_error',
    )
  } finally {
    clearTimeout(timer)
  }
}

const json = (body: unknown): RequestOptions => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

/** Options sent alongside a multipart search upload. */
const searchOptionsBlob = (options: SearchOptions): string =>
  JSON.stringify({
    top_k: options.top_k,
    offset: options.offset,
    filters: options.filters,
    sort: options.sort,
    explain: options.explain,
    ...(options.fusion ? { fusion: options.fusion } : {}),
  })

export const api = {
  health: (signal?: AbortSignal) =>
    request<HealthResponse>('/health', { signal, timeoutMs: 10_000 }),

  searchText: (query: string, options: SearchOptions, signal?: AbortSignal) =>
    request<SearchResponse>('/search/text', {
      ...json({ query, ...JSON.parse(searchOptionsBlob(options)) }),
      signal,
    }),

  searchImage: (file: File, options: SearchOptions, signal?: AbortSignal) => {
    const form = new FormData()
    form.append('file', file)
    form.append('options', searchOptionsBlob(options))
    return request<SearchResponse>('/search/image', { method: 'POST', body: form, signal })
  },

  searchMultimodal: (
    file: File | null,
    query: string,
    options: SearchOptions,
    signal?: AbortSignal,
  ) => {
    const form = new FormData()
    if (file) form.append('file', file)
    if (query) form.append('query', query)
    form.append('options', searchOptionsBlob(options))
    return request<SearchResponse>('/search/multimodal', { method: 'POST', body: form, signal })
  },

  searchSimilar: (productId: string, options: SearchOptions, signal?: AbortSignal) =>
    request<SearchResponse>(`/search/similar/${encodeURIComponent(productId)}`, {
      ...json(JSON.parse(searchOptionsBlob(options))),
      signal,
    }),

  listProducts: (
    params: {
      limit?: number
      offset?: number
      search?: string
      category?: string[]
      brand?: string[]
      colour?: string[]
      min_price?: number | null
      max_price?: number | null
      in_stock_only?: boolean
      sort?: SortOption
    } = {},
    signal?: AbortSignal,
  ) => {
    const query = new URLSearchParams()
    const append = (key: string, value: unknown) => {
      if (value === undefined || value === null || value === '' || value === false) return
      if (Array.isArray(value)) value.forEach((v) => query.append(key, String(v)))
      else query.append(key, String(value))
    }
    Object.entries(params).forEach(([key, value]) => append(key, value))
    const suffix = query.toString()
    return request<Page<Product>>(`/products${suffix ? `?${suffix}` : ''}`, { signal })
  },

  getProduct: (id: string, signal?: AbortSignal) =>
    request<Product>(`/products/${encodeURIComponent(id)}`, { signal }),

  createProduct: (payload: ProductInput, signal?: AbortSignal) =>
    request<Product>('/products', { ...json(payload), signal }),

  updateProduct: (id: string, payload: ProductInput, signal?: AbortSignal) =>
    request<Product>(`/products/${encodeURIComponent(id)}`, {
      ...json(payload),
      method: 'PUT',
      signal,
    }),

  deleteProduct: (id: string, signal?: AbortSignal) =>
    request<void>(`/products/${encodeURIComponent(id)}`, { method: 'DELETE', signal }),

  facets: (signal?: AbortSignal) => request<CatalogFacets>('/products/facets', { signal }),

  stats: (signal?: AbortSignal) => request<CatalogStats>('/catalog/stats', { signal }),

  indexCatalog: (
    body: { product_ids?: string[]; force?: boolean; limit?: number } = {},
    signal?: AbortSignal,
  ) =>
    request<IndexReport>('/catalog/index', {
      ...json(body),
      signal,
      // Indexing is synchronous and can legitimately run for minutes.
      timeoutMs: INDEX_TIMEOUT_MS,
    }),
}

/**
 * Resolve a product's stored image reference to a displayable URL.
 *
 * Absolute URLs pass through; relative paths are served by the backend's media
 * endpoint, which is what the local dataset uses.
 */
export const imageUrl = (reference: string | null | undefined): string | null => {
  if (!reference) return null
  if (/^https?:\/\//i.test(reference)) return reference
  return `${BASE_URL}/media/${reference.replace(/^\/+/, '')}`
}
