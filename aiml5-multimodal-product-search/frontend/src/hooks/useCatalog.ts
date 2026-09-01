/**
 * Catalogue data hooks: health, facets, stats and product listings.
 *
 * Each hook aborts its in-flight request on unmount so a navigation away does not
 * trigger a state update on an unmounted component.
 *
 * Note on `set-state-in-effect`: the React Compiler lint flags the
 * `setIsLoading(true)` at the start of each fetch effect. That is expected here.
 * These hooks synchronise with an external system (the API), and the loading flag
 * must reset whenever the query parameters change. The alternative is a
 * data-fetching library such as TanStack Query, which was deliberately not added
 * for a handful of endpoints. The warning is advisory, not a defect.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api } from '@/services/api'
import type {
  CatalogFacets,
  CatalogStats,
  HealthResponse,
  Product,
  SortOption,
} from '@/types'

/** Poll `/health` until the model has finished loading, then stop. */
export function useHealth(pollMs = 4000) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<ApiError | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined

    const poll = async () => {
      try {
        const data = await api.health(controller.signal)
        if (controller.signal.aborted) return
        setHealth(data)
        setError(null)
        // The model loads asynchronously at startup; keep polling until ready so
        // the UI can stop showing "warming up" without a manual refresh.
        if (!data.model_loaded) timer = setTimeout(poll, pollMs)
      } catch (caught) {
        if (controller.signal.aborted) return
        if (caught instanceof ApiError) setError(caught)
        timer = setTimeout(poll, pollMs * 2)
      }
    }

    void poll()
    return () => {
      controller.abort()
      if (timer) clearTimeout(timer)
    }
  }, [pollMs])

  return { health, error, isReady: health?.model_loaded === true }
}

export function useFacets() {
  const [facets, setFacets] = useState<CatalogFacets | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const data = await api.facets(signal)
      if (!signal?.aborted) setFacets(data)
    } catch {
      // Facets are decoration: a failure must not block searching, so the filter
      // panel simply renders empty.
    } finally {
      if (!signal?.aborted) setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void load(controller.signal)
    return () => controller.abort()
  }, [load])

  return { facets, isLoading, reload: () => load() }
}

export function useCatalogStats(refreshKey = 0) {
  const [stats, setStats] = useState<CatalogStats | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    setIsLoading(true)
    api
      .stats(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setStats(data)
          setError(null)
        }
      })
      .catch((caught) => {
        if (!controller.signal.aborted && caught instanceof ApiError) setError(caught)
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoading(false)
      })
    return () => controller.abort()
  }, [refreshKey])

  return { stats, error, isLoading }
}

export interface ProductListParams {
  limit: number
  offset: number
  search?: string
  sort?: SortOption
}

export function useProducts(params: ProductListParams, refreshKey = 0) {
  const [products, setProducts] = useState<Product[]>([])
  const [total, setTotal] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<ApiError | null>(null)
  const controller = useRef<AbortController | null>(null)

  const { limit, offset, search, sort } = params

  useEffect(() => {
    controller.current?.abort()
    const next = new AbortController()
    controller.current = next
    setIsLoading(true)

    api
      .listProducts({ limit, offset, search: search || undefined, sort }, next.signal)
      .then((page) => {
        if (next.signal.aborted) return
        setProducts(page.items)
        setTotal(page.total)
        setError(null)
      })
      .catch((caught) => {
        if (next.signal.aborted) return
        if (caught instanceof ApiError) setError(caught)
      })
      .finally(() => {
        if (!next.signal.aborted) setIsLoading(false)
      })

    return () => next.abort()
  }, [limit, offset, search, sort, refreshKey])

  return { products, total, isLoading, error }
}

/** Debounce a rapidly changing value, e.g. a search-as-you-type field. */
export function useDebounced<T>(value: T, delayMs = 350): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(timer)
  }, [value, delayMs])
  return debounced
}
