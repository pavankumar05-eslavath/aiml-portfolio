/**
 * Search state and execution.
 *
 * Owns the request lifecycle so components stay declarative:
 *
 * - Every new search aborts the previous one. Without this, a slow multimodal
 *   request can resolve after a fast text request and overwrite fresher results.
 * - `loadMore` appends a page rather than replacing, and is guarded so it cannot
 *   run concurrently with itself.
 * - Aborted requests never surface as errors, only real failures do.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api } from '@/services/api'
import {
  type FusionOverrides,
  type SearchFilters,
  type SearchMode,
  type SearchResponse,
  type SearchResult,
  type SortOption,
  emptyFilters,
} from '@/types'

export interface SearchRequestState {
  mode: SearchMode
  query: string
  file: File | null
  filters: SearchFilters
  sort: SortOption
  fusion: FusionOverrides
  topK: number
}

export const initialRequest = (): SearchRequestState => ({
  mode: 'text',
  query: '',
  file: null,
  filters: emptyFilters(),
  sort: 'relevance',
  fusion: {},
  topK: 24,
})

interface UseSearchResult {
  results: SearchResult[]
  response: SearchResponse | null
  isLoading: boolean
  isLoadingMore: boolean
  error: ApiError | null
  hasSearched: boolean
  canLoadMore: boolean
  runSearch: (request: SearchRequestState) => Promise<void>
  loadMore: () => Promise<void>
  reset: () => void
}

export function useSearch(): UseSearchResult {
  const [results, setResults] = useState<SearchResult[]>([])
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const [hasSearched, setHasSearched] = useState(false)

  const controller = useRef<AbortController | null>(null)
  const lastRequest = useRef<SearchRequestState | null>(null)
  const offset = useRef(0)

  useEffect(() => () => controller.current?.abort(), [])

  const execute = useCallback(
    async (request: SearchRequestState, searchOffset: number, signal: AbortSignal) => {
      const options = {
        top_k: request.topK,
        offset: searchOffset,
        filters: request.filters,
        sort: request.sort,
        explain: true,
        ...(Object.keys(request.fusion).length > 0 ? { fusion: request.fusion } : {}),
      }

      switch (request.mode) {
        case 'text':
          return api.searchText(request.query, options, signal)
        case 'image':
          if (!request.file) throw new ApiError('Select an image first.', 422, 'empty_query')
          return api.searchImage(request.file, options, signal)
        case 'multimodal':
          return api.searchMultimodal(request.file, request.query, options, signal)
      }
    },
    [],
  )

  const runSearch = useCallback(
    async (request: SearchRequestState) => {
      controller.current?.abort()
      const next = new AbortController()
      controller.current = next

      lastRequest.current = request
      offset.current = 0
      setIsLoading(true)
      setError(null)
      setHasSearched(true)

      try {
        const data = await execute(request, 0, next.signal)
        if (next.signal.aborted) return
        setResponse(data)
        setResults(data.results)
      } catch (caught) {
        if (next.signal.aborted) return
        setResults([])
        setResponse(null)
        setError(
          caught instanceof ApiError
            ? caught
            : new ApiError('Search failed unexpectedly.', 0, 'unknown'),
        )
      } finally {
        if (!next.signal.aborted) setIsLoading(false)
      }
    },
    [execute],
  )

  const loadMore = useCallback(async () => {
    const request = lastRequest.current
    if (!request || !response || isLoadingMore || isLoading) return
    if (results.length >= response.total_candidates) return

    const nextOffset = offset.current + request.topK
    const next = new AbortController()
    setIsLoadingMore(true)

    try {
      const data = await execute(request, nextOffset, next.signal)
      if (next.signal.aborted) return
      offset.current = nextOffset
      // De-duplicate defensively: a concurrent catalogue change could otherwise
      // produce a repeated product and a duplicate React key.
      setResults((current) => {
        const seen = new Set(current.map((r) => r.product.id))
        return [...current, ...data.results.filter((r) => !seen.has(r.product.id))]
      })
      setResponse((current) => (current ? { ...current, warnings: data.warnings } : data))
    } catch (caught) {
      if (caught instanceof ApiError) setError(caught)
    } finally {
      setIsLoadingMore(false)
    }
  }, [execute, isLoading, isLoadingMore, response, results.length])

  const reset = useCallback(() => {
    controller.current?.abort()
    setResults([])
    setResponse(null)
    setError(null)
    setHasSearched(false)
    offset.current = 0
    lastRequest.current = null
  }, [])

  return {
    results,
    response,
    isLoading,
    isLoadingMore,
    error,
    hasSearched,
    canLoadMore: Boolean(response && results.length < response.total_candidates),
    runSearch,
    loadMore,
    reset,
  }
}
