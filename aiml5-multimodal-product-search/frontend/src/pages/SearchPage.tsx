/** Main search page: composer, filters and ranked results. */

import { useCallback, useState } from 'react'
import { FilterPanel } from '@/components/filters/FilterPanel'
import { ResultsSection } from '@/components/results/ResultsSection'
import { SearchPanel } from '@/components/search/SearchPanel'
import { useFacets, useHealth } from '@/hooks/useCatalog'
import { initialRequest, useSearch, type SearchRequestState } from '@/hooks/useSearch'
import type { SearchFilters, SortOption } from '@/types'

export function SearchPage() {
  const [request, setRequest] = useState<SearchRequestState>(initialRequest)
  const { facets } = useFacets()
  const { isReady } = useHealth()
  const {
    results,
    response,
    isLoading,
    isLoadingMore,
    error,
    hasSearched,
    canLoadMore,
    runSearch,
    loadMore,
  } = useSearch()

  const update = useCallback(
    (patch: Partial<SearchRequestState>) => setRequest((current) => ({ ...current, ...patch })),
    [],
  )

  const submit = useCallback(() => void runSearch(request), [request, runSearch])

  /**
   * Re-run immediately when filters or sort change, but only if a search has
   * already happened. Changing a filter before searching should not fire a query
   * with no criteria.
   */
  const applyAndRerun = useCallback(
    (patch: Partial<SearchRequestState>) => {
      const next = { ...request, ...patch }
      setRequest(next)
      if (hasSearched) void runSearch(next)
    },
    [hasSearched, request, runSearch],
  )

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <SearchPanel
        request={request}
        onChange={update}
        onSubmit={submit}
        isLoading={isLoading}
        isReady={isReady}
      />

      <div className="mt-6 grid gap-6 lg:grid-cols-[16rem_1fr]">
        <div className="hidden lg:block">
          <FilterPanel
            facets={facets}
            filters={request.filters}
            onChange={(filters: SearchFilters) => update({ filters })}
            onApply={() => applyAndRerun({})}
          />
        </div>

        <div className="min-w-0">
          <ResultsSection
            results={results}
            response={response}
            isLoading={isLoading}
            isLoadingMore={isLoadingMore}
            error={error}
            hasSearched={hasSearched}
            canLoadMore={canLoadMore}
            sort={request.sort}
            onSortChange={(sort: SortOption) => applyAndRerun({ sort })}
            onLoadMore={() => void loadMore()}
            onRetry={submit}
          />
        </div>
      </div>
    </div>
  )
}
