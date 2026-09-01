/**
 * The search composer: mode selector, text box, image dropzone and the ranking
 * controls.
 *
 * The ranking controls are exposed in the UI rather than hidden in config because
 * the experiments showed the optimal image/text weighting is *query-dependent*
 * (contradiction-style queries want ~0.1, agreement-style want ~0.5). Letting the
 * user move the slider and watch the ranking change is the clearest way to show
 * what multimodal fusion actually does.
 */

import { ChevronDown, Image as ImageIcon, Layers, Search, Sparkles, Type } from 'lucide-react'
import { useState } from 'react'
import { ImageDropzone } from '@/components/search/ImageDropzone'
import { Badge, Button } from '@/components/ui'
import { cx } from '@/lib/format'
import type { SearchRequestState } from '@/hooks/useSearch'
import type { FusionStrategy, SearchMode } from '@/types'

const MODES: Array<{
  id: SearchMode
  label: string
  icon: typeof Type
  hint: string
}> = [
  { id: 'text', label: 'Text', icon: Type, hint: 'Describe what you want' },
  { id: 'image', label: 'Image', icon: ImageIcon, hint: 'Find visually similar products' },
  { id: 'multimodal', label: 'Image + Text', icon: Layers, hint: 'An image, refined by words' },
]

const EXAMPLE_QUERIES = [
  'black running shoes suitable for daily workouts',
  'smart leather shoes to wear to the office',
  'dark glasses to keep the sun out of my eyes',
  'roomy black rucksack for carrying books',
  'gold bangles to wear on the wrist',
]

interface Props {
  request: SearchRequestState
  onChange: (next: Partial<SearchRequestState>) => void
  onSubmit: () => void
  isLoading: boolean
  isReady: boolean
}

export function SearchPanel({ request, onChange, onSubmit, isLoading, isReady }: Props) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const needsImage = request.mode === 'image' && !request.file
  const needsText = request.mode === 'text' && !request.query.trim()
  const needsSomething =
    request.mode === 'multimodal' && !request.file && !request.query.trim()
  const cannotSubmit = !isReady || isLoading || needsImage || needsText || needsSomething

  const submit = () => {
    if (!cannotSubmit) onSubmit()
  }

  return (
    <section className="card p-4 sm:p-5" aria-label="Search">
      <div
        role="tablist"
        aria-label="Search mode"
        className="bg-ink-100 mb-4 inline-flex rounded-lg p-1"
      >
        {MODES.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            role="tab"
            type="button"
            aria-selected={request.mode === id}
            onClick={() => onChange({ mode: id })}
            className={cx(
              'inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-all',
              request.mode === id
                ? 'text-ink-900 bg-white shadow-sm'
                : 'text-ink-500 hover:text-ink-700',
            )}
          >
            <Icon className="size-4" aria-hidden />
            {label}
          </button>
        ))}
      </div>

      <div className="space-y-4">
        {request.mode !== 'text' && (
          <div>
            <label className="label mb-1.5 block">
              Query image {request.mode === 'multimodal' && '(optional)'}
            </label>
            <ImageDropzone
              file={request.file}
              onChange={(file) => onChange({ file })}
              disabled={isLoading}
            />
          </div>
        )}

        {request.mode !== 'image' && (
          <div>
            <label htmlFor="search-query" className="label mb-1.5 block">
              {request.mode === 'multimodal' ? 'Refine with words' : 'What are you looking for?'}
            </label>
            <div className="relative">
              <Search
                className="text-ink-400 pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2"
                aria-hidden
              />
              <input
                id="search-query"
                type="search"
                value={request.query}
                onChange={(event) => onChange({ query: event.target.value })}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') submit()
                }}
                placeholder={
                  request.mode === 'multimodal'
                    ? 'e.g. "the same style but in black"'
                    : 'e.g. "black running shoes for daily workouts"'
                }
                disabled={isLoading}
                className="border-ink-200 focus:border-accent-500 focus:ring-accent-500/20 h-11 w-full rounded-lg border pr-3 pl-9 text-sm transition-colors focus:ring-4 disabled:opacity-60"
              />
            </div>

            {request.mode === 'text' && !request.query && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {EXAMPLE_QUERIES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => onChange({ query: example })}
                    className="text-ink-600 border-ink-200 hover:border-accent-300 hover:bg-accent-50 hover:text-accent-700 rounded-full border px-2.5 py-1 text-xs transition-colors"
                  >
                    {example}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <Button
            onClick={submit}
            disabled={cannotSubmit}
            isLoading={isLoading}
            size="lg"
            icon={<Sparkles className="size-4" aria-hidden />}
          >
            Search
          </Button>

          <button
            type="button"
            onClick={() => setShowAdvanced((open) => !open)}
            aria-expanded={showAdvanced}
            className="text-ink-500 hover:text-ink-800 inline-flex items-center gap-1 text-xs font-medium"
          >
            <ChevronDown
              className={cx('size-3.5 transition-transform', showAdvanced && 'rotate-180')}
              aria-hidden
            />
            Ranking controls
          </button>

          {!isReady && (
            <Badge tone="warning">Model is still loading — search will 503 until ready</Badge>
          )}
          {needsImage && <span className="text-ink-400 text-xs">Add an image to search</span>}
          {needsText && <span className="text-ink-400 text-xs">Type a query to search</span>}
          {needsSomething && (
            <span className="text-ink-400 text-xs">Add an image or some text</span>
          )}
        </div>

        {showAdvanced && (
          <RankingControls request={request} onChange={onChange} />
        )}
      </div>
    </section>
  )
}

function RankingControls({
  request,
  onChange,
}: {
  request: SearchRequestState
  onChange: (next: Partial<SearchRequestState>) => void
}) {
  const fusion = request.fusion
  const imageWeight = fusion.image_weight ?? 0.2
  const crossModal = fusion.cross_modal_weight ?? 0.3
  const strategy: FusionStrategy = fusion.strategy ?? 'weighted_sum'

  const setFusion = (patch: Partial<typeof fusion>) =>
    onChange({ fusion: { ...fusion, ...patch } })

  return (
    <div className="border-ink-200 bg-ink-50/60 space-y-4 rounded-xl border p-4">
      <div>
        <div className="mb-1 flex items-baseline justify-between">
          <label htmlFor="image-weight" className="label">
            Image vs text weight
          </label>
          <span className="text-ink-600 font-mono text-xs">
            image {imageWeight.toFixed(2)} / text {(1 - imageWeight).toFixed(2)}
          </span>
        </div>
        <input
          id="image-weight"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={imageWeight}
          onChange={(event) => {
            const value = Number(event.target.value)
            setFusion({ image_weight: value, text_weight: 1 - value })
          }}
          className="accent-accent-600 w-full"
        />
        <p className="text-ink-400 mt-1 text-xs">
          Only applies when a query has both an image and text. Measured optima differ by intent:
          ~0.1 when the text contradicts the image, ~0.5 when they agree.
        </p>
      </div>

      <div>
        <div className="mb-1 flex items-baseline justify-between">
          <label htmlFor="cross-modal" className="label">
            Cross-modal weight
          </label>
          <span className="text-ink-600 font-mono text-xs">{crossModal.toFixed(2)}</span>
        </div>
        <input
          id="cross-modal"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={crossModal}
          onChange={(event) => setFusion({ cross_modal_weight: Number(event.target.value) })}
          className="accent-accent-600 w-full"
        />
        <p className="text-ink-400 mt-1 text-xs">
          Share given to searching the <em>other</em> modality&apos;s vector, e.g. a text query
          against product images. 0 disables cross-modal retrieval.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label htmlFor="strategy" className="label mb-1 block">
            Fusion strategy
          </label>
          <select
            id="strategy"
            value={strategy}
            onChange={(event) =>
              setFusion({ strategy: event.target.value as FusionStrategy })
            }
            className="border-ink-200 h-9 w-full rounded-lg border bg-white px-2 text-sm"
          >
            <option value="weighted_sum">weighted_sum — normalise, weight, add</option>
            <option value="rrf">rrf — reciprocal rank fusion</option>
            <option value="embedding_fusion">embedding_fusion — blend query vectors</option>
          </select>
        </div>
        <div>
          <label htmlFor="top-k" className="label mb-1 block">
            Results per page
          </label>
          <select
            id="top-k"
            value={request.topK}
            onChange={(event) => onChange({ topK: Number(event.target.value) })}
            className="border-ink-200 h-9 w-full rounded-lg border bg-white px-2 text-sm"
          >
            {[12, 24, 48, 96].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
      </div>

      {strategy === 'embedding_fusion' && (
        <p className="rounded-lg bg-amber-50 p-2.5 text-xs text-amber-800">
          <strong>Note:</strong> embedding fusion blends the two query vectors before retrieval,
          so results can no longer be attributed to a modality — the per-result image/text
          similarity breakdown disappears. That trade-off is the reason score-level fusion is
          the default.
        </p>
      )}

      {Object.keys(fusion).length > 0 && (
        <Button variant="ghost" size="sm" onClick={() => onChange({ fusion: {} })}>
          Reset to server defaults
        </Button>
      )}
    </div>
  )
}
