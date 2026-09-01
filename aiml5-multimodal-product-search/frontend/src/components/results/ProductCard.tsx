/**
 * A product result card with its similarity score and, on demand, the per-channel
 * breakdown of how that score was produced.
 */

import { ChevronDown, ImageOff } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useState } from 'react'
import { Badge } from '@/components/ui'
import { cx, formatPrice } from '@/lib/format'
import { imageUrl } from '@/services/api'
import { channelLabel, type SearchResult } from '@/types'

export function ProductCard({
  result,
  showBreakdown = true,
}: {
  result: SearchResult
  showBreakdown?: boolean
}) {
  const [isExpanded, setIsExpanded] = useState(false)
  const { product, breakdown, score, rank } = result
  const src = imageUrl(product.image_url)

  return (
    <article className="card group flex flex-col overflow-hidden transition-shadow hover:shadow-md">
      <Link
        to={`/product/${product.id}`}
        className="bg-swatch relative block aspect-square overflow-hidden"
      >
        {src ? (
          <img
            src={src}
            alt={product.name}
            loading="lazy"
            className="size-full object-contain p-3 transition-transform duration-300 group-hover:scale-105"
          />
        ) : (
          <div className="text-ink-300 flex size-full items-center justify-center">
            <ImageOff className="size-8" aria-hidden />
          </div>
        )}
        <span className="text-ink-600 absolute top-2 left-2 rounded-md bg-white/90 px-1.5 py-0.5 text-[11px] font-semibold tabular-nums shadow-sm">
          #{rank}
        </span>
        {!product.in_stock && (
          <span className="absolute top-2 right-2 rounded-md bg-ink-900/80 px-1.5 py-0.5 text-[11px] font-medium text-white">
            Out of stock
          </span>
        )}
      </Link>

      <div className="flex flex-1 flex-col p-3">
        <Link to={`/product/${product.id}`} className="min-w-0">
          <h3
            title={product.name}
            className="text-ink-900 group-hover:text-accent-700 line-clamp-2 text-sm font-medium transition-colors"
          >
            {product.name}
          </h3>
        </Link>

        <div className="text-ink-500 mt-1 flex flex-wrap items-center gap-x-1.5 text-xs">
          {product.brand && <span className="font-medium">{product.brand}</span>}
          {product.brand && product.subcategory && <span aria-hidden>·</span>}
          {product.subcategory && <span>{product.subcategory}</span>}
        </div>

        <div className="mt-2 flex items-center justify-between gap-2">
          <span className="text-ink-900 text-sm font-semibold tabular-nums">
            {formatPrice(product.price, product.currency)}
          </span>
          <ScorePill score={score} />
        </div>

        {(breakdown?.image_similarity != null || breakdown?.text_similarity != null) && (
          <div className="mt-2 flex flex-wrap gap-1">
            {breakdown.image_similarity != null && (
              <Badge tone="accent" title="Cosine similarity on the image side of the query">
                img {breakdown.image_similarity.toFixed(3)}
              </Badge>
            )}
            {breakdown.text_similarity != null && (
              <Badge tone="success" title="Cosine similarity on the text side of the query">
                txt {breakdown.text_similarity.toFixed(3)}
              </Badge>
            )}
          </div>
        )}

        {showBreakdown && breakdown && breakdown.channels.length > 0 && (
          <div className="mt-auto pt-2">
            <button
              type="button"
              onClick={() => setIsExpanded((open) => !open)}
              aria-expanded={isExpanded}
              className="text-ink-400 hover:text-ink-700 inline-flex items-center gap-1 text-[11px] font-medium"
            >
              <ChevronDown
                className={cx('size-3 transition-transform', isExpanded && 'rotate-180')}
                aria-hidden
              />
              {isExpanded ? 'Hide' : 'Why this rank?'}
            </button>
            {isExpanded && <ChannelTable breakdown={breakdown} />}
          </div>
        )}
      </div>
    </article>
  )
}

/** Colour-codes the final score so a scan of the grid shows relative confidence. */
function ScorePill({ score }: { score: number }) {
  const tone =
    score >= 0.75 ? 'success' : score >= 0.45 ? 'accent' : score >= 0.2 ? 'neutral' : 'warning'
  return (
    <Badge tone={tone} title="Final fused score (relative to this result set)">
      {score.toFixed(3)}
    </Badge>
  )
}

function ChannelTable({ breakdown }: { breakdown: NonNullable<SearchResult['breakdown']> }) {
  return (
    <div className="border-ink-100 mt-2 space-y-1.5 border-t pt-2">
      {breakdown.channels.map((channel) => (
        <div key={channel.channel} className="text-[11px]">
          <div className="text-ink-500 flex items-center justify-between gap-2">
            <span className="truncate">{channelLabel[channel.channel]}</span>
            <span className="text-ink-700 shrink-0 font-mono tabular-nums">
              {channel.raw.toFixed(3)}
            </span>
          </div>
          <div className="mt-0.5 flex items-center gap-1.5">
            <div className="bg-ink-100 h-1 flex-1 overflow-hidden rounded-full">
              <div
                className="bg-accent-500 h-full rounded-full"
                style={{ width: `${Math.max(0, Math.min(1, channel.normalized)) * 100}%` }}
              />
            </div>
            <span className="text-ink-400 shrink-0 font-mono">×{channel.weight.toFixed(2)}</span>
          </div>
        </div>
      ))}
      <p className="text-ink-400 pt-1 text-[10px] leading-relaxed">
        Bar = normalised score within this channel; ×n = its weight. Strategy{' '}
        <span className="font-mono">{breakdown.strategy}</span>, normalisation{' '}
        <span className="font-mono">{breakdown.normalization}</span>.
      </p>
    </div>
  )
}
