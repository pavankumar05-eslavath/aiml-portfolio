/**
 * Types mirroring the backend's Pydantic schemas.
 *
 * Hand-maintained rather than generated, to keep the toolchain simple. The
 * backend publishes an OpenAPI document at `/openapi.json`, so these can be
 * replaced with generated clients (`openapi-typescript`) if the surface grows.
 * When changing a backend schema, change the matching type here.
 */

export type SearchMode = 'text' | 'image' | 'multimodal'

export type SortOption = 'relevance' | 'price_asc' | 'price_desc' | 'name_asc'

export type FusionStrategy = 'weighted_sum' | 'rrf' | 'embedding_fusion'

export type ScoreNormalization = 'none' | 'minmax' | 'zscore'

/** A single source of ranking evidence. */
export type RetrievalChannel =
  | 'text_to_text'
  | 'text_to_image'
  | 'image_to_image'
  | 'image_to_text'
  | 'fused_to_image'
  | 'fused_to_text'
  | 'lexical'

export interface ProductSummary {
  id: string
  name: string
  description: string | null
  category: string | null
  subcategory: string | null
  brand: string | null
  colour: string | null
  gender: string | null
  usage: string | null
  price: number | null
  currency: string
  in_stock: boolean
  image_url: string | null
}

export interface Product extends ProductSummary {
  external_id: string | null
  season: string | null
  year: number | null
  search_document: string | null
  indexed_at: string | null
  has_image_vector: boolean
  has_text_vector: boolean
  index_error: string | null
  created_at: string
  updated_at: string
}

export interface ChannelScore {
  channel: RetrievalChannel
  raw: number
  normalized: number
  weight: number
  rank: number | null
}

export interface ScoreBreakdown {
  final_score: number
  image_similarity: number | null
  text_similarity: number | null
  lexical_score: number | null
  channels: ChannelScore[]
  strategy: FusionStrategy
  normalization: ScoreNormalization
}

export interface SearchResult {
  rank: number
  product: ProductSummary
  score: number
  breakdown: ScoreBreakdown | null
}

export interface SearchTimings {
  embed_ms: number
  retrieve_ms: number
  fuse_ms: number
  hydrate_ms: number
  total_ms: number
}

export interface SearchResponse {
  mode: SearchMode
  query: string | null
  has_image_query: boolean
  total_candidates: number
  returned: number
  top_k: number
  offset: number
  results: SearchResult[]
  channels_used: RetrievalChannel[]
  weights: Record<string, number>
  strategy: FusionStrategy
  timings: SearchTimings
  model_name: string
  warnings: string[]
}

export interface SearchFilters {
  categories: string[]
  subcategories: string[]
  brands: string[]
  colours: string[]
  genders: string[]
  min_price: number | null
  max_price: number | null
  in_stock_only: boolean
}

export interface FusionOverrides {
  strategy?: FusionStrategy
  normalization?: ScoreNormalization
  image_weight?: number
  text_weight?: number
  lexical_weight?: number
  cross_modal_weight?: number
  embedding_fusion_alpha?: number
  rrf_k?: number
}

export interface SearchOptions {
  top_k: number
  offset: number
  filters: SearchFilters
  sort: SortOption
  fusion?: FusionOverrides
  explain: boolean
}

export interface FacetValue {
  value: string
  count: number
}

export interface CatalogFacets {
  categories: FacetValue[]
  brands: FacetValue[]
  colours: FacetValue[]
  price_min: number | null
  price_max: number | null
}

export interface CatalogStats {
  total_products: number
  indexed_products: number
  pending_products: number
  failed_products: number
  with_image_vector: number
  with_text_vector: number
  vector_points: number
  collection: string
  embedding_dim: number | null
  model_name: string
  last_indexed_at: string | null
}

export interface IndexFailure {
  product_id: string
  name: string | null
  reason: string
}

export interface IndexReport {
  requested: number
  embedded: number
  skipped_unchanged: number
  failed: number
  image_vectors: number
  text_vectors: number
  duration_ms: number
  failures: IndexFailure[]
}

export interface DependencyHealth {
  name: string
  status: 'ok' | 'degraded' | 'unavailable'
  detail: string | null
  latency_ms: number | null
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'unavailable'
  version: string
  environment: string
  model_name: string
  model_loaded: boolean
  embedding_dim: number | null
  device: string | null
  dependencies: DependencyHealth[]
}

export interface Page<T> {
  items: T[]
  total: number
  limit: number
  offset: number
}

export interface ProductInput {
  external_id?: string | null
  name: string
  description?: string | null
  category?: string | null
  subcategory?: string | null
  brand?: string | null
  colour?: string | null
  gender?: string | null
  usage?: string | null
  season?: string | null
  year?: number | null
  price?: number | null
  currency?: string
  in_stock?: boolean
  image_url?: string | null
}

/** Default, unconstrained filter set. */
export const emptyFilters = (): SearchFilters => ({
  categories: [],
  subcategories: [],
  brands: [],
  colours: [],
  genders: [],
  min_price: null,
  max_price: null,
  in_stock_only: false,
})

/** Whether a filter set constrains anything. */
export const hasActiveFilters = (f: SearchFilters): boolean =>
  f.categories.length > 0 ||
  f.subcategories.length > 0 ||
  f.brands.length > 0 ||
  f.colours.length > 0 ||
  f.genders.length > 0 ||
  f.min_price !== null ||
  f.max_price !== null ||
  f.in_stock_only

/** Number of active filter constraints, for the UI badge. */
export const countActiveFilters = (f: SearchFilters): number =>
  f.categories.length +
  f.subcategories.length +
  f.brands.length +
  f.colours.length +
  f.genders.length +
  (f.min_price !== null ? 1 : 0) +
  (f.max_price !== null ? 1 : 0) +
  (f.in_stock_only ? 1 : 0)

/** Human-readable label for a retrieval channel. */
export const channelLabel: Record<RetrievalChannel, string> = {
  text_to_text: 'text -> product text',
  text_to_image: 'text -> product image',
  image_to_image: 'image -> product image',
  image_to_text: 'image -> product text',
  fused_to_image: 'fused -> product image',
  fused_to_text: 'fused -> product text',
  lexical: 'keyword match',
}
