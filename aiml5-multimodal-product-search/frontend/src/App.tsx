/** Application shell: layout, navigation, health indicator and routes. */

import { Activity, BookOpen, Boxes, Search } from 'lucide-react'
import { NavLink, Route, Routes } from 'react-router-dom'
import { cx } from '@/lib/format'
import { useHealth } from '@/hooks/useCatalog'
import { CatalogPage } from '@/pages/CatalogPage'
import { ProductPage } from '@/pages/ProductPage'
import { SearchPage } from '@/pages/SearchPage'

export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/product/:id" element={<ProductPage />} />
          <Route path="/catalog" element={<CatalogPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

function Header() {
  const { health } = useHealth()

  const navClass = ({ isActive }: { isActive: boolean }) =>
    cx(
      'inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors',
      isActive ? 'bg-accent-50 text-accent-700' : 'text-ink-600 hover:bg-ink-100',
    )

  return (
    <header className="border-ink-200 sticky top-0 z-40 border-b bg-white/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-4 px-4 sm:px-6 lg:px-8">
        <NavLink to="/" className="flex items-center gap-2">
          <span className="bg-accent-600 grid size-8 place-items-center rounded-lg text-white">
            <Search className="size-4" aria-hidden />
          </span>
          <span className="hidden sm:block">
            <span className="text-ink-900 block text-sm font-semibold leading-tight">
              Multimodal Product Search
            </span>
            <span className="text-ink-400 block text-[11px] leading-tight">
              CLIP · Qdrant · FastAPI
            </span>
          </span>
        </NavLink>

        <nav className="ml-2 flex items-center gap-1">
          <NavLink to="/" end className={navClass}>
            <Search className="size-4" aria-hidden />
            Search
          </NavLink>
          <NavLink to="/catalog" className={navClass}>
            <Boxes className="size-4" aria-hidden />
            Catalogue
          </NavLink>
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <HealthPill health={health} />
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            title="Interactive OpenAPI documentation"
            className="text-ink-500 hover:text-ink-900 inline-flex items-center gap-1 text-xs font-medium"
          >
            <BookOpen className="size-3.5" aria-hidden />
            <span className="hidden sm:inline">API docs</span>
          </a>
        </div>
      </div>
    </header>
  )
}

function HealthPill({ health }: { health: ReturnType<typeof useHealth>['health'] }) {
  if (!health) {
    return (
      <span className="text-ink-400 inline-flex items-center gap-1.5 text-xs">
        <span className="bg-ink-300 size-2 animate-pulse rounded-full" />
        connecting
      </span>
    )
  }

  const tone =
    health.status === 'ok'
      ? 'bg-emerald-500'
      : health.status === 'degraded'
        ? 'bg-amber-500'
        : 'bg-red-500'

  const detail = health.model_loaded
    ? `${health.model_name.split('/').pop()} · ${health.embedding_dim}d · ${health.device}`
    : 'model loading…'

  return (
    <span
      title={health.dependencies.map((d) => `${d.name}: ${d.status} (${d.detail ?? ''})`).join('\n')}
      className="text-ink-500 inline-flex items-center gap-1.5 text-xs"
    >
      <Activity className="size-3.5" aria-hidden />
      <span className={cx('size-2 rounded-full', tone)} />
      <span className="hidden font-mono md:inline">{detail}</span>
    </span>
  )
}

function Footer() {
  return (
    <footer className="border-ink-200 mt-10 border-t bg-white">
      <div className="text-ink-400 mx-auto max-w-7xl px-4 py-5 text-xs sm:px-6 lg:px-8">
        <p>
          Product images and attributes come from the public{' '}
          <a
            href="https://huggingface.co/datasets/ashraq/fashion-product-images-small"
            target="_blank"
            rel="noreferrer"
            className="hover:text-ink-700 underline"
          >
            fashion-product-images-small
          </a>{' '}
          dataset. Brands are parsed from product names; descriptions are templated from real
          attributes; <strong>prices are synthetic</strong> and exist only to demonstrate
          filtering.
        </p>
      </div>
    </footer>
  )
}

function NotFound() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-20 text-center">
      <h1 className="text-ink-900 text-2xl font-semibold">Page not found</h1>
      <NavLink to="/" className="text-accent-600 mt-3 inline-block text-sm underline">
        Back to search
      </NavLink>
    </div>
  )
}
