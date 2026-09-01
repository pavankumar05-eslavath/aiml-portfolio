/**
 * Image input with drag-and-drop, file picker, paste and a preview.
 *
 * Validates type and size in the browser before upload so an obviously wrong file
 * gets instant feedback instead of a round trip. This is a convenience only - the
 * backend re-validates by actually decoding the bytes, because client-side checks
 * are trivially bypassed.
 */

import { Image as ImageIcon, Upload, X } from 'lucide-react'
import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { Button } from '@/components/ui'
import { cx } from '@/lib/format'
import { api, imageUrl } from '@/services/api'

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']
const MAX_BYTES = 10 * 1024 * 1024

interface Props {
  file: File | null
  onChange: (file: File | null) => void
  disabled?: boolean
}

export function ImageDropzone({ file, onChange, disabled = false }: Props) {
  const [isDragging, setIsDragging] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const inputId = useId()

  // Object URLs must be revoked or the blob stays in memory for the page's life.
  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const accept = useCallback(
    (candidate: File | null | undefined) => {
      setLocalError(null)
      if (!candidate) return
      if (!ACCEPTED.includes(candidate.type)) {
        setLocalError(`${candidate.type || 'That file type'} is not supported. Use JPEG, PNG or WEBP.`)
        return
      }
      if (candidate.size > MAX_BYTES) {
        setLocalError(
          `Image is ${(candidate.size / 1_048_576).toFixed(1)} MB; the limit is 10 MB.`,
        )
        return
      }
      onChange(candidate)
    },
    [onChange],
  )

  // Let the user paste a screenshot straight into the page.
  useEffect(() => {
    if (disabled) return
    const onPaste = (event: ClipboardEvent) => {
      const item = Array.from(event.clipboardData?.items ?? []).find((i) =>
        i.type.startsWith('image/'),
      )
      if (item) accept(item.getAsFile())
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [accept, disabled])

  const clear = () => {
    onChange(null)
    setLocalError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  if (file && preview) {
    return (
      <div className="space-y-2">
        <div className="border-ink-200 bg-swatch relative overflow-hidden rounded-xl border">
          <img
            src={preview}
            alt="Query image preview"
            className="mx-auto h-44 w-auto object-contain p-2"
          />
          <button
            type="button"
            onClick={clear}
            disabled={disabled}
            aria-label="Remove query image"
            className="text-ink-600 hover:text-ink-900 absolute top-2 right-2 rounded-full bg-white/95 p-1.5 shadow-sm transition-colors hover:bg-white"
          >
            <X className="size-4" aria-hidden />
          </button>
        </div>
        <p className="text-ink-500 truncate text-xs">
          {file.name} · {(file.size / 1024).toFixed(0)} KB
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <label
        htmlFor={inputId}
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setIsDragging(false)
          if (!disabled) accept(event.dataTransfer.files?.[0])
        }}
        className={cx(
          'flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors',
          isDragging
            ? 'border-accent-500 bg-accent-50'
            : 'border-ink-200 bg-ink-50/50 hover:border-ink-300 hover:bg-ink-50',
          disabled && 'pointer-events-none opacity-60',
        )}
      >
        <div className="bg-white p-2.5 rounded-full shadow-sm">
          {isDragging ? (
            <Upload className="text-accent-600 size-5" aria-hidden />
          ) : (
            <ImageIcon className="text-ink-400 size-5" aria-hidden />
          )}
        </div>
        <div>
          <p className="text-ink-700 text-sm font-medium">
            {isDragging ? 'Drop to use this image' : 'Drop an image, or click to browse'}
          </p>
          <p className="text-ink-400 mt-0.5 text-xs">
            JPEG, PNG or WEBP · up to 10 MB · paste also works
          </p>
        </div>
        <input
          id={inputId}
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(',')}
          disabled={disabled}
          className="sr-only"
          onChange={(event) => accept(event.target.files?.[0])}
        />
      </label>

      {localError && (
        <p role="alert" className="text-xs text-red-600">
          {localError}
        </p>
      )}
      <ExampleImages onPick={onChange} disabled={disabled} />
    </div>
  )
}

/**
 * Load a catalogue image as the query, so the demo is usable without the visitor
 * having a product photo to hand. Fetched through the backend's media endpoint
 * and converted to a File so it takes exactly the same upload path as a real one.
 */
function ExampleImages({
  onPick,
  disabled,
}: {
  onPick: (file: File) => void
  disabled?: boolean
}) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const useExample = async () => {
    setIsLoading(true)
    setError(null)
    try {
      // Pick a random indexed product that has an image.
      const page = await api.listProducts({ limit: 40, offset: 0 })
      const candidates = page.items.filter((product) => product.image_url)
      if (candidates.length === 0) throw new Error('no catalogue images available')
      const chosen = candidates[Math.floor(Math.random() * candidates.length)]
      const url = imageUrl(chosen.image_url)
      if (!url) throw new Error('could not resolve the image URL')
      const response = await fetch(url)
      if (!response.ok) throw new Error(`image request failed (${response.status})`)
      const blob = await response.blob()
      onPick(new File([blob], `${chosen.name || 'example'}.jpg`, { type: blob.type }))
    } catch {
      setError('Could not load an example. Is the catalogue indexed?')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={useExample}
        isLoading={isLoading}
        disabled={disabled}
      >
        Use a random catalogue image
      </Button>
      {error && (
        <p role="alert" className="text-xs text-red-600">
          {error}
        </p>
      )}
    </div>
  )
}
