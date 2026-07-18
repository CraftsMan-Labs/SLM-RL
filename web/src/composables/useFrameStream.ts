import { onUnmounted, ref, toValue, watch, type MaybeRefOrGetter } from 'vue'

/**
 * Consume a multipart/x-mixed-replace PNG stream via fetch and paint each
 * part as a blob URL. Chromium often fails to update `<img src=multipart…>`
 * for long-lived PNG streams; blob swap works everywhere.
 */
export function useFrameStream(url: MaybeRefOrGetter<string | null | undefined>) {
  const frameSrc = ref('')
  const loading = ref(false)
  const error = ref<string | null>(null)

  let abort: AbortController | null = null
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null
  let objectUrl: string | null = null

  function clearObjectUrl() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl)
      objectUrl = null
    }
  }

  function setFrame(bytes: Uint8Array) {
    const copy = new Uint8Array(bytes.byteLength)
    copy.set(bytes)
    const next = URL.createObjectURL(new Blob([copy], { type: 'image/png' }))
    clearObjectUrl()
    objectUrl = next
    frameSrc.value = next
    loading.value = false
  }

  async function openResponse(streamUrl: string, signal: AbortSignal): Promise<Response> {
    let lastStatus = 0
    for (let attempt = 0; attempt < 3; attempt++) {
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError')
      const res = await fetch(streamUrl, {
        signal,
        headers: { Accept: 'multipart/x-mixed-replace' },
        cache: 'no-store',
      })
      lastStatus = res.status
      if (res.status === 503 && attempt < 2) {
        // Prior live/Watch stream may still be releasing its ALE slot.
        await new Promise((r) => setTimeout(r, 350 * (attempt + 1)))
        continue
      }
      return res
    }
    throw new Error(`Frame stream HTTP ${lastStatus || 503}`)
  }

  async function run(streamUrl: string, signal: AbortSignal) {
    loading.value = true
    error.value = null
    frameSrc.value = ''
    clearObjectUrl()

    const res = await openResponse(streamUrl, signal)
    if (!res.ok) {
      const detail = (await res.text().catch(() => '')).trim().slice(0, 160)
      if (res.status === 501) {
        throw new Error(
          detail ||
            'No visual replay — Atari extras missing, or this game has no frame stream.',
        )
      }
      if (res.status === 404) {
        throw new Error('Episode not found for visual replay yet.')
      }
      if (res.status === 503) {
        throw new Error(
          'Too many live screens open — close the other Watch/live screen and retry.',
        )
      }
      throw new Error(detail || `Frame stream HTTP ${res.status}`)
    }
    if (!res.body) {
      throw new Error('Frame stream has no body')
    }

    const ctype = res.headers.get('content-type') || ''
    const boundaryMatch = /boundary=([^;\s]+)/i.exec(ctype)
    if (!boundaryMatch) {
      const buf = new Uint8Array(await res.arrayBuffer())
      if (buf.byteLength > 8 && buf[0] === 0x89 && buf[1] === 0x50) {
        setFrame(buf)
        return
      }
      throw new Error('Not a multipart frame stream')
    }
    const boundary = boundaryMatch[1]!.replace(/^["']|["']$/g, '')
    const delim = new TextEncoder().encode(`--${boundary}`)

    reader = res.body.getReader()
    const activeReader = reader
    let buf = new Uint8Array(0)

    const concat = (a: Uint8Array, b: Uint8Array) => {
      const out = new Uint8Array(a.length + b.length)
      out.set(a, 0)
      out.set(b, a.length)
      return out
    }

    const indexOf = (hay: Uint8Array, needle: Uint8Array, from = 0): number => {
      outer: for (let i = from; i <= hay.length - needle.length; i++) {
        for (let j = 0; j < needle.length; j++) {
          if (hay[i + j] !== needle[j]) continue outer
        }
        return i
      }
      return -1
    }

    try {
      while (!signal.aborted) {
        const { done, value } = await activeReader.read()
        if (done) break
        if (value?.byteLength) buf = concat(buf, value)

        while (true) {
          const start = indexOf(buf, delim)
          if (start < 0) {
            if (buf.length > delim.length) buf = buf.slice(buf.length - delim.length)
            break
          }
          let after = start + delim.length
          if (buf[after] === 0x2d && buf[after + 1] === 0x2d) {
            return
          }
          if (buf[after] === 0x0d && buf[after + 1] === 0x0a) after += 2

          const headerEnd = indexOf(buf, new TextEncoder().encode('\r\n\r\n'), after)
          if (headerEnd < 0) {
            buf = buf.slice(start)
            break
          }
          const headerText = new TextDecoder().decode(buf.slice(after, headerEnd))
          const lenMatch = /content-length:\s*(\d+)/i.exec(headerText)
          const bodyStart = headerEnd + 4
          if (!lenMatch) {
            const next = indexOf(buf, delim, bodyStart)
            if (next < 0) {
              buf = buf.slice(start)
              break
            }
            buf = buf.slice(next)
            continue
          }
          const length = Number(lenMatch[1])
          if (buf.length < bodyStart + length) {
            buf = buf.slice(start)
            break
          }
          const png = buf.slice(bodyStart, bodyStart + length)
          if (png.byteLength >= 8 && png[0] === 0x89 && png[1] === 0x50) {
            setFrame(png)
          }
          let nextFrom = bodyStart + length
          if (buf[nextFrom] === 0x0d && buf[nextFrom + 1] === 0x0a) nextFrom += 2
          buf = buf.slice(nextFrom)
        }
      }
    } finally {
      if (reader === activeReader) reader = null
      await activeReader.cancel().catch(() => undefined)
    }
  }

  function stop() {
    abort?.abort()
    abort = null
    const r = reader
    reader = null
    void r?.cancel().catch(() => undefined)
  }

  function start(streamUrl: string) {
    stop()
    const ctrl = new AbortController()
    abort = ctrl
    void run(streamUrl, ctrl.signal).catch((err: unknown) => {
      if (ctrl.signal.aborted) return
      loading.value = false
      const msg = err instanceof Error ? err.message : 'Frame stream failed'
      error.value = msg.includes('404')
        ? 'No visual replay for this episode yet (or Atari extras missing).'
        : msg
    })
  }

  watch(
    () => toValue(url),
    (next) => {
      if (!next) {
        stop()
        loading.value = false
        error.value = null
        frameSrc.value = ''
        clearObjectUrl()
        return
      }
      start(next)
    },
    { immediate: true },
  )

  onUnmounted(() => {
    stop()
    clearObjectUrl()
  })

  return { frameSrc, loading, error }
}
