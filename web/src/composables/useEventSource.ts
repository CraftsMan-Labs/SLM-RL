import { onUnmounted, ref, toValue, watch, type MaybeRefOrGetter } from 'vue'

export type EventSourceStatus = 'connecting' | 'open' | 'closed'

/**
 * Subscribe to an SSE URL. Browser EventSource reconnects automatically;
 * we surface connection state and forward parsed JSON messages.
 */
export function useEventSource<T = unknown>(
  url: MaybeRefOrGetter<string | null | undefined>,
  onMessage: (data: T) => void,
) {
  const status = ref<EventSourceStatus>('closed')
  let source: EventSource | null = null

  function disconnect() {
    if (source) {
      source.onopen = null
      source.onerror = null
      source.onmessage = null
      source.close()
      source = null
    }
    status.value = 'closed'
  }

  function connect(target: string) {
    disconnect()
    status.value = 'connecting'
    const es = new EventSource(target)
    source = es
    es.onopen = () => {
      if (source === es) status.value = 'open'
    }
    es.onerror = () => {
      if (source === es) status.value = 'closed'
    }
    es.onmessage = (msg) => {
      try {
        onMessage(JSON.parse(msg.data) as T)
      } catch {
        /* ignore malformed events; stream continues */
      }
    }
  }

  watch(
    () => toValue(url),
    (next) => {
      if (!next) {
        disconnect()
        return
      }
      connect(next)
    },
    { immediate: true },
  )

  onUnmounted(disconnect)

  return { status, disconnect }
}
