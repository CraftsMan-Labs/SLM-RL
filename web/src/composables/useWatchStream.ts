import { computed, ref, toValue, watch, type MaybeRefOrGetter } from 'vue'
import {
  theaterEventsUrl,
  watchEventsUrl,
  type TheaterSide,
  type WatchEvent,
} from '@/api/watch'
import { useEventSource } from '@/composables/useEventSource'

const MAX_EPISODES = 30
const MAX_STEPS_PER_EPISODE = 40

export type EpisodeStep = WatchEvent

export type EpisodeState = {
  id: string
  generation: number | null
  modelId: string | null
  seed: number | null
  steps: EpisodeStep[]
  outcome: string | null
  cumReward: number | null
  terminated: boolean
  truncated: boolean
}

export function useWatchStream(
  projectName: MaybeRefOrGetter<string>,
  gen: MaybeRefOrGetter<number | null | undefined> = null,
  theaterSide: MaybeRefOrGetter<TheaterSide | null | undefined> = null,
) {
  const episodes = ref<EpisodeState[]>([])
  const wins = ref(0)
  const currentGen = ref<number | null>(null)
  const lastEventAt = ref<number | null>(null)
  const seen = new Set<string>()

  const eventsUrl = computed(() => {
    const name = toValue(projectName)
    if (!name) return null
    const side = toValue(theaterSide)
    if (side) return theaterEventsUrl(name, side)
    return watchEventsUrl(name, toValue(gen))
  })

  function handleEvent(ev: WatchEvent) {
    lastEventAt.value = Date.now()
    if (ev.generation !== undefined && ev.generation !== null) {
      currentGen.value = ev.generation
    }

    const id = ev.episode_id == null ? '' : String(ev.episode_id)
    if (!id) return

    let ep = episodes.value.find((e) => e.id === id)
    if (!ep) {
      ep = {
        id,
        generation: ev.generation ?? null,
        modelId: ev.model_id ?? null,
        seed: ev.seed ?? null,
        steps: [],
        outcome: null,
        cumReward: null,
        terminated: false,
        truncated: false,
      }
      episodes.value = [ep, ...episodes.value].slice(0, MAX_EPISODES)
      if (!seen.has(id)) seen.add(id)
    }

    if (ev.generation != null) ep.generation = ev.generation
    if (ev.model_id != null) ep.modelId = ev.model_id
    if (ev.seed != null) ep.seed = ev.seed

    ep.steps = [ev, ...ep.steps].slice(0, MAX_STEPS_PER_EPISODE)

    if (ev.terminated || ev.truncated) {
      ep.terminated = Boolean(ev.terminated)
      ep.truncated = Boolean(ev.truncated)
      const outcome = ev.outcome || (ev.truncated ? 'truncated' : '')
      const wasWin = ep.outcome === 'win'
      ep.outcome = outcome || null
      ep.cumReward = ev.cum_reward ?? ep.cumReward
      if (outcome === 'win' && !wasWin) wins.value += 1
    } else if (ev.cum_reward != null) {
      ep.cumReward = ev.cum_reward
    }
  }

  const { status } = useEventSource<WatchEvent>(eventsUrl, handleEvent)

  const episodeCount = computed(() => seen.size)

  const activeEpisode = computed(() => {
    const open = episodes.value.find((e) => !e.terminated && !e.truncated)
    return open ?? episodes.value[0] ?? null
  })

  function reset() {
    episodes.value = []
    wins.value = 0
    currentGen.value = null
    lastEventAt.value = null
    seen.clear()
  }

  // Drop stale episodes when the stream URL changes (e.g. theater panel
  // opens and switches from /watch → /theater/.../champion/events).
  watch(
    eventsUrl,
    (next, prev) => {
      if (next !== prev) reset()
    },
  )

  return {
    status,
    episodes,
    wins,
    currentGen,
    episodeCount,
    lastEventAt,
    activeEpisode,
    reset,
  }
}
