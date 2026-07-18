<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiModal from '@/components/ui/UiModal.vue'
import ScreenPanel from '@/components/watch/ScreenPanel.vue'
import { getTheaterScores, type TheaterScores } from '@/api/projects'
import { useWatchStream, type EpisodeState } from '@/composables/useWatchStream'

const props = defineProps<{
  projectName: string
  /** When true, show the A/B panel (opened by Theater A/B). */
  open: boolean
  busy?: boolean
  theaterRunning?: boolean
}>()

const maxTurns = defineModel<number>('maxTurns', { required: true })
const theaterEpisodes = defineModel<number>('theaterEpisodes', { required: true })

const emit = defineEmits<{
  close: []
  start: []
  stop: []
}>()

const scores = ref<TheaterScores | null>(null)
let poll: ReturnType<typeof setInterval> | null = null

// Only subscribe while the panel is open — otherwise both sides silently
// attach to /watch and keep stale evolve episodes (e.g. g2-2) in "Champion".
const {
  episodes: baseEpisodes,
} = useWatchStream(
  () => (props.open ? props.projectName : ''),
  null,
  () => (props.open ? 'base' : null),
)
const {
  episodes: champEpisodes,
} = useWatchStream(
  () => (props.open ? props.projectName : ''),
  null,
  () => (props.open ? 'champion' : null),
)

const watching = ref<{ side: 'base' | 'champion'; id: string; gen: number | null } | null>(
  null,
)
const watchRestartToken = ref(0)

async function refreshScores() {
  if (!props.open || !props.projectName) return
  try {
    scores.value = await getTheaterScores(props.projectName)
  } catch {
    // transient while theater is starting
  }
}

function sideLabel(label: string, side: TheaterScores['base']) {
  const run = scores.value?.run
  if (!side) {
    if (label === 'champion') {
      if (run?.phase === 'failed') {
        return `champion: stopped — ${run.error || run.message || 'retry Start Theater'}`
      }
      // Job died mid-run (watchfiles / OOM) but status.json still says "base".
      if (
        !props.theaterRunning
        && (run?.phase === 'base' || run?.phase === 'champion' || run?.phase === 'base_done')
      ) {
        const ep = run.episode ?? run.completed ?? 0
        const total = run.episodes ?? '?'
        return `champion: stalled at base ${ep}/${total} — Start Theater to retry`
      }
      if (run?.phase === 'base' || run?.phase === 'base_done') {
        const ep = run.episode ?? run.completed ?? 0
        const total = run.episodes ?? '?'
        return `champion: waiting for base (${ep}/${total})`
      }
      if (run?.phase === 'done' && run.message) {
        return `champion: ${run.message}`
      }
      if (run?.phase === 'champion') {
        const ep = run.episode ?? 0
        const total = run.episodes ?? '?'
        return `champion: starting (${ep}/${total})`
      }
    }
    return `${label}: waiting…`
  }
  const parts = [`episodes ${side.episodes ?? 0}`]
  if (side.mean_score != null) parts.push(`mean ${Number(side.mean_score).toFixed(2)}`)
  // Atari uses score:<n>, not win/loss — hide a misleading 0% win rate.
  if (side.win_rate != null && side.mean_score == null) {
    parts.push(`win ${(Number(side.win_rate) * 100).toFixed(0)}%`)
  }
  if (side.status) parts.push(String(side.status))
  return `${label}: ${parts.join(' · ')}`
}

function onWatch(side: 'base' | 'champion', ep: EpisodeState) {
  watching.value = { side, id: ep.id, gen: ep.generation }
  watchRestartToken.value += 1
}

function onCloseScreen() {
  watching.value = null
}

function liveEpisode(side: 'base' | 'champion'): EpisodeState | null {
  const list = side === 'base' ? baseEpisodes.value : champEpisodes.value
  return list.find((e) => !e.terminated && !e.truncated) ?? list[0] ?? null
}

const baseLive = computed(() => liveEpisode('base'))
const champLive = computed(() => liveEpisode('champion'))

const autoFollow = computed(() => {
  const b = baseLive.value
  if (b && !b.terminated && !b.truncated) return { side: 'base' as const, ep: b }
  const c = champLive.value
  if (c && !c.terminated && !c.truncated) return { side: 'champion' as const, ep: c }
  return null
})

const screenSide = computed(() => watching.value?.side ?? autoFollow.value?.side ?? null)
const screenId = computed(() => watching.value?.id ?? autoFollow.value?.ep.id ?? null)
const screenGen = computed(
  () => watching.value?.gen ?? autoFollow.value?.ep.generation ?? null,
)

watch(
  () => [props.open, props.projectName] as const,
  ([isOpen]) => {
    if (poll) {
      clearInterval(poll)
      poll = null
    }
    watching.value = null
    if (!isOpen) return
    void refreshScores()
    poll = setInterval(() => {
      void refreshScores()
    }, 3000)
  },
  { immediate: true },
)

onMounted(() => {
  if (props.open) void refreshScores()
})

onUnmounted(() => {
  if (poll) clearInterval(poll)
})
</script>

<template>
  <UiModal :open="open" size="wide" title="Base vs champion" @close="emit('close')">
    <template #header>
      <div class="card-head">
        <div>
          <p class="eyebrow">Theater</p>
          <h2>Base vs champion</h2>
        </div>
        <UiButton variant="ghost" :disabled="busy" @click="emit('close')">Close</UiButton>
      </div>
    </template>

    <p class="lede">
      Opening this window does not start Theater — use <strong>Start Theater</strong> when you are
      ready. Laptop mode: <strong>base finishes every episode first</strong>, then champion loads
      (one model in memory). If the job dies mid-base (Docker hot-reload / OOM), champion stays
      idle — click Start Theater again. Avoid editing <code>slm_rl/</code> during a run.
    </p>
    <p v-if="scores?.run?.phase === 'failed'" class="fail-banner" role="alert">
      Theater stopped early: {{ scores.run.error || scores.run.message }}
    </p>

    <div class="controls">
      <div class="fields">
        <UiField
          label="Episodes"
          for-id="theater-panel-ep"
          hint="How many seeded games each side plays (base, then champion)"
        >
          <input
            id="theater-panel-ep"
            v-model.number="theaterEpisodes"
            type="number"
            min="1"
            max="200"
            :disabled="busy || theaterRunning"
          />
        </UiField>
        <UiField
          label="Max turns"
          for-id="theater-panel-turns"
          hint="Decisions per episode (playground default 30 truncates early; Boxing full bout ≈ 2500)"
        >
          <input
            id="theater-panel-turns"
            v-model.number="maxTurns"
            type="number"
            min="1"
            max="20000"
            :disabled="busy || theaterRunning"
          />
        </UiField>
      </div>
      <div class="actions">
        <UiButton class="start" :disabled="busy || theaterRunning" @click="emit('start')">
          {{ theaterRunning ? 'Theater running…' : 'Start Theater' }}
        </UiButton>
        <UiButton
          v-if="theaterRunning"
          variant="danger"
          :disabled="busy"
          @click="emit('stop')"
        >
          Stop Theater
        </UiButton>
      </div>
    </div>

    <p class="scores" role="status">
      {{ sideLabel('base', scores?.base) }}
      <span class="sep">|</span>
      {{ sideLabel('champion', scores?.champion) }}
    </p>

    <div class="layout" :class="{ withScreen: screenId }">
      <div class="grid">
        <section class="side">
          <header class="side-head">Base (gen 0)</header>
          <ul class="eps">
            <li v-if="!baseEpisodes.length" class="empty">Waiting for base episodes…</li>
            <li v-for="ep in baseEpisodes" :key="ep.id" class="ep">
              <button type="button" class="ep-btn" @click="onWatch('base', ep)">
                <span class="ep-id">{{ ep.id }}</span>
                <span class="ep-meta">
                  {{ ep.outcome || (ep.terminated || ep.truncated ? 'done' : 'live') }}
                </span>
              </button>
            </li>
          </ul>
        </section>
        <section class="side">
          <header class="side-head">Champion</header>
          <ul class="eps">
            <li v-if="!champEpisodes.length" class="empty">
              <template v-if="scores?.run?.phase === 'failed'">
                Theater stopped — use Start Theater to retry.
              </template>
              <template v-else-if="scores?.run?.phase === 'base' || scores?.run?.phase === 'base_done'">
                Waiting until base finishes
                ({{ scores.run.episode ?? scores.run.completed ?? 0 }}/{{ scores.run.episodes ?? '?' }})…
              </template>
              <template v-else>
                Waiting until base finishes…
              </template>
            </li>
            <li v-for="ep in champEpisodes" :key="ep.id" class="ep">
              <button type="button" class="ep-btn" @click="onWatch('champion', ep)">
                <span class="ep-id">{{ ep.id }}</span>
                <span class="ep-meta">
                  {{ ep.outcome || (ep.terminated || ep.truncated ? 'done' : 'live') }}
                </span>
              </button>
            </li>
          </ul>
        </section>
      </div>

      <ScreenPanel
        v-if="screenId && screenSide"
        :key="`${screenSide}-${screenId}-${watchRestartToken}`"
        embedded
        :project-name="projectName"
        :episode-id="screenId"
        :gen="screenGen"
        :theater-side="screenSide"
        :restart-token="watchRestartToken"
        :hud="screenSide === 'base' ? 'Base (gen 0)' : 'Champion'"
        @close="onCloseScreen"
      />
    </div>
  </UiModal>
</template>

<style scoped>
.card-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--space-4);
}

.eyebrow {
  margin: 0 0 var(--space-2);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
}

h2 {
  margin: 0;
  font-size: var(--text-xl);
  letter-spacing: -0.03em;
}

.lede {
  margin: 0;
  color: var(--muted);
  font-size: var(--text-sm);
  line-height: 1.45;
}

.lede strong {
  color: var(--text);
  font-weight: 600;
}

.lede code {
  font-family: var(--font-mono);
  font-size: 0.9em;
}

.fail-banner {
  margin: 0;
  padding: var(--space-3);
  border-radius: var(--radius-sm);
  background: rgba(240, 113, 120, 0.12);
  color: #f07178;
  font-size: var(--text-sm);
  line-height: 1.4;
}

.controls {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
  gap: var(--space-4);
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
  align-items: center;
}

input[type='number'] {
  width: 100%;
  max-width: 12rem;
  min-height: 40px;
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg);
  color: var(--text);
  font: inherit;
}

input:disabled {
  opacity: 0.55;
}

.actions :deep(.start.btn) {
  min-width: 10rem;
}

.scores {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--muted);
  line-height: 1.5;
}

.sep {
  margin: 0 var(--space-2);
  opacity: 0.5;
}

.layout {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-4);
  align-items: start;
}

.layout.withScreen {
  grid-template-columns: minmax(0, 1fr) minmax(220px, 360px);
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-4);
  min-width: 0;
}

.side {
  min-width: 0;
  border-radius: var(--radius-md);
  box-shadow: inset 0 0 0 1px var(--border-soft);
  overflow: hidden;
  background: rgba(0, 0, 0, 0.25);
}

.side-head {
  padding: var(--space-2) var(--space-3);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--border-soft);
}

.eps {
  list-style: none;
  margin: 0;
  padding: var(--space-2);
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: min(40vh, 320px);
  overflow: auto;
}

.empty {
  margin: 0;
  padding: var(--space-3);
  color: var(--meta);
  font-size: var(--text-sm);
}

.ep-btn {
  width: 100%;
  display: flex;
  justify-content: space-between;
  gap: var(--space-2);
  margin: 0;
  padding: 6px 10px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--fg);
  font-family: var(--font-mono);
  font-size: 11px;
  cursor: pointer;
  text-align: left;
}

.ep-btn:hover {
  background: rgba(255, 255, 255, 0.06);
}

.ep-meta {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

@media (max-width: 900px) {
  .layout.withScreen,
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
