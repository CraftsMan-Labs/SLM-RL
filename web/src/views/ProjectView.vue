<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppShell from '@/components/shell/AppShell.vue'
import ProjectConfigAside from '@/components/projects/ProjectConfigAside.vue'
import PublishModal from '@/components/projects/PublishModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCard from '@/components/ui/UiCard.vue'
import EvolvePanel from '@/components/watch/EvolvePanel.vue'
import LiveWatchPanel from '@/components/watch/LiveWatchPanel.vue'
import RunPanel from '@/components/watch/RunPanel.vue'
import TheaterComparePanel from '@/components/watch/TheaterComparePanel.vue'
import { ApiError } from '@/api/client'
import {
  evolveProject,
  getKnobs,
  getProjectLog,
  launchTheater,
  listProjects,
  publishProject,
  runProject,
  stopProject,
  updateProjectKnobs,
  type KnobSchema,
  type Project,
} from '@/api/projects'
import { stopAllFrameStreams } from '@/composables/useFrameStream'
import { useProfile } from '@/composables/useProfile'
import { workshopDqnUrl, workshopPackUrl } from '@/lib/workshopHf'

const route = useRoute()
const router = useRouter()
const { hasToken, profile } = useProfile()

const name = computed(() => String(route.params.name))
const project = ref<Project | null>(null)
const publishOpen = ref(false)
const knobs = ref<KnobSchema[]>([])
const knobValues = ref<Record<string, unknown>>({})
const datasetUrl = ref('')
const adapterUrl = ref('')
const dqnUrl = ref('')
const generations = ref(2)
const runEpisodes = ref(30)
/** Decisions per episode for Run / Theater (playground create default is 30). */
const maxTurns = ref(30)
/** Seeded games per Theater side (base then champion). */
const theaterEpisodes = ref(4)
/** LLM rollouts per Evolve generation (editable; playground default 2). */
const episodesPerGen = ref(2)
const logText = ref('')
const logKind = ref<'rollout' | 'evolve' | 'theater'>('rollout')
const theaterOpen = ref(false)
const evolveOpen = ref(false)
const runOpen = ref(false)
const busy = ref(false)
const message = ref<string | null>(null)
const error = ref<string | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null
let toastTimer: ReturnType<typeof setTimeout> | null = null

const statusLabel = computed(() => {
  const s = project.value?.status
  if (s === 'running') return 'Running'
  if (s === 'complete') return 'Complete'
  if (s === 'stopped') return 'Stopped'
  if (s === 'no_data') return 'Ready'
  return s || '—'
})

const statusTone = computed(() => {
  const s = project.value?.status
  if (s === 'running') return 'live'
  if (s === 'complete') return 'ok'
  if (s === 'stopped') return 'stopped'
  return 'idle'
})

const canStop = computed(() => (project.value?.active_jobs?.length ?? 0) > 0)
const canStopTheater = computed(() =>
  (project.value?.active_jobs ?? []).includes('theater'),
)
const evolveRunning = computed(() =>
  (project.value?.active_jobs ?? []).includes('evolve'),
)
const runRunning = computed(() =>
  (project.value?.active_jobs ?? []).includes('quick'),
)
const theaterRunning = computed(() =>
  (project.value?.active_jobs ?? []).includes('theater'),
)

const studioOpen = computed(() => evolveOpen.value || runOpen.value || theaterOpen.value)

function closeOtherStudios(keep: 'evolve' | 'run' | 'theater') {
  if (keep !== 'evolve') evolveOpen.value = false
  if (keep !== 'run') runOpen.value = false
  if (keep !== 'theater') theaterOpen.value = false
}

// Opening/closing Run / Evolve / Theater unmounts LiveWatch (or the studio
// screen). Abort every ALE replay immediately so the 8-slot cap never sticks
// on a panel the user already left.
watch(studioOpen, () => {
  stopAllFrameStreams()
})

const meanLabel = computed(() => {
  const m = project.value?.mean_score
  return m == null ? '—' : m.toFixed(2)
})

function selectLogKind(kind: 'rollout' | 'evolve' | 'theater') {
  logKind.value = kind
  void refreshLog()
}

function flash(msg: string) {
  message.value = msg
  if (toastTimer) clearTimeout(toastTimer)
  toastTimer = setTimeout(() => {
    message.value = null
  }, 4200)
}

function onEvolveDead(detail: { phase: string; phaseGen: number | null }) {
  const gen = detail.phaseGen != null ? ` gen ${detail.phaseGen}` : ''
  const phase = detail.phase ? ` (was ${detail.phase}${gen})` : ''
  message.value =
    `Evolve process died${phase}. Nothing is collecting rollouts — open Evolve → Start Evolve to resume.`
  if (toastTimer) clearTimeout(toastTimer)
  toastTimer = setTimeout(() => {
    message.value = null
  }, 10_000)
  evolveOpen.value = true
}

async function loadProject() {
  const rows = await listProjects()
  project.value = rows.find((p) => p.name === name.value) ?? null
  if (!project.value) {
    error.value = `Project “${name.value}” not found.`
    return
  }
  error.value = null
  const game = project.value.game
  const packDefault = game ? workshopPackUrl(game) : ''
  const dqnDefault = game ? workshopDqnUrl(game) : ''
  datasetUrl.value = project.value.dataset_url || packDefault
  adapterUrl.value = project.value.adapter_url || packDefault
  dqnUrl.value = project.value.dqn_url || dqnDefault
  if (game) {
    knobs.value = await getKnobs(game)
    const frozen = project.value.knob_values ?? {}
    const next: Record<string, unknown> = {}
    for (const k of knobs.value) {
      next[k.key] = frozen[k.key] ?? k.default
    }
    knobValues.value = next
    const ep = Number(next.episodes_per_generation)
    episodesPerGen.value = Number.isFinite(ep) && ep > 0 ? ep : 1
    // Don't clobber an in-progress edit while Run / Theater studio is open.
    if (!runOpen.value && !theaterOpen.value) {
      const turns = Number(next.max_turns)
      maxTurns.value = Number.isFinite(turns) && turns > 0 ? turns : 30
    }
  }
}

function clampMaxTurns(raw: unknown): number {
  const n = Math.floor(Number(raw) || 30)
  return Math.max(1, Math.min(20_000, n))
}

async function persistMaxTurns(): Promise<number> {
  const n = clampMaxTurns(maxTurns.value)
  maxTurns.value = n
  const result = await updateProjectKnobs(name.value, { max_turns: n })
  knobValues.value = { ...knobValues.value, ...result.knob_values }
  return n
}

async function onSaveEpisodesPerGen() {
  error.value = null
  const n = Math.max(1, Math.min(200, Math.floor(Number(episodesPerGen.value) || 1)))
  episodesPerGen.value = n
  busy.value = true
  try {
    const result = await updateProjectKnobs(name.value, { episodes_per_generation: n })
    knobValues.value = { ...knobValues.value, ...result.knob_values }
    flash(`Episodes per generation → ${n}. Stop Evolve, then Start Evolve again to apply.`)
    await loadProject()
  } catch (err) {
    error.value =
      err instanceof ApiError && err.status === 409
        ? 'Stop Evolve before changing episodes per generation.'
        : err instanceof Error
          ? err.message
          : 'Failed to update knobs'
  } finally {
    busy.value = false
  }
}

async function refreshLog() {
  try {
    logText.value = await getProjectLog(name.value, logKind.value)
  } catch {
    /* keep previous */
  }
}

function openRunPanel() {
  error.value = null
  logKind.value = 'rollout'
  closeOtherStudios('run')
  runOpen.value = true
  void refreshLog()
}

async function onStartRun() {
  error.value = null
  busy.value = true
  logKind.value = 'rollout'
  runOpen.value = true
  try {
    const turns = await persistMaxTurns()
    await runProject(name.value, { episodes: runEpisodes.value, agent: 'solver' })
    flash(`Quick run started (max turns ${turns}) — watch progress in this window.`)
    await loadProject()
    await refreshLog()
  } catch (err) {
    error.value =
      err instanceof ApiError && err.status === 409
        ? 'Busy — another quick run is already in progress. Stop it before changing max turns.'
        : err instanceof Error
          ? err.message
          : 'Run failed'
  } finally {
    busy.value = false
  }
}

async function onStopRun() {
  await onStop(['quick'])
}

function openEvolvePanel() {
  error.value = null
  logKind.value = 'evolve'
  closeOtherStudios('evolve')
  evolveOpen.value = true
  void refreshLog()
}

async function onStartEvolve() {
  error.value = null
  busy.value = true
  logKind.value = 'evolve'
  evolveOpen.value = true
  try {
    await evolveProject(name.value, generations.value, {
      dataset_url: datasetUrl.value.trim() || null,
      adapter_url: adapterUrl.value.trim() || null,
      dqn_url: dqnUrl.value.trim() || null,
    })
    flash('Evolve started — watch progress in this window.')
    await loadProject()
    await refreshLog()
  } catch (err) {
    error.value =
      err instanceof ApiError && err.status === 409
        ? 'Busy — another evolve is already running.'
        : err instanceof Error
          ? err.message
          : 'Evolve failed'
  } finally {
    busy.value = false
  }
}

async function onStopEvolve() {
  await onStop(['evolve'])
}

function openTheaterPanel() {
  error.value = null
  logKind.value = 'theater'
  closeOtherStudios('theater')
  theaterOpen.value = true
  void refreshLog()
}

async function onStartTheater() {
  error.value = null
  busy.value = true
  logKind.value = 'theater'
  theaterOpen.value = true
  try {
    const turns = await persistMaxTurns()
    const eps = Math.max(1, Math.min(200, Math.floor(Number(theaterEpisodes.value) || 4)))
    theaterEpisodes.value = eps
    await launchTheater(name.value, eps)
    flash(`Theater A/B started (max turns ${turns}, ${eps} eps) — watch this window.`)
    await loadProject()
    await refreshLog()
  } catch (err) {
    // Still show the compare panel if a previous exhibition left data on disk.
    error.value =
      err instanceof ApiError && err.status === 409
        ? 'Busy — another job is running. Stop evolve first, then Start Theater.'
        : err instanceof Error
          ? err.message
          : 'Theater failed'
  } finally {
    busy.value = false
  }
}

function openPublish() {
  if (!hasToken.value) return
  error.value = null
  publishOpen.value = true
}

function formatPublishFlash(result: {
  model_repo: string | null
  dataset_repo: string | null
  model_error: string | null
  dataset_error: string | null
  message: string | null
}) {
  const parts: string[] = []
  if (result.model_repo && !result.model_error) {
    parts.push(`model ${result.model_repo}`)
  } else if (result.model_error) {
    parts.push(`model failed: ${result.model_error}`)
  }
  if (result.dataset_repo && !result.dataset_error) {
    parts.push(`dataset ${result.dataset_repo}`)
  } else if (result.dataset_error) {
    parts.push(`dataset failed: ${result.dataset_error}`)
  }
  if (result.message) parts.push(result.message)
  return parts.length ? parts.join(' · ') : 'Published.'
}

async function onPublishConfirm(repoName: string) {
  error.value = null
  busy.value = true
  try {
    const result = await publishProject(name.value, repoName)
    flash(formatPublishFlash(result))
    publishOpen.value = false
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Publish failed'
  } finally {
    busy.value = false
  }
}

async function onStop(kinds?: string[]) {
  error.value = null
  busy.value = true
  try {
    const result = await stopProject(name.value, kinds)
    flash(`Stopped ${result.stopped.join(', ')}.`)
    await loadProject()
    await refreshLog()
  } catch (err) {
    error.value =
      err instanceof ApiError && err.status === 404
        ? 'Nothing running for this project.'
        : err instanceof Error
          ? err.message
          : 'Stop failed'
  } finally {
    busy.value = false
  }
}

async function onStopTheater() {
  await onStop(['theater'])
}

watch(name, async () => {
  await loadProject()
  await refreshLog()
})

onMounted(async () => {
  try {
    await loadProject()
    await refreshLog()
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load project'
  }
  pollTimer = setInterval(() => {
    void refreshLog()
    void loadProject().catch(() => undefined)
  }, 4000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (toastTimer) clearTimeout(toastTimer)
})
</script>

<template>
  <AppShell :title="name">
    <template #aside>
      <ProjectConfigAside
        v-model:dataset-url="datasetUrl"
        v-model:adapter-url="adapterUrl"
        v-model:dqn-url="dqnUrl"
        v-model:generations="generations"
        v-model:episodes-per-gen="episodesPerGen"
        v-model:run-episodes="runEpisodes"
        :game="project?.game"
        :knobs="knobs"
        :knob-values="knobValues"
        :busy="busy"
        :can-stop="canStop"
        @save-episodes-per-gen="onSaveEpisodesPerGen"
      />
    </template>

    <button class="back" type="button" @click="router.push({ name: 'projects' })">
      ← All projects
    </button>

    <header class="hero">
      <div class="hero-copy">
        <p class="eyebrow">Workspace</p>
        <div class="title-row">
          <h1>{{ name }}</h1>
          <span class="status" :data-tone="statusTone" role="status">{{ statusLabel }}</span>
        </div>
      </div>

      <div class="kpi-row" aria-label="Run metrics">
        <div class="kpi" style="--i: 0">
          <span class="kpi-label">Game</span>
          <span class="kpi-value mono">{{ project?.game || '—' }}</span>
        </div>
        <div class="kpi" style="--i: 1">
          <span class="kpi-label">Mean score</span>
          <span :key="meanLabel" class="kpi-value kpi-num">{{ meanLabel }}</span>
        </div>
        <div class="kpi" style="--i: 2">
          <span class="kpi-label">Quick episodes</span>
          <span class="kpi-value kpi-num">{{ runEpisodes }}</span>
        </div>
      </div>
    </header>

    <div class="dock" role="toolbar" aria-label="Project actions">
      <UiButton class="dock-primary" :disabled="busy" @click="openEvolvePanel">
        Evolve
      </UiButton>

      <div class="dock-group" role="group" aria-label="Play and compare">
        <button
          type="button"
          class="dock-item"
          :disabled="busy"
          title="Open quick-run studio (does not start)"
          @click="openRunPanel"
        >
          Run
        </button>
        <span class="dock-rule" aria-hidden="true" />
        <button
          type="button"
          class="dock-item"
          :disabled="busy"
          title="Open Theater A/B studio (does not start)"
          @click="openTheaterPanel"
        >
          Theater
        </button>
      </div>

      <div class="dock-trail">
        <button
          v-if="canStopTheater"
          type="button"
          class="dock-quiet"
          :disabled="busy"
          title="Stop theater only; evolve / training keeps running"
          @click="onStopTheater"
        >
          Stop theater
        </button>
        <button
          v-if="canStop"
          type="button"
          class="dock-danger"
          :disabled="busy"
          title="Stop all jobs for this project (evolve + theater + rollout)"
          @click="onStop()"
        >
          Stop all
        </button>
        <button
          type="button"
          class="dock-quiet"
          :disabled="busy || !hasToken"
          :title="hasToken ? 'Publish to Hugging Face' : 'Add a Hugging Face token on the welcome screen to publish'"
          @click="openPublish"
        >
          Publish
        </button>
      </div>
    </div>

    <p v-if="error" class="err" role="alert">{{ error }}</p>
    <p v-if="message" class="toast" role="status">{{ message }}</p>

    <PublishModal
      :open="publishOpen"
      :project-name="name"
      :hf-username="profile?.hf_username ?? null"
      :busy="busy"
      @close="publishOpen = false"
      @confirm="onPublishConfirm"
    />

    <RunPanel
      v-model:run-episodes="runEpisodes"
      v-model:max-turns="maxTurns"
      :open="runOpen"
      :project-name="name"
      :project-status="project?.status"
      :active-jobs="project?.active_jobs"
      :busy="busy"
      :run-running="runRunning"
      @close="runOpen = false"
      @start="onStartRun"
      @stop="onStopRun"
    />

    <EvolvePanel
      v-model:dataset-url="datasetUrl"
      v-model:adapter-url="adapterUrl"
      v-model:dqn-url="dqnUrl"
      v-model:generations="generations"
      v-model:episodes-per-gen="episodesPerGen"
      :open="evolveOpen"
      :project-name="name"
      :project-status="project?.status"
      :active-jobs="project?.active_jobs"
      :busy="busy"
      :can-stop="canStop"
      :evolve-running="evolveRunning"
      @close="evolveOpen = false"
      @start="onStartEvolve"
      @stop="onStopEvolve"
      @save-episodes-per-gen="onSaveEpisodesPerGen"
      @evolve-dead="onEvolveDead"
    />

    <TheaterComparePanel
      v-model:max-turns="maxTurns"
      v-model:theater-episodes="theaterEpisodes"
      :project-name="name"
      :open="theaterOpen"
      :busy="busy"
      :theater-running="theaterRunning"
      @close="theaterOpen = false"
      @start="onStartTheater"
      @stop="onStopTheater"
    />

    <section class="hero-surface">
      <LiveWatchPanel
        v-if="!studioOpen"
        :project-name="name"
        :project-status="project?.status"
        :active-jobs="project?.active_jobs"
        @evolve-dead="onEvolveDead"
      />
    </section>

    <UiCard eyebrow="Ops" title="Log" class="log-card">
      <div class="log-tabs" role="tablist" aria-label="Log kind">
        <template v-for="(kind, i) in (['rollout', 'evolve', 'theater'] as const)" :key="kind">
          <span v-if="i > 0" class="log-rule" aria-hidden="true" />
          <button
            type="button"
            role="tab"
            class="log-tab"
            :class="{ active: logKind === kind }"
            :aria-selected="logKind === kind"
            @click="selectLogKind(kind)"
          >
            {{ kind }}
          </button>
        </template>
      </div>
      <pre class="log">{{ logText || 'No log yet. Run evolve or theater to see output.' }}</pre>
    </UiCard>
  </AppShell>
</template>

<style scoped>
.back {
  background: none;
  border: 0;
  color: var(--muted);
  padding: 0;
  margin-bottom: var(--space-5);
  cursor: pointer;
  font-size: var(--text-sm);
  transition: color var(--motion-fast) var(--ease-standard);
}

.back:hover {
  color: var(--accent);
}

.hero {
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
  margin-bottom: var(--space-5);
}

.eyebrow {
  margin: 0 0 var(--space-3);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
}

.title-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-3) var(--space-4);
}

h1 {
  margin: 0;
  font-family: var(--font-display);
  font-size: var(--text-2xl);
  font-weight: 500;
  letter-spacing: var(--tracking-display);
  line-height: var(--leading-tight);
}

.status {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: var(--radius-pill);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: var(--frosted);
  color: var(--meta);
}

.status[data-tone='live'] {
  color: var(--accent);
  box-shadow: 0 0 0 1px rgba(0, 153, 255, 0.35);
}

.status[data-tone='live']::before {
  content: '';
  width: 6px;
  height: 6px;
  margin-right: 6px;
  border-radius: 50%;
  background: currentColor;
  animation: pulse 1.4s var(--ease-standard) infinite;
}

.status[data-tone='stopped'] {
  color: var(--danger);
  background: rgba(220, 38, 38, 0.12);
}

.status[data-tone='ok'] {
  color: var(--success);
}

.kpi-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-3);
}

.kpi {
  padding: var(--space-4);
  border-radius: var(--radius-md);
  box-shadow: var(--elev-ring);
  background: var(--surface);
  animation: rise 420ms var(--ease-standard) both;
  animation-delay: calc(var(--i, 0) * 60ms);
}

.kpi-label {
  display: block;
  margin-bottom: var(--space-2);
  font-size: var(--text-xs);
  color: var(--meta);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-family: var(--font-mono);
}

.kpi-value {
  display: block;
  color: var(--fg);
  font-size: var(--text-lg);
  letter-spacing: -0.02em;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kpi-num {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  font-size: var(--text-2xl);
  letter-spacing: -0.01em;
  animation: tick 600ms var(--ease-standard);
}

.mono {
  font-family: var(--font-mono);
  font-size: 0.92em;
}

.dock {
  --dock-h: 40px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-bottom: var(--space-5);
  padding: 8px;
  border-radius: var(--radius-pill);
  background: rgba(255, 255, 255, 0.04);
  box-shadow:
    inset 0 0 0 1px var(--border-soft),
    0 8px 28px rgba(0, 0, 0, 0.35);
  position: sticky;
  top: var(--space-3);
  z-index: 5;
  container-type: inline-size;
  container-name: dock;
}

.dock :deep(.dock-primary.btn) {
  min-height: var(--dock-h);
  padding: 0 18px;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: -0.01em;
  flex: 0 0 auto;
}

.dock :deep(.dock-primary.btn:focus-visible) {
  outline: none;
  box-shadow: var(--focus-ring);
}

.dock-group {
  display: inline-flex;
  align-items: stretch;
  min-height: var(--dock-h);
  border-radius: var(--radius-pill);
  background: rgba(255, 255, 255, 0.06);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.dock-item,
.dock-quiet,
.dock-danger {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--fg);
  font: inherit;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: -0.01em;
  cursor: pointer;
  min-height: var(--dock-h);
  padding: 0 14px;
  transition:
    background var(--motion-fast) var(--ease-standard),
    color var(--motion-fast) var(--ease-standard),
    opacity var(--motion-fast) var(--ease-standard);
}

.dock-item:hover:not(:disabled),
.dock-quiet:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.1);
  color: var(--fg);
}

.dock-item:active:not(:disabled),
.dock-quiet:active:not(:disabled),
.dock-danger:active:not(:disabled) {
  background: rgba(255, 255, 255, 0.06);
}

.dock-item:focus-visible,
.dock-quiet:focus-visible,
.dock-danger:focus-visible {
  outline: none;
  box-shadow: inset 0 0 0 2px color-mix(in oklab, var(--accent), transparent 40%);
  z-index: 1;
}

.dock-item:disabled,
.dock-quiet:disabled,
.dock-danger:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.dock-rule {
  width: 1px;
  align-self: stretch;
  margin: 8px 0;
  background: rgba(255, 255, 255, 0.12);
  flex: 0 0 1px;
}

.dock-trail {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  margin-left: auto;
}

.dock-quiet {
  color: var(--muted);
  border-radius: var(--radius-pill);
}

.dock-danger {
  color: #fecaca;
  border-radius: var(--radius-pill);
  font-weight: 600;
}

.dock-danger:hover:not(:disabled) {
  background: rgba(220, 38, 38, 0.18);
  color: #fecaca;
}

.err {
  color: var(--danger);
  margin: 0 0 var(--space-4);
  font-size: var(--text-sm);
}

.toast {
  margin: 0 0 var(--space-4);
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-pill);
  background: var(--frosted);
  color: var(--muted);
  font-size: var(--text-sm);
  animation: rise var(--motion-base) var(--ease-standard);
}

.hero-surface {
  display: flex;
  flex-direction: column;
  gap: var(--space-5);
  margin-bottom: var(--space-6);
}

.log-card {
  margin-bottom: var(--space-8);
}

.log-tabs {
  --log-tab-h: 36px;
  display: inline-flex;
  align-items: stretch;
  max-width: 100%;
  min-height: var(--log-tab-h);
  margin-bottom: var(--space-3);
  border-radius: var(--radius-pill);
  background: rgba(255, 255, 255, 0.06);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.log-rule {
  width: 1px;
  align-self: stretch;
  margin: 8px 0;
  background: rgba(255, 255, 255, 0.12);
  flex: 0 0 1px;
}

.log-tab {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--muted);
  font: inherit;
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  cursor: pointer;
  min-height: var(--log-tab-h);
  padding: 0 16px;
  flex: 1 1 0;
  transition:
    background var(--motion-fast) var(--ease-standard),
    color var(--motion-fast) var(--ease-standard);
}

.log-tab:hover {
  color: var(--fg);
  background: rgba(255, 255, 255, 0.08);
}

.log-tab:focus-visible {
  outline: none;
  box-shadow: inset 0 0 0 2px color-mix(in oklab, var(--accent), transparent 40%);
  z-index: 1;
}

.log-tab.active {
  color: #000;
  background: #fff;
  font-weight: 600;
}

.log-tab.active:hover {
  color: #000;
  background: #f2f2f2;
}

.log {
  margin: 0;
  max-height: min(42vh, 420px);
  overflow: auto;
  padding: var(--space-4);
  border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.03);
  box-shadow: inset 0 0 0 1px var(--border-soft);
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.5;
  color: var(--muted);
  white-space: pre-wrap;
  word-break: break-word;
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes tick {
  from {
    opacity: 0.35;
  }
  to {
    opacity: 1;
  }
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.35;
  }
}

@media (max-width: 809px) {
  .kpi-row {
    grid-template-columns: 1fr;
  }

  .dock {
    position: static;
    border-radius: var(--radius-lg);
    gap: 8px;
  }

  .dock :deep(.dock-primary.btn) {
    flex: 1 1 100%;
  }

  .dock-group {
    flex: 1 1 100%;
  }

  .dock-item {
    flex: 1 1 0;
    text-align: center;
  }

  .dock-trail {
    margin-left: 0;
    width: 100%;
    justify-content: flex-end;
    flex-wrap: wrap;
  }

  h1 {
    font-size: var(--text-xl);
  }
}

@media (prefers-reduced-motion: reduce) {
  .kpi,
  .kpi-num,
  .toast,
  .status[data-tone='live']::before {
    animation: none;
  }
}
</style>
