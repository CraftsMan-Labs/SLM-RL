<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import AppShell from '@/components/shell/AppShell.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCard from '@/components/ui/UiCard.vue'
import UiField from '@/components/ui/UiField.vue'
import UiModal from '@/components/ui/UiModal.vue'
import {
  bakePack,
  createProject,
  getBakeLog,
  getKnobs,
  listGames,
  listPacks,
  listProjects,
  stopProject,
  type KnobSchema,
  type PackInfo,
  type Project,
} from '@/api/projects'
import { ApiError } from '@/api/client'
import { useProfile } from '@/composables/useProfile'
import { workshopDqnUrl, workshopPackUrl } from '@/lib/workshopHf'

const router = useRouter()
const { profile, refresh: refreshProfile } = useProfile()

const projects = ref<Project[]>([])
const games = ref<string[]>([])
const defaultGame = ref('')
const packs = ref<PackInfo[]>([])
const baking = ref(false)
const bakeLog = ref('')
const bakeFoldEl = ref<HTMLDetailsElement | null>(null)
const bakeLogEl = ref<HTMLPreElement | null>(null)
const loading = ref(true)
const creating = ref(false)
const stopping = ref<string | null>(null)
const error = ref<string | null>(null)
const bakeMsg = ref<string | null>(null)

const showCreate = ref(false)
const createError = ref<string | null>(null)
const newName = ref('')
const newGame = ref('')
const knobs = ref<KnobSchema[]>([])
const knobValues = ref<Record<string, unknown>>({})
const datasetUrl = ref('')
const adapterUrl = ref('')
const dqnUrl = ref('')

const bakeGame = ref('')
const bakeAll = ref(false)
const bakeEpisodes = ref(200)
const bakeDqn = ref(10_000)
const bakeQuantile = ref(0.25)
const bakePush = ref('')
const bakePushPrefix = ref('')

const nameValid = computed(() => /^[a-z0-9-]{1,40}$/.test(newName.value.trim()))

const ruleKnobs = computed(() =>
  knobs.value.filter((k) => k.target === 'game' || k.target === 'game.extra'),
)
const trainKnobs = computed(() => {
  const list = knobs.value.filter(
    (k) => k.target === 'run.train' || k.target === 'run.teacher',
  )
  // Rollout budget first — biggest workshop time sink on Docker CPU.
  const rank = (key: string) => (key === 'episodes_per_generation' ? 0 : 1)
  return [...list].sort((a, b) => rank(a.key) - rank(b.key) || a.key.localeCompare(b.key))
})
const monitorKnobs = computed(() => knobs.value.filter((k) => k.target === 'game.monitor'))

const teacherIsDqn = computed(() => knobValues.value.teacher === 'dqn')
const localDqnPack = computed(() =>
  packs.value.find((p) => p.game === newGame.value && p.has_dqn),
)
const dqnUrlRequired = computed(() => teacherIsDqn.value && !localDqnPack.value)

let bakePoll: ReturnType<typeof setInterval> | null = null

function inputType(k: KnobSchema) {
  if (k.type === 'float' || k.type === 'int') return 'number'
  return 'text'
}

async function loadKnobsForGame(game: string, overrides?: Record<string, unknown>) {
  if (!game) {
    knobs.value = []
    knobValues.value = {}
    return
  }
  knobs.value = await getKnobs(game)
  const next: Record<string, unknown> = {}
  for (const k of knobs.value) {
    next[k.key] = overrides?.[k.key] ?? k.default
  }
  knobValues.value = next
}

async function refreshPacks() {
  try {
    const wasBaking = baking.value
    const payload = await listPacks()
    packs.value = payload.packs
    baking.value = payload.baking
    // Always reload from disk — bake.log survives navigation; the in-memory
    // ref does not. Skipping when idle was why finished bake output vanished.
    bakeLog.value = await getBakeLog()
    if (wasBaking && !payload.baking) {
      const pushed = bakeLog.value.match(/\[bake\] pushed ([^\s:]+):/)
      bakeMsg.value = pushed
        ? `Bake finished — pushed to ${pushed[1]}`
        : 'Bake finished — pack is local (no HF push in this run).'
    }
    if (payload.baking && bakeFoldEl.value) {
      bakeFoldEl.value.open = true
    }
    await nextTick()
    if (bakeLogEl.value && payload.baking) {
      bakeLogEl.value.scrollTop = bakeLogEl.value.scrollHeight
    }
  } catch {
    /* non-fatal on projects page */
  }
}

async function refresh() {
  loading.value = true
  error.value = null
  try {
    const [rows, gamePayload] = await Promise.all([listProjects(), listGames()])
    projects.value = rows
    games.value = gamePayload.games
    defaultGame.value = gamePayload.default
    if (!newGame.value) newGame.value = gamePayload.default
    if (!bakeGame.value) bakeGame.value = gamePayload.default
    await refreshPacks()
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load projects'
  } finally {
    loading.value = false
  }
}

function suggestWorkshopUrls() {
  const game = newGame.value
  // All four keeper games have published workshop packs.
  datasetUrl.value = workshopPackUrl(game)
  adapterUrl.value = workshopPackUrl(game)
  dqnUrl.value = workshopDqnUrl(game)
}

async function openCreate() {
  showCreate.value = true
  createError.value = null
  error.value = null
  if (newGame.value) await loadKnobsForGame(newGame.value)
  suggestWorkshopUrls()
}

function closeCreate() {
  if (creating.value) return
  showCreate.value = false
  createError.value = null
}

watch(newGame, async (game) => {
  if (!showCreate.value || !game) return
  await loadKnobsForGame(game)
  suggestWorkshopUrls()
})

watch(
  () => knobValues.value.teacher,
  (teacher) => {
    if (teacher !== 'dqn' || dqnUrl.value.trim()) return
    const dqn = workshopDqnUrl(newGame.value)
    if (dqn) dqnUrl.value = dqn
  },
)

async function onCreate() {
  createError.value = null
  if (!nameValid.value) {
    createError.value = 'Name must be lowercase letters, numbers, or hyphens (1–40 chars).'
    return
  }
  if (!newGame.value) {
    createError.value = 'Pick a game for this project.'
    return
  }
  if (dqnUrlRequired.value && !dqnUrl.value.trim()) {
    createError.value =
      'Teacher is DQN — paste a Hugging Face repo (org/name) that contains dqn.pt, or bake a pack for this game first.'
    return
  }
  creating.value = true
  try {
    const created = await createProject({
      name: newName.value.trim(),
      game: newGame.value,
      knob_values: { ...knobValues.value },
      agent: 'solver',
      dataset_url: datasetUrl.value.trim() || null,
      adapter_url: adapterUrl.value.trim() || null,
      dqn_url: teacherIsDqn.value
        ? dqnUrl.value.trim() || datasetUrl.value.trim() || null
        : dqnUrl.value.trim() || null,
    })
    showCreate.value = false
    await router.push({ name: 'project', params: { name: created.name } })
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      createError.value = 'Busy — another run is already in progress. Try again in a moment.'
    } else {
      createError.value = err instanceof Error ? err.message : 'Could not create project'
    }
  } finally {
    creating.value = false
  }
}

async function onBake() {
  bakeMsg.value = null
  error.value = null
  try {
    const result = await bakePack({
      game: bakeAll.value ? null : bakeGame.value,
      all: bakeAll.value,
      episodes: bakeEpisodes.value,
      dqn_decisions: bakeDqn.value,
      selection_quantile: bakeQuantile.value,
      push: bakePush.value.trim() || null,
      push_prefix: bakePushPrefix.value.trim() || null,
    })
    baking.value = true
    const pushTo = (result as { push?: string | null }).push
    bakeMsg.value = pushTo
      ? `Bake started — will push to ${pushTo} when demos finish. Watch the log below.`
      : 'Bake started (local only) — watch the log below.'
    bakeLog.value = await getBakeLog()
    void refreshProfile()
  } catch (err) {
    error.value =
      err instanceof ApiError && err.status === 409
        ? 'Busy — a bake is already running.'
        : err instanceof Error
          ? err.message
          : 'Bake failed'
  }
}

function statusLabel(p: Project) {
  if (p.status === 'running') return 'Running'
  if (p.status === 'complete') return 'Complete'
  if (p.status === 'stopped') return 'Stopped'
  return 'Ready'
}

function isActive(p: Project) {
  return (p.active_jobs?.length ?? 0) > 0
}

async function onStop(p: Project, ev: Event) {
  ev.stopPropagation()
  ev.preventDefault()
  error.value = null
  stopping.value = p.name
  try {
    await stopProject(p.name)
    await refresh()
  } catch (err) {
    error.value =
      err instanceof ApiError && err.status === 404
        ? 'Nothing running for that project.'
        : err instanceof Error
          ? err.message
          : 'Stop failed'
  } finally {
    stopping.value = null
  }
}

function scoreLabel(p: Project) {
  if (p.mean_score == null) return '—'
  return p.mean_score.toFixed(2)
}

function copyHint(p: PackInfo) {
  return p.hf_repo || (p.repo_hint.includes('/') ? p.repo_hint : p.game || p.slug)
}

function suggestBakePush() {
  const user = profile.value?.hf_username
  if (!user || !bakeGame.value || bakeAll.value) return
  const suggested = `${user}/slm-rl-${bakeGame.value}`
  // Only fill empty / placeholder orgs so we don't clobber a real paste.
  const cur = bakePush.value.trim()
  if (!cur || /^blank\//i.test(cur) || /^your-?org\//i.test(cur)) {
    bakePush.value = suggested
  }
}

watch([bakeGame, () => profile.value?.hf_username], () => {
  suggestBakePush()
})

onMounted(async () => {
  await refresh()
  // Re-open the instructor fold when a persisted bake log is on disk.
  if (bakeLog.value && bakeFoldEl.value) {
    bakeFoldEl.value.open = true
    await nextTick()
    if (bakeLogEl.value) {
      bakeLogEl.value.scrollTop = bakeLogEl.value.scrollHeight
    }
  }
  bakePoll = setInterval(() => {
    void refreshPacks()
  }, 4000)
})

onUnmounted(() => {
  if (bakePoll) clearInterval(bakePoll)
})
</script>

<template>
  <AppShell title="Projects">
    <div class="head">
      <div>
        <p class="eyebrow">Your workspace</p>
        <h1>Projects</h1>
        <p class="sub">
          Load the server once. Bake packs, create projects, and run games here —
          no terminal.
        </p>
      </div>
      <UiButton @click="openCreate">New project</UiButton>
    </div>

    <p v-if="error" class="err" role="alert">{{ error }}</p>
    <p v-if="bakeMsg" class="ok">{{ bakeMsg }}</p>

    <UiCard class="bake">
      <!-- ponytail: native details; open while baking so log stays visible -->
      <details ref="bakeFoldEl" class="bake-fold" :open="baking || undefined">
        <summary class="bake-summary">
          <span class="bake-summary-text">
            <span class="eyebrow">Instructor</span>
            <span class="bake-title">Bake workshop packs</span>
          </span>
          <span class="bake-state">{{
            baking
              ? 'Baking…'
              : bakeLog
                ? packs.length
                  ? `${packs.length} pack${packs.length === 1 ? '' : 's'} · log saved`
                  : 'Log saved'
                : packs.length
                  ? `${packs.length} pack${packs.length === 1 ? '' : 's'}`
                  : 'Optional'
          }}</span>
        </summary>
        <p class="hint">
          Pre-bake teacher demos (and Atari DQN) into local packs. Optional: push public HF
          repos, then attendees paste the dataset URL on a project. Bake output is kept in
          <span class="mono">runs/packs/bake.log</span> (appended per run) — leaving this page
          no longer clears it.
        </p>
        <div class="bake-form">
          <UiField label="Game" for-id="bake-game">
            <select id="bake-game" v-model="bakeGame" :disabled="bakeAll || baking">
              <option v-for="g in games" :key="g" :value="g">{{ g }}</option>
            </select>
          </UiField>
          <label class="check">
            <input v-model="bakeAll" type="checkbox" :disabled="baking" />
            Bake all games
          </label>
          <UiField label="Demo episodes" for-id="bake-ep">
            <input id="bake-ep" v-model.number="bakeEpisodes" type="number" min="1" :disabled="baking" />
          </UiField>
          <UiField label="DQN decisions (Atari)" for-id="bake-dqn" hint="0 skips DQN train">
            <input id="bake-dqn" v-model.number="bakeDqn" type="number" min="0" :disabled="baking" />
          </UiField>
          <UiField
            label="Top quantile"
            for-id="bake-q"
            hint="Keep top fraction of demos by return (0.25 = best 25%). 1.0 keeps all."
          >
            <input
              id="bake-q"
              v-model.number="bakeQuantile"
              type="number"
              min="0.01"
              max="1"
              step="0.05"
              :disabled="baking"
            />
          </UiField>
          <UiField
            label="Push repo (optional)"
            for-id="bake-push"
            hint="Your HF dataset org/name — not BLANK. Empty = local only."
          >
            <input
              id="bake-push"
              v-model="bakePush"
              type="text"
              :placeholder="
                profile?.hf_username
                  ? `${profile.hf_username}/slm-rl-${bakeGame || 'game'}`
                  : 'your-hf-user/slm-rl-boxing'
              "
              :disabled="baking || bakeAll"
              autocomplete="off"
            />
          </UiField>
          <UiField
            label="Push prefix (all games)"
            for-id="bake-prefix"
            hint="org/slm-rl → org/slm-rl-&lt;game&gt;"
          >
            <input
              id="bake-prefix"
              v-model="bakePushPrefix"
              type="text"
              :disabled="baking || !bakeAll"
              autocomplete="off"
            />
          </UiField>
        </div>
        <div class="actions">
          <UiButton :disabled="baking" @click="onBake">
            {{ baking ? 'Baking…' : 'Bake pack' }}
          </UiButton>
        </div>
        <ul v-if="packs.length" class="pack-list">
          <li v-for="p in packs" :key="p.slug">
            <span class="mono">{{ p.game || p.slug }}</span>
            <span class="meta">
              <template v-if="p.n_episodes_raw != null && p.n_episodes_raw !== p.n_episodes">
                {{ p.n_episodes ?? '—' }}/{{ p.n_episodes_raw }} eps
                <template v-if="p.selection_quantile != null">
                  (top {{ Math.round(Number(p.selection_quantile) * 100) }}%)
                </template>
              </template>
              <template v-else>{{ p.n_episodes ?? '—' }} eps</template>
              <template v-if="p.hf_repo">
                · pushed
                <a
                  class="pack-link"
                  :href="`https://huggingface.co/datasets/${p.hf_repo}`"
                  target="_blank"
                  rel="noopener"
                  >{{ p.hf_repo }}</a
                >
              </template>
              <template v-else> · local · paste id: {{ copyHint(p) }}</template>
            </span>
          </li>
        </ul>
        <pre v-if="bakeLog" ref="bakeLogEl" class="log">{{ bakeLog }}</pre>
        <p v-else-if="!baking" class="hint bake-log-empty">No bake log yet — run a pack to see live output here.</p>
      </details>
    </UiCard>

    <UiModal
      :open="showCreate"
      eyebrow="Create"
      title="New project"
      @close="closeCreate"
    >
      <p class="hint">
        Knobs pre-fill from this game’s config when you pick a game. Hover the ⓘ for why each
        default exists. Want different knobs later? Create another project (new run-id).
      </p>
      <p v-if="createError" class="err" role="alert">{{ createError }}</p>
      <form class="create-form" @submit.prevent="onCreate">
        <UiField
          label="Project name"
          for-id="proj-name"
          :hint="
            newName.trim() && !nameValid
              ? 'Invalid — use only lowercase a–z, 0–9, and hyphens (1–40 chars). No spaces or underscores.'
              : 'Required. Lowercase, numbers, hyphens → URL /projects/your-name'
          "
        >
          <input
            id="proj-name"
            v-model="newName"
            placeholder="e.g. tighter-loop"
            autocomplete="off"
            :aria-invalid="Boolean(newName.trim() && !nameValid)"
          />
        </UiField>
        <UiField label="Game" for-id="proj-game" hint="The environment this project trains on.">
          <select id="proj-game" v-model="newGame">
            <option v-for="g in games" :key="g" :value="g">{{ g }}</option>
          </select>
        </UiField>

        <UiField
          label="Dataset pack (Hugging Face)"
          for-id="proj-dataset-url"
          hint="Pre-filled for all keeper games — clear only if you train from a live teacher or local bake."
          info-title="Dataset pack"
          info="Defaults to the workshop dataset (BLANK/slm-rl-<game>). Evolve uses this for gen-1 warm-start. Clear the field only if you will train from a live teacher / local bake."
        >
          <input
            id="proj-dataset-url"
            v-model="datasetUrl"
            type="text"
            placeholder="e.g. BLANK/slm-rl-boxing"
            autocomplete="off"
          />
        </UiField>

        <UiField
          label="SFT model (Hugging Face)"
          for-id="proj-adapter-url"
          hint="Pre-filled workshop LoRA — clear to re-SFT from the dataset pack instead."
          info-title="SFT adapter"
          info="Defaults to the workshop LoRA model (same id as the dataset, with adapter/). Evolve imports it as gen-1 champion and runs GRPO from gen 2 — no reject_sft. Wins over the dataset pack for gen 1 when both are set."
        >
          <input
            id="proj-adapter-url"
            v-model="adapterUrl"
            type="text"
            placeholder="e.g. BLANK/slm-rl-boxing"
            autocomplete="off"
          />
        </UiField>

        <div v-if="ruleKnobs.length" class="knob-block">
          <p class="knob-heading">Rules</p>
          <div class="knobs">
            <UiField
              v-for="k in ruleKnobs"
              :key="k.key"
              :label="k.label"
              :for-id="`ck-${k.key}`"
              :info-title="k.help?.title"
              :info="k.help?.body"
            >
              <select
                v-if="k.type === 'enum' && k.choices"
                :id="`ck-${k.key}`"
                v-model="knobValues[k.key]"
              >
                <option v-for="c in k.choices" :key="c" :value="c">{{ c }}</option>
              </select>
              <input
                v-else
                :id="`ck-${k.key}`"
                v-model.number="knobValues[k.key]"
                :type="inputType(k)"
                :min="k.min"
                :max="k.max"
                :step="k.type === 'float' ? 'any' : 1"
              />
            </UiField>
          </div>
        </div>

        <div v-if="trainKnobs.length" class="knob-block">
          <p class="knob-heading">Training</p>
          <p class="knob-note">
            Defaults: 20 rollout episodes, 20 gate evals, 2 GRPO gens — real
            signal, strict EvalGate (no stubs). Expect longer than a demo cycle.
          </p>
          <div class="knobs">
            <UiField
              v-for="k in trainKnobs"
              :key="k.key"
              :label="k.label"
              :for-id="`ck-${k.key}`"
              :info-title="k.help?.title"
              :info="k.help?.body"
            >
              <select
                v-if="k.type === 'enum' && k.choices"
                :id="`ck-${k.key}`"
                v-model="knobValues[k.key]"
              >
                <option v-for="c in k.choices" :key="c" :value="c">{{ c }}</option>
              </select>
              <input
                v-else
                :id="`ck-${k.key}`"
                v-model.number="knobValues[k.key]"
                :type="inputType(k)"
                :min="k.min"
                :max="k.max"
                :step="k.type === 'float' ? 'any' : 1"
              />
            </UiField>
          </div>
          <UiField
            v-if="teacherIsDqn"
            class="dqn-url-field"
            label="DQN Hugging Face repo"
            for-id="ck-dqn-url"
            :hint="
              dqnUrlRequired
                ? 'Required — public org/name (or full HF URL) whose files include dqn.pt. Downloaded at create.'
                : `Optional — local bake found for ${newGame}. Paste a repo to download instead.`
            "
            info-title="DQN Hugging Face repo"
            info="Public dataset or model repo that contains dqn.pt (from Bake workshop packs → push, or train-dqn upload). On create we download it into runs/packs/ and wire teacher.dqn_checkpoint."
          >
            <input
              id="ck-dqn-url"
              v-model="dqnUrl"
              type="text"
              placeholder="e.g. BLANK/slm-rl-boxing-dqn"
              autocomplete="off"
              :required="dqnUrlRequired"
            />
          </UiField>
        </div>

        <details v-if="monitorKnobs.length" class="monitor">
          <summary>Monitor (advanced)</summary>
          <div class="knobs">
            <UiField
              v-for="k in monitorKnobs"
              :key="k.key"
              :label="k.label"
              :for-id="`ck-${k.key}`"
              :info-title="k.help?.title"
              :info="k.help?.body"
            >
              <input
                :id="`ck-${k.key}`"
                v-model.number="knobValues[k.key]"
                :type="inputType(k)"
                :min="k.min"
                :max="k.max"
                :step="k.type === 'float' ? 'any' : 1"
              />
            </UiField>
          </div>
        </details>

        <div class="actions">
          <p v-if="!nameValid" class="actions-hint">
            {{
              newName.trim()
                ? 'Fix the project name to enable create.'
                : 'Enter a project name above, then create.'
            }}
          </p>
          <UiButton type="submit" :disabled="creating">
            {{ creating ? 'Creating…' : 'Create and open' }}
          </UiButton>
          <UiButton variant="ghost" type="button" :disabled="creating" @click="closeCreate">
            Cancel
          </UiButton>
        </div>
      </form>
    </UiModal>

    <div v-if="loading" class="empty">Loading projects…</div>

    <div v-else-if="projects.length === 0 && !showCreate" class="empty-card">
      <p class="eyebrow">Nothing here yet</p>
      <h2>Create your first project</h2>
      <p>Pick a name, game, and knobs — then open the workspace to run.</p>
      <UiButton @click="openCreate">New project</UiButton>
    </div>

    <ul v-else class="list">
      <li v-for="p in projects" :key="p.name">
        <div class="row-wrap">
          <button
            class="row"
            type="button"
            @click="router.push({ name: 'project', params: { name: p.name } })"
          >
            <div class="row-main">
              <span class="name">{{ p.name }}</span>
              <span class="game">{{ p.game || '—' }}</span>
            </div>
            <div class="row-meta">
              <span class="pill" :data-status="p.status">{{ statusLabel(p) }}</span>
              <span class="score">
                <span class="score-label">mean</span>
                {{ scoreLabel(p) }}
              </span>
              <span class="chev" aria-hidden="true">→</span>
            </div>
          </button>
          <UiButton
            v-if="isActive(p)"
            class="stop-btn"
            variant="danger"
            type="button"
            :disabled="stopping === p.name"
            @click="onStop(p, $event)"
          >
            {{ stopping === p.name ? 'Stopping…' : 'Stop' }}
          </UiButton>
        </div>
      </li>
    </ul>
  </AppShell>
</template>

<style scoped>
.head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--space-6);
  margin-bottom: var(--space-8);
}

.eyebrow {
  margin: 0 0 var(--space-3);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
}

h1 {
  font-size: var(--text-3xl);
  margin-bottom: var(--space-3);
}

.sub {
  max-width: 42ch;
  color: var(--fg-2);
}

.err {
  color: var(--danger);
  margin-bottom: var(--space-4);
  font-size: var(--text-sm);
}

.ok {
  color: var(--success);
  margin-bottom: var(--space-4);
  font-size: var(--text-sm);
}

.bake {
  margin-bottom: var(--space-6);
}

.bake-fold {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.bake-summary {
  list-style: none;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  cursor: pointer;
  user-select: none;
}

.bake-summary::-webkit-details-marker {
  display: none;
}

.bake-summary-text {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.bake-summary .eyebrow {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
}

.bake-title {
  font-size: var(--text-xl);
  letter-spacing: -0.03em;
  font-weight: 500;
  color: var(--fg);
}

.bake-state {
  flex-shrink: 0;
  margin-top: 2px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--meta);
}

.bake-fold[open] .bake-state::after {
  content: ' · hide';
}

.bake-fold:not([open]) .bake-state::after {
  content: ' · show';
}

.hint {
  color: var(--meta);
  font-size: var(--text-sm);
  margin-bottom: var(--space-4);
}

.bake-form {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-4);
  margin-bottom: var(--space-4);
}

.check {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-sm);
  color: var(--muted);
  grid-column: 1 / -1;
}

.create-form {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.knob-block {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.knob-heading {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}

.knob-note {
  margin: 0;
  font-size: var(--text-sm);
  color: var(--muted);
  line-height: 1.4;
}

.knobs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-3);
}

.dqn-url-field {
  margin-top: var(--space-3);
}

.monitor {
  color: var(--muted);
  font-size: var(--text-sm);
}

.monitor summary {
  cursor: pointer;
  margin-bottom: var(--space-3);
}

.monitor .knobs {
  margin-top: var(--space-3);
}

.actions {
  position: sticky;
  bottom: 0;
  z-index: 2;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-3);
  margin-top: var(--space-2);
  padding-top: var(--space-4);
  background: linear-gradient(to top, var(--surface) 70%, transparent);
}

.actions-hint {
  flex: 1 1 100%;
  margin: 0;
  font-size: var(--text-sm);
  color: var(--meta);
}

.pack-list {
  list-style: none;
  margin: var(--space-4) 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.pack-list li {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
  font-size: var(--text-sm);
}

.mono {
  font-family: var(--font-mono);
}

.meta {
  color: var(--meta);
}

.pack-link {
  color: var(--accent);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.log {
  margin: var(--space-4) 0 0;
  max-height: 220px;
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
}

.empty {
  color: var(--meta);
}

.empty-card {
  padding: var(--space-8);
  border-radius: var(--radius-md);
  box-shadow: var(--elev-ring);
  background: var(--surface);
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--space-4);
  max-width: 480px;
}

.empty-card h2 {
  font-size: var(--text-2xl);
}

.empty-card p:not(.eyebrow) {
  color: var(--fg-2);
}

.list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.row-wrap {
  display: flex;
  align-items: stretch;
  gap: var(--space-3);
}

.row {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-5) var(--space-5);
  background: var(--surface);
  border: 0;
  border-radius: var(--radius-md);
  box-shadow: var(--elev-ring);
  cursor: pointer;
  text-align: left;
  transition: box-shadow var(--motion-base) var(--ease-standard);
}

.stop-btn {
  flex-shrink: 0;
  align-self: center;
  min-height: 40px;
  padding: 8px 14px;
}

.row:hover {
  box-shadow:
    0 0 0 1px rgba(0, 153, 255, 0.35),
    var(--elev-raised);
}

.row-main {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.name {
  font-family: var(--font-display);
  font-size: var(--text-lg);
  letter-spacing: -0.03em;
}

.game {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--muted);
}

.row-meta {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  flex-shrink: 0;
}

.pill {
  font-size: 11px;
  padding: 4px 10px;
  border-radius: var(--radius-pill);
  background: var(--frosted);
  color: var(--muted);
}

.pill[data-status='running'] {
  color: var(--accent);
  box-shadow: 0 0 0 1px rgba(0, 153, 255, 0.35);
}

.pill[data-status='complete'] {
  color: var(--success);
}

.pill[data-status='stopped'] {
  color: var(--danger);
}

.score {
  font-variant-numeric: tabular-nums;
  font-size: var(--text-sm);
  color: var(--fg);
  min-width: 4.5rem;
  text-align: right;
}

.score-label {
  display: block;
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--meta);
}

.chev {
  color: var(--meta);
}

@media (max-width: 809px) {
  .head {
    flex-direction: column;
  }

  .bake-form {
    grid-template-columns: 1fr;
  }

  .row {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
