<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { getEvolveMetrics, type EvolveGeneration, type EvolveMetrics } from '@/api/evolve'
import LivePill from '@/components/ui/LivePill.vue'
import UiCard from '@/components/ui/UiCard.vue'
import EpisodeCard from '@/components/watch/EpisodeCard.vue'
import ScreenPanel from '@/components/watch/ScreenPanel.vue'
import { stopAllFrameStreams } from '@/composables/useFrameStream'
import { useWatchStream } from '@/composables/useWatchStream'

const props = defineProps<{
  projectName: string
  /** Scoreboard status from /api/experiments: no_data | running | complete | stopped */
  projectStatus?: string | null
  /** Live job kinds from /api/experiments (e.g. evolve, theater). */
  activeJobs?: string[] | null
}>()

const emit = defineEmits<{
  /** Fired once when evolve dies mid-phase so the parent can toast. */
  evolveDead: [detail: { phase: string; phaseGen: number | null }]
}>()

const { status, episodes, wins, currentGen, episodeCount, lastEventAt, activeEpisode, reset } =
  useWatchStream(() => props.projectName)

const watchingId = ref<string | null>(null)
const watchingGen = ref<number | null>(null)
/** Bumped on every Watch screen click so replay always restarts from step 0. */
const watchRestartToken = ref(0)
/**
 * Inline live stage in the phase guide (user can Close like Watch screen).
 * Starts false and enables shortly after mount so a just-closed theater/studio
 * can release its ALE replay slot before we open another.
 */
const liveScreenOpen = ref(false)
const now = ref(Date.now())
const evolveMetrics = ref<EvolveMetrics | null>(null)
/** Avoid re-toasting the same dead state every poll. */
const evolveDeadNotified = ref(false)
let tick: ReturnType<typeof setInterval> | null = null
let evolvePoll: ReturnType<typeof setInterval> | null = null
let liveScreenArm: ReturnType<typeof setTimeout> | null = null

function clearScreens() {
  watchingId.value = null
  watchingGen.value = null
  liveScreenOpen.value = false
  stopAllFrameStreams()
}

function armLiveScreen() {
  if (liveScreenArm) clearTimeout(liveScreenArm)
  liveScreenArm = setTimeout(() => {
    liveScreenArm = null
    liveScreenOpen.value = true
  }, 400)
}

const runId = computed(() => (props.projectName ? `pg-${props.projectName}` : ''))

const MID_PHASES = new Set([
  'starting',
  'baseline',
  'rollout',
  'rollout_done',
  'train',
  'train_done',
  'eval',
])

/** Process table / metrics say evolve is still collecting or training. */
const jobAlive = computed(() => {
  if ((props.activeJobs ?? []).includes('evolve')) return true
  if (props.projectStatus === 'running') return true
  if (evolveMetrics.value?.running === true) return true
  return false
})

async function refreshEvolveMetrics() {
  if (!runId.value) return
  try {
    evolveMetrics.value = await getEvolveMetrics(runId.value)
  } catch {
    /* no evolve artifacts yet */
  }
}

watch(
  () => props.projectName,
  () => {
    clearScreens()
    watchRestartToken.value = 0
    evolveMetrics.value = null
    evolveDeadNotified.value = false
    reset()
    void refreshEvolveMetrics()
    armLiveScreen()
  },
)

watch(jobAlive, (alive) => {
  if (alive) evolveDeadNotified.value = false
})

onMounted(() => {
  tick = setInterval(() => {
    now.value = Date.now()
  }, 1000)
  void refreshEvolveMetrics()
  evolvePoll = setInterval(() => {
    void refreshEvolveMetrics()
  }, 2000)
  armLiveScreen()
})

onUnmounted(() => {
  if (tick) clearInterval(tick)
  if (evolvePoll) clearInterval(evolvePoll)
  if (liveScreenArm) clearTimeout(liveScreenArm)
  clearScreens()
})

function onWatch(episodeId: string) {
  // One ALE replay stream at a time — close the inline live stage first.
  liveScreenOpen.value = false
  stopAllFrameStreams()
  const ep = episodes.value.find((e) => e.id === episodeId)
  watchingId.value = episodeId
  watchingGen.value = ep?.generation ?? currentGen.value
  watchRestartToken.value += 1
}

function onCloseScreen() {
  watchingId.value = null
  watchingGen.value = null
  stopAllFrameStreams()
}

function onCloseLiveScreen() {
  liveScreenOpen.value = false
  stopAllFrameStreams()
}

function onShowLiveScreen() {
  // Prefer the inline stage over the side Watch panel (same slot budget).
  watchingId.value = null
  watchingGen.value = null
  stopAllFrameStreams()
  liveScreenOpen.value = true
}

function agentLabel(modelId: string | null | undefined): string {
  if (!modelId) return 'unknown agent'
  if (modelId.startsWith('teacher:') || modelId.startsWith('solver')) {
    return `teacher (${modelId.replace(/^teacher:/, '')})`
  }
  if (modelId === 'random') return 'random agent'
  return `model (${modelId})`
}

function ago(ms: number | null): string {
  if (ms == null) return ''
  const s = Math.max(0, Math.floor((now.value - ms) / 1000))
  if (s < 2) return 'just now'
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  return `${m}m ago`
}

const PHASE_LABELS: Record<string, string> = {
  starting: 'Starting',
  baseline: 'Baseline eval',
  rollout: 'Rollout',
  rollout_done: 'Rollout done',
  train: 'Training',
  train_done: 'Train done',
  eval: 'Gate eval',
  promoted: 'Promoted',
  rejected: 'Rejected',
  early_stop: 'Early stop',
}

function latestGate(gens: EvolveGeneration[]): EvolveGeneration | null {
  for (let i = gens.length - 1; i >= 0; i--) {
    const g = gens[i]
    if (g && (g.promoted === true || g.promoted === false)) return g
  }
  return null
}

type PhaseGuide = { title: string; body: string; tone: 'info' | 'ok' | 'warn' | 'idle' | 'danger' }

const evolveProgress = computed(() => {
  const idle = {
    pct: 0,
    phase: '',
    phaseLabel: 'Idle',
    phaseGen: null as number | null,
    gateTone: 'idle' as const,
    gateLabel: 'No gate yet',
    gateDetail: 'Promotion is decided after train + frozen eval.',
    primary: null as number | null,
    indeterminate: false,
    finishedInGen: 0,
    seenInGen: 0,
    processDead: false,
    trainStep: null as number | null,
    trainTotal: null as number | null,
    trainKl: null as number | null,
    trainEntropy: null as number | null,
    trainReward: null as number | null,
    guide: {
      title: 'Waiting',
      body: 'Start Evolve for 2 real GRPO generations. EvalGate promotes only if primary beats the champion.',
      tone: 'idle' as const,
    } satisfies PhaseGuide,
    crashError: null as string | null,
  }
  try {
    const alive = jobAlive.value
    const m = evolveMetrics.value
    const crashError = (!alive && m?.crash_error) || null
    const phaseRaw = m?.phase || (alive ? 'starting' : '')
    let phase = phaseRaw
    const ep = activeEpisode.value
    if (
      alive &&
      ep &&
      !ep.terminated &&
      !ep.truncated &&
      (!phase || phase === 'starting' || phase === 'baseline')
    ) {
      phase = 'rollout'
    }

    const phaseGen =
      m?.phase_generation ?? ep?.generation ?? currentGen.value ?? null

    const gens = Array.isArray(m?.generations) ? m.generations : []
    const gate = latestGate(gens)
    const phaseIsPromote = phase === 'promoted'
    const phaseIsReject = phase === 'rejected'
    const phaseIsEarlyStop = phase === 'early_stop'
    const midPhase = MID_PHASES.has(phase) || (alive && phase === '')
    // Never-started projects have empty phase — not a dead mid-run.
    const everRan =
      Boolean(phase) ||
      gens.length > 0 ||
      Boolean(m?.started_at) ||
      Boolean(crashError)
    const processDead = midPhase && !alive && everRan
    const inFlight = alive && midPhase
    const priorGateStillShowing =
      Boolean(gate) &&
      phaseGen != null &&
      gate!.generation < phaseGen &&
      inFlight &&
      !phaseIsPromote &&
      !phaseIsReject

    let gateTone: 'ok' | 'no' | 'pending' | 'idle' = 'idle'
    let gateLabel = 'No gate yet'
    let gateDetail = 'Promotion is decided after train + frozen eval.'

    if (processDead) {
      gateTone = 'no'
      gateLabel = 'Process dead'
      gateDetail = 'Evolve is not running — last log phase is stale until you restart.'
    } else if (phaseIsPromote) {
      gateTone = 'ok'
      gateLabel = `Promoted · gen ${phaseGen ?? '—'}`
      gateDetail = gate?.gate_reason || 'New champion adopted.'
    } else if (phaseIsReject) {
      gateTone = 'no'
      gateLabel = `Rejected · gen ${phaseGen ?? '—'}`
      gateDetail = gate?.gate_reason || 'Did not beat champion — adapter discarded.'
    } else if (priorGateStillShowing && gate?.promoted === true) {
      gateTone = 'pending'
      gateLabel = `Gen ${phaseGen} in progress`
      gateDetail =
        `Champion is gen ${gate.generation} (warm-start/SFT import is free). ` +
        `This gen still has to train + pass the gate.`
    } else if (priorGateStillShowing && gate?.promoted === false) {
      gateTone = 'pending'
      gateLabel = `Gen ${phaseGen} in progress`
      gateDetail = `Last reject was gen ${gate.generation}; current gen still running.`
    } else if (gate?.promoted === true) {
      gateTone = 'ok'
      gateLabel = `Last gate: promoted gen ${gate.generation}`
      gateDetail = gate.gate_reason || 'Champion updated.'
    } else if (gate?.promoted === false) {
      gateTone = 'no'
      gateLabel = `Last gate: rejected gen ${gate.generation}`
      gateDetail = gate.gate_reason || 'Kept previous champion.'
    } else if (inFlight || phase) {
      gateTone = 'pending'
      gateLabel = 'Gate pending'
      gateDetail =
        phase === 'eval'
          ? 'Frozen eval running — promote/reject next.'
          : phase === 'train' || phase === 'train_done'
            ? 'Training — eval/gate follows.'
            : 'Rollout / train first; gate comes after eval.'
    }

    const basePhase: Record<string, number> = {
      starting: 0.06,
      baseline: 0.12,
      rollout: 0.28,
      rollout_done: 0.55,
      train: 0.7,
      train_done: 0.78,
      eval: 0.9,
      promoted: 1,
      rejected: 1,
    }

    let pct = phase ? (basePhase[phase] ?? 0.1) : 0
    const epList = episodes.value ?? []
    const inGen =
      phaseGen == null ? [] : epList.filter((e) => e.generation === phaseGen)
    const finishedInGen = inGen.filter((e) => e.terminated || e.truncated).length
    const seenInGen = inGen.length
    const trainStep = m?.train_step ?? null
    const trainTotal = m?.train_total_steps ?? null
    const trainPct =
      trainStep != null && trainTotal != null && trainTotal > 0
        ? Math.min(1, trainStep / trainTotal)
        : null

    if (processDead) {
      pct = Math.min(pct, 0.35)
    } else if (phase === 'train' && trainPct != null) {
      // Determinate GRPO bar once TRL prints step N/M.
      pct = 0.55 + 0.22 * trainPct
    } else if (phase === 'rollout' && ep) {
      const step = Number(ep.steps?.[0]?.step_idx ?? 0)
      const softMax = Math.max(80, step + 40)
      pct = 0.18 + 0.32 * Math.min(0.95, step / softMax)
    } else if (phase === 'rollout' && phaseGen != null) {
      pct =
        0.18 + 0.32 * Math.min(0.95, finishedInGen / Math.max(finishedInGen + 2, 6))
    }

    if (!phase && epList.length === 0) {
      pct = alive ? 0.04 : 0
    }

    const phaseLabel = processDead
      ? `Stopped · was ${PHASE_LABELS[phase] || phase}`
      : phase
        ? PHASE_LABELS[phase] || phase
        : 'Idle'
    const currentGenPrimary = gens.find((g) => g.generation === phaseGen)?.primary
    const primary =
      currentGenPrimary ??
      (priorGateStillShowing
        ? null
        : (gate?.primary ??
          [...gens].reverse().find((g) => g.primary != null)?.primary ??
          null))

    let guide: PhaseGuide = idle.guide
    const genBit = phaseGen != null ? `gen ${phaseGen}` : 'this generation'
    if (processDead) {
      const crashBit = crashError
        ? ` Crash: ${crashError}.`
        : ' Dead from Docker restart, Stop, or crash.'
      guide = {
        title: `Evolve stopped — not collecting ${genBit}`,
        body:
          `The evolve process is dead.${crashBit} ` +
          `Last log still said “${PHASE_LABELS[phase] || phase}”; that is stale. ` +
          `Finished in stream before stop: ${finishedInGen}` +
          (seenInGen ? ` · seen ${seenInGen}` : '') +
          `. Click Evolve to resume.`,
        tone: 'danger',
      }
    } else if (phase === 'rollout' || (inFlight && phase === '')) {
      guide = {
        title: `Collecting ${genBit} rollouts`,
        body:
          `Champion is playing (~8–12 min on workshop defaults: 2 episodes, short turns). ` +
          `Finished in stream: ${finishedInGen}` +
          (seenInGen ? ` · seen ${seenInGen}` : '') +
          `. Gate numbers above may still be from an earlier gen.`,
        tone: 'info',
      }
    } else if (phase === 'baseline' || phase === 'starting') {
      guide = {
        title: 'Starting evolve',
        body: 'Loading the model or measuring the stock baseline (~4 min first time). Episode cards appear when the first rollout lands.',
        tone: 'info',
      }
    } else if (phase === 'rollout_done') {
      guide = {
        title: 'Rollout finished — preparing train',
        body: `All play for ${genBit} is in. Next: GRPO training (~4 min on CPU).`,
        tone: 'info',
      }
    } else if (phase === 'train' || phase === 'train_done') {
      const stepBit =
        trainStep != null && trainTotal != null
          ? ` Step ${trainStep}/${trainTotal}.`
          : ''
      guide = {
        title: `GRPO training ${genBit}`,
        body:
          `Updating the LoRA with group-relative advantages (real GRPO).` +
          stepBit +
          ` Gate runs after frozen eval.`,
        tone: 'info',
      }
    } else if (phase === 'eval') {
      guide = {
        title: `Gate exam for ${genBit}`,
        body: 'Frozen eval seeds only (no teacher help). Promote if primary beats the champion; else reject (~8–12 min on workshop defaults).',
        tone: 'warn',
      }
    } else if (phaseIsPromote) {
      guide = {
        title: `${genBit} promoted`,
        body: 'New champion. Next generation starts from this adapter.',
        tone: 'ok',
      }
    } else if (phaseIsEarlyStop) {
      guide = {
        title: 'Stopped after reject streak',
        body: 'Champion unchanged. Re-Evolve when you have better warm-start data or want another try.',
        tone: 'warn',
      }
    } else if (phaseIsReject) {
      guide = {
        title: `${genBit} rejected`,
        body: 'Did not beat the champion. Previous champion stays; remaining generations keep going until a reject streak stops the run.',
        tone: 'warn',
      }
    } else if (alive) {
      guide = {
        title: 'Evolve is running',
        body: 'Real GRPO: ~20 rollout eps + full train epochs + 20-ep gate. Watch the stepper for gen N of M.',
        tone: 'info',
      }
    } else if (gate) {
      guide = {
        title: 'Last evolve finished',
        body: gate.promoted
          ? `Champion is gen ${gate.generation}. Click Evolve again for another RL round.`
          : `Gen ${gate.generation} was not promoted. Champion unchanged — Evolve again to retry.`,
        tone: gate.promoted ? 'ok' : 'warn',
      }
    }

    return {
      pct: Math.round(pct * 100),
      phase,
      phaseLabel,
      phaseGen,
      gateTone,
      gateLabel,
      gateDetail,
      primary,
      // Determinate once we know step N/M; keep sliding only for eval.
      indeterminate: alive && phase === 'eval',
      finishedInGen,
      seenInGen,
      processDead,
      trainStep,
      trainTotal,
      trainKl: m?.train_kl ?? null,
      trainEntropy: m?.train_entropy ?? null,
      trainReward: m?.train_reward ?? null,
      guide,
      crashError,
    }
  } catch {
    return idle
  }
})

const GUIDE_TONE = {
  ok: 'live',
  info: 'live',
  warn: 'warn',
  danger: 'danger',
  idle: 'idle',
} as const

/** Glanceable “what’s happening” — jobs + evolve guide beat raw episode idle. */
const activity = computed(() => {
  const jobs = props.activeJobs ?? []
  if (jobs.includes('theater')) {
    return {
      tone: 'live' as const,
      title: 'Theater — base vs champion',
      detail:
        'Same seeded games twice: stock model first, then the champion. Open Theater to watch both sides.',
    }
  }
  if (jobs.includes('quick')) {
    return {
      tone: 'live' as const,
      title: 'Quick Run — teacher on screen',
      detail: 'No training. The teacher (Hugging Face DQN or heuristic) is playing for you.',
    }
  }

  if (status.value === 'connecting') {
    return {
      tone: 'idle' as const,
      title: 'Connecting…',
      detail: 'Opening the live episode feed.',
    }
  }

  const g = evolveProgress.value.guide
  const ep = activeEpisode.value
  const playing = Boolean(ep && !ep.terminated && !ep.truncated)
  const alive = jobAlive.value
  const pipelineStory =
    evolveProgress.value.processDead ||
    alive ||
    (g.tone !== 'idle' && g.title !== 'Waiting')

  if (pipelineStory) {
    let detail = g.body
    if (playing && ep && alive) {
      const step = ep.steps?.[0]
      detail += ` Right now: seed ${ep.seed ?? '—'}, step ${step?.step_idx ?? '—'}, action ${step?.parsed_action ?? '—'}.`
    } else if (
      lastEventAt.value != null &&
      now.value - lastEventAt.value > 12_000 &&
      alive
    ) {
      detail += ` Quiet for ${ago(lastEventAt.value)} — still connected.`
    }
    return {
      tone: GUIDE_TONE[g.tone],
      title: g.title,
      detail,
    }
  }

  if (!ep) {
    return {
      tone: 'idle' as const,
      title: 'Ready',
      detail: 'Nothing running. Use Run (teacher demo), Evolve (train), or Theater (compare).',
    }
  }

  const who = agentLabel(ep.modelId)
  const gen = ep.generation ?? currentGen.value ?? '—'
  const seed = ep.seed ?? '—'
  const outcome = ep.outcome || (ep.truncated ? 'truncated' : 'done')
  return {
    tone: 'idle' as const,
    title: `Last game finished (${outcome})`,
    detail: `Gen ${gen}, seed ${seed}, ${who}. Start Evolve, Run, or Theater for the next chapter.`,
  }
})

watch(
  () => evolveProgress.value.processDead,
  (dead) => {
    if (!dead || evolveDeadNotified.value) return
    evolveDeadNotified.value = true
    emit('evolveDead', {
      phase: evolveProgress.value.phase,
      phaseGen: evolveProgress.value.phaseGen,
    })
  },
)

/** Open episode currently being written — drives the inline Atari stage. */
const liveFollowEpisode = computed(() => {
  if (!jobAlive.value || evolveProgress.value.processDead) return null
  const phase = evolveProgress.value.phase
  if (phase && !MID_PHASES.has(phase) && phase !== '') return null
  const ep = activeEpisode.value
  if (!ep || ep.terminated || ep.truncated) return null
  return ep
})

const liveHud = computed(() => {
  const ep = liveFollowEpisode.value
  if (!ep) return null
  const step = ep.steps?.[0]
  const action = step?.parsed_action ?? '—'
  const idx = step?.step_idx ?? '—'
  const seed = ep.seed ?? '—'
  return `step ${idx} · ${action} · seed ${seed}`
})

const showLiveScreen = computed(
  () => Boolean(liveFollowEpisode.value) && liveScreenOpen.value,
)

/** Latest promoted / registry champion gen for the header chip. */
const championGen = computed(() => {
  const fromRegistry = evolveMetrics.value?.champion
  if (typeof fromRegistry === 'number' && Number.isFinite(fromRegistry)) {
    return fromRegistry
  }
  const gens = evolveMetrics.value?.generations
  if (!Array.isArray(gens)) return null
  let best: number | null = null
  for (const g of gens) {
    if (g?.promoted === true) best = g.generation
  }
  return best
})

/** "Generation 3 of 5" + stepper cells for the run plan. */
const runPlan = computed(() => {
  const m = evolveMetrics.value
  const target = m?.target_generations ?? null
  const start = m?.start_generation ?? null
  const end = m?.end_generation ?? null
  const phaseGen = evolveProgress.value.phaseGen
  const gens = Array.isArray(m?.generations) ? m!.generations : []
  const byGen = new Map(gens.map((g) => [g.generation, g]))

  // Index within this evolve call (1-based), not absolute gen number.
  let index: number | null = null
  if (phaseGen != null && start != null) {
    index = Math.max(1, phaseGen - start + 1)
  } else if (phaseGen != null && phaseGen > 0) {
    index = phaseGen
  }

  const cells: { gen: number; state: 'done' | 'current' | 'pending' | 'promoted' | 'rejected' }[] =
    []
  if (start != null && end != null && end >= start) {
    for (let g = start; g <= end; g++) {
      const summary = byGen.get(g)
      let state: 'done' | 'current' | 'pending' | 'promoted' | 'rejected' = 'pending'
      if (phaseGen != null && g === phaseGen && jobAlive.value) {
        state = 'current'
      } else if (summary?.promoted === true) {
        state = 'promoted'
      } else if (summary?.promoted === false) {
        state = 'rejected'
      } else if (phaseGen != null && g < phaseGen) {
        state = 'done'
      }
      cells.push({ gen: g, state })
    }
  }

  return { target, start, end, index, cells }
})

function fmtDuration(sec: number): string {
  if (sec < 60) return `${Math.max(0, Math.floor(sec))}s`
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return s ? `${m}m ${s}s` : `${m}m`
}

/** Elapsed + rough ETA from finished gens (falls back to ~5 min/gen). */
const runClock = computed(() => {
  const m = evolveMetrics.value
  const started = m?.started_at
  if (started == null) return null
  const elapsedSec = Math.max(0, (now.value - started * 1000) / 1000)
  const target = m?.target_generations
  const start = m?.start_generation
  const phaseGen = evolveProgress.value.phaseGen
  const gens = Array.isArray(m?.generations) ? m!.generations : []
  const finished = gens.filter((g) => g.promoted === true || g.promoted === false).length

  // ponytail: rough ETA for 20-ep rollout + real GRPO; refine when we have samples.
  const DEFAULT_GEN_SEC = 45 * 60
  const avgSec = finished > 0 ? elapsedSec / finished : DEFAULT_GEN_SEC
  let remaining: number | null = null
  if (target != null && jobAlive.value) {
    const doneIdx =
      phaseGen != null && start != null
        ? Math.max(0, phaseGen - start) // current gen not finished yet
        : finished
    remaining = Math.max(0, (target - doneIdx) * avgSec)
  }
  return {
    elapsed: fmtDuration(elapsedSec),
    remaining: remaining != null ? fmtDuration(remaining) : null,
  }
})

const trainHud = computed(() => {
  const p = evolveProgress.value
  if (p.phase !== 'train' && p.phase !== 'train_done') return null
  const bits: string[] = []
  if (p.trainStep != null && p.trainTotal != null) {
    bits.push(`step ${p.trainStep}/${p.trainTotal}`)
  }
  if (p.trainKl != null) bits.push(`kl ${p.trainKl.toFixed(2)}`)
  if (p.trainEntropy != null) bits.push(`entropy ${p.trainEntropy.toFixed(2)}`)
  if (p.trainReward != null) bits.push(`reward ${p.trainReward.toFixed(2)}`)
  return bits.length ? bits.join(' · ') : null
})

</script>

<template>
  <div class="live-stack">
    <div
      class="status-ticker"
      :data-tone="activity.tone"
      role="status"
      aria-live="polite"
    >
      <span class="status-ticker-now" aria-hidden="true">Now</span>
      <div class="status-ticker-copy">
        <p class="status-ticker-title">{{ activity.title }}</p>
        <p class="status-ticker-detail">{{ activity.detail }}</p>
      </div>
    </div>

  <UiCard>
    <template #header>
      <div class="card-head">
        <div>
          <p class="eyebrow">Live</p>
          <h2>Episode stream</h2>
        </div>
        <div class="head-meta">
          <span
            v-if="championGen != null"
            class="champion-chip"
            title="Latest promoted champion generation"
          >
            Champion gen {{ championGen }}
          </span>
          <span v-else class="champion-chip idle" title="No promoted champion yet">
            No champion yet
          </span>
          <LivePill :status="status" />
        </div>
      </div>
    </template>

    <div
      class="progress-block"
      :data-gate="evolveProgress.gateTone"
      role="group"
      aria-label="Evolve progress and gate"
    >
      <div class="progress-meta">
        <span class="progress-phase">
          <template v-if="runPlan.index != null && runPlan.target != null">
            RL gen {{ runPlan.index }} of {{ runPlan.target }}
            <template v-if="evolveProgress.phaseGen != null">
              (abs {{ evolveProgress.phaseGen }})
            </template>
            <template v-if="evolveProgress.phaseLabel"> · </template>
          </template>
          {{ evolveProgress.phaseLabel }}
          <template v-if="evolveProgress.phaseGen != null && runPlan.target == null">
            · gen {{ evolveProgress.phaseGen }}
          </template>
          <template v-if="evolveProgress.primary != null">
            · primary {{ Number(evolveProgress.primary).toFixed(2) }}
          </template>
        </span>
        <span class="gate-chip" :data-gate="evolveProgress.gateTone">
          {{ evolveProgress.gateLabel }}
        </span>
      </div>
      <ol v-if="runPlan.cells.length" class="gen-stepper" aria-label="Generation plan">
        <li
          v-for="cell in runPlan.cells"
          :key="cell.gen"
          class="gen-step"
          :data-state="cell.state"
          :title="`gen ${cell.gen}: ${cell.state}`"
        >
          <span class="gen-step-mark" aria-hidden="true">
            {{
              cell.state === 'promoted'
                ? '✓'
                : cell.state === 'rejected'
                  ? '×'
                  : cell.state === 'current'
                    ? '●'
                    : cell.state === 'done'
                      ? '·'
                      : '○'
            }}
          </span>
          <span class="gen-step-label">{{ cell.gen }}</span>
        </li>
      </ol>
      <div
        class="progress-track"
        role="progressbar"
        :aria-valuenow="evolveProgress.pct"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-label="`${evolveProgress.phaseLabel} ${evolveProgress.pct}%`"
      >
        <div
          class="progress-fill"
          :class="{ indeterminate: evolveProgress.indeterminate }"
          :style="
            evolveProgress.indeterminate ? undefined : { width: `${evolveProgress.pct}%` }
          "
          :data-gate="evolveProgress.gateTone"
        />
      </div>
      <p v-if="trainHud" class="train-hud">{{ trainHud }}</p>
      <p class="progress-detail">
        {{ evolveProgress.gateDetail }}
        <template v-if="runClock">
          · {{ runClock.elapsed }} elapsed
          <template v-if="runClock.remaining">
            · ~{{ runClock.remaining }} left
          </template>
        </template>
      </p>
    </div>

    <details
      class="phase-guide"
      :data-tone="evolveProgress.guide.tone"
      :class="{ 'has-live': showLiveScreen }"
      open
    >
      <summary class="phase-guide-summary">
        <span class="phase-guide-mark" aria-hidden="true">
          {{ evolveProgress.guide.tone === 'danger' ? '!' : 'i' }}
        </span>
        <span class="phase-guide-title">{{ evolveProgress.guide.title }}</span>
        <span class="phase-guide-chevron" aria-hidden="true" />
      </summary>
      <div class="phase-guide-content">
        <div class="phase-guide-copy">
          <p class="phase-guide-body" role="status" aria-live="polite">
            {{ evolveProgress.guide.body }}
          </p>
          <button
            v-if="liveFollowEpisode && !liveScreenOpen"
            type="button"
            class="show-live"
            @click="onShowLiveScreen"
          >
            Show live screen
          </button>
        </div>
        <ScreenPanel
          v-if="showLiveScreen && liveFollowEpisode"
          :key="`live-${liveFollowEpisode.id}`"
          embedded
          :project-name="projectName"
          :episode-id="liveFollowEpisode.id"
          :gen="liveFollowEpisode.generation"
          :restart-token="0"
          :hud="liveHud"
          @close="onCloseLiveScreen"
        />
      </div>
    </details>

    <div class="stats">
      <span>gen <strong>{{ currentGen ?? '—' }}</strong></span>
      <span>episodes <strong>{{ episodeCount }}</strong></span>
      <span>wins <strong>{{ wins }}</strong></span>
      <span v-if="lastEventAt">updated <strong>{{ ago(lastEventAt) }}</strong></span>
    </div>

    <div class="layout" :class="{ split: watchingId }">
      <div class="feed">
        <p v-if="!episodes.length" class="empty">
          Waiting for rollout events — run a game or evolve to see episodes here.
        </p>
        <EpisodeCard
          v-for="ep in episodes"
          :key="ep.id"
          :episode="ep"
          @watch="onWatch"
        />
      </div>
      <ScreenPanel
        v-if="watchingId"
        :key="`${watchingId}-${watchRestartToken}`"
        :project-name="projectName"
        :episode-id="watchingId"
        :gen="watchingGen"
        :restart-token="watchRestartToken"
        @close="onCloseScreen"
      />
    </div>
  </UiCard>
  </div>
</template>

<style scoped>
.card-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--space-4);
  margin-bottom: var(--space-4);
}

.head-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: var(--space-2);
}

.champion-chip {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: var(--radius-pill);
  color: var(--ok, #34d399);
  background: rgba(52, 211, 153, 0.1);
  box-shadow: inset 0 0 0 1px rgba(52, 211, 153, 0.4);
}

.champion-chip.idle {
  color: var(--muted);
  background: rgba(255, 255, 255, 0.03);
  box-shadow: inset 0 0 0 1px var(--border-soft);
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
  font-size: var(--text-xl);
  letter-spacing: -0.03em;
  margin: 0;
}

.live-stack {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.status-ticker {
  display: flex;
  gap: var(--space-3);
  align-items: flex-start;
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-md);
  background: color-mix(in srgb, var(--surface) 92%, var(--accent));
  box-shadow: inset 0 0 0 1px var(--border-soft);
}

.status-ticker[data-tone='live'] {
  box-shadow: inset 0 0 0 1px rgba(0, 153, 255, 0.4);
}

.status-ticker[data-tone='warn'] {
  box-shadow: inset 0 0 0 1px rgba(234, 179, 8, 0.4);
}

.status-ticker[data-tone='danger'] {
  background: rgba(248, 113, 113, 0.1);
  box-shadow: inset 0 0 0 1px rgba(220, 38, 38, 0.45);
}

.status-ticker-now {
  flex: 0 0 auto;
  margin-top: 2px;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
}

.status-ticker[data-tone='danger'] .status-ticker-now {
  color: var(--danger, #f87171);
}

.status-ticker-copy {
  min-width: 0;
  flex: 1;
}

.status-ticker-title {
  margin: 0 0 4px;
  font-size: var(--text-sm);
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--fg, var(--text));
  line-height: 1.3;
}

.status-ticker-detail {
  margin: 0;
  font-size: var(--text-sm);
  color: var(--muted);
  line-height: 1.4;
}

.progress-block {
  margin: calc(-1 * var(--space-2)) 0 var(--space-4);
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.02);
  box-shadow: inset 0 0 0 1px var(--border-soft);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.progress-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2) var(--space-3);
}

.gen-stepper {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.gen-step {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.04em;
  padding: 2px 8px;
  border-radius: 999px;
  color: var(--muted);
  box-shadow: inset 0 0 0 1px var(--border-soft);
}

.gen-step[data-state='current'] {
  color: var(--accent);
  box-shadow: inset 0 0 0 1px rgba(0, 153, 255, 0.5);
}

.gen-step[data-state='promoted'] {
  color: var(--ok, #34d399);
  box-shadow: inset 0 0 0 1px rgba(52, 211, 153, 0.45);
}

.gen-step[data-state='rejected'] {
  color: var(--danger, #f87171);
  box-shadow: inset 0 0 0 1px rgba(248, 113, 113, 0.45);
}

.gen-step[data-state='done'] {
  color: var(--fg-2);
}

.train-hud {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.04em;
  color: var(--accent);
}

.progress-phase {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--fg-2);
}

.gate-chip {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 999px;
  box-shadow: inset 0 0 0 1px var(--border-soft);
  color: var(--muted);
}

.gate-chip[data-gate='ok'] {
  color: var(--ok, #34d399);
  box-shadow: inset 0 0 0 1px rgba(52, 211, 153, 0.45);
}

.gate-chip[data-gate='no'] {
  color: var(--danger, #f87171);
  box-shadow: inset 0 0 0 1px rgba(248, 113, 113, 0.45);
}

.gate-chip[data-gate='pending'] {
  color: var(--accent);
  box-shadow: inset 0 0 0 1px rgba(0, 153, 255, 0.4);
}

.progress-track {
  height: 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  width: 0%;
  border-radius: inherit;
  background: linear-gradient(90deg, rgba(0, 153, 255, 0.55), rgba(0, 153, 255, 0.95));
  transition: width 0.35s ease;
}

.progress-fill[data-gate='ok'] {
  background: linear-gradient(90deg, rgba(52, 211, 153, 0.45), rgba(52, 211, 153, 0.95));
}

.progress-fill[data-gate='no'] {
  background: linear-gradient(90deg, rgba(248, 113, 113, 0.4), rgba(248, 113, 113, 0.9));
}

.progress-fill.indeterminate {
  width: 36%;
  animation: progress-slide 1.2s ease-in-out infinite;
}

.progress-detail {
  margin: 0;
  font-size: var(--text-xs);
  color: var(--meta);
  line-height: 1.4;
}

.phase-guide {
  margin: calc(-1 * var(--space-2)) 0 var(--space-4);
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-md);
  background: rgba(0, 153, 255, 0.06);
  box-shadow: inset 0 0 0 1px rgba(0, 153, 255, 0.28);
}

.phase-guide[data-tone='ok'] {
  background: rgba(52, 211, 153, 0.06);
  box-shadow: inset 0 0 0 1px rgba(52, 211, 153, 0.35);
}

.phase-guide[data-tone='warn'] {
  background: rgba(234, 179, 8, 0.08);
  box-shadow: inset 0 0 0 1px rgba(234, 179, 8, 0.35);
}

.phase-guide[data-tone='idle'] {
  background: rgba(255, 255, 255, 0.02);
  box-shadow: inset 0 0 0 1px var(--border-soft);
}

.phase-guide[data-tone='danger'] {
  background: rgba(248, 113, 113, 0.1);
  box-shadow: inset 0 0 0 1px rgba(248, 113, 113, 0.45);
}

.phase-guide-summary {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  cursor: pointer;
  list-style: none;
  user-select: none;
}

.phase-guide-summary::-webkit-details-marker {
  display: none;
}

.phase-guide-mark {
  flex-shrink: 0;
  width: 20px;
  height: 20px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  font-style: italic;
  color: var(--accent);
  box-shadow: inset 0 0 0 1px rgba(0, 153, 255, 0.45);
}

.phase-guide[data-tone='ok'] .phase-guide-mark {
  color: var(--ok, #34d399);
  box-shadow: inset 0 0 0 1px rgba(52, 211, 153, 0.5);
}

.phase-guide[data-tone='warn'] .phase-guide-mark {
  color: var(--warn);
  box-shadow: inset 0 0 0 1px rgba(234, 179, 8, 0.5);
}

.phase-guide[data-tone='danger'] .phase-guide-mark {
  color: var(--danger, #f87171);
  font-style: normal;
  box-shadow: inset 0 0 0 1px rgba(248, 113, 113, 0.55);
}

.phase-guide-title {
  flex: 1;
  min-width: 0;
  margin: 0;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--fg);
}

.phase-guide-chevron {
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  border-right: 2px solid var(--muted);
  border-bottom: 2px solid var(--muted);
  transform: rotate(45deg);
  transition: transform 0.15s ease;
  margin-top: -4px;
}

.phase-guide[open] .phase-guide-chevron {
  transform: rotate(225deg);
  margin-top: 2px;
}

.phase-guide-content {
  margin-top: var(--space-2);
  padding-left: calc(20px + var(--space-3));
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-3);
  align-items: start;
}

.phase-guide.has-live .phase-guide-content {
  grid-template-columns: minmax(0, 1fr) minmax(220px, 360px);
}

.phase-guide-copy {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  min-width: 0;
}

.phase-guide-body {
  margin: 0;
  font-size: var(--text-sm);
  color: var(--muted);
  line-height: 1.45;
}

.show-live {
  align-self: flex-start;
  margin: 0;
  padding: 4px 12px;
  min-height: 32px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  box-shadow: inset 0 0 0 1px var(--border-soft);
  color: var(--fg);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  letter-spacing: 0.04em;
  cursor: pointer;
}

.show-live:hover {
  box-shadow: inset 0 0 0 1px var(--accent);
  color: var(--accent);
}

@media (max-width: 809px) {
  .phase-guide.has-live .phase-guide-content {
    grid-template-columns: 1fr;
  }
}

@keyframes progress-slide {
  0% {
    transform: translateX(-120%);
  }
  100% {
    transform: translateX(320%);
  }
}

.stats {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-4);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--meta);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: var(--space-4);
}

.stats strong {
  color: var(--fg);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  margin-left: 4px;
}

.layout {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-5);
  align-items: start;
}

.layout.split {
  grid-template-columns: minmax(0, 1fr) minmax(240px, 320px);
}

.feed {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  max-height: 560px;
  overflow: auto;
  padding-right: 2px;
}

.empty {
  margin: 0;
  color: var(--meta);
  font-size: var(--text-sm);
}

@media (max-width: 809px) {
  .layout.split {
    grid-template-columns: 1fr;
  }

  .feed {
    max-height: 420px;
  }
}
</style>
