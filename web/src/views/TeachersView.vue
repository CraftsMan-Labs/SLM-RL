<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import AppShell from '@/components/shell/AppShell.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCard from '@/components/ui/UiCard.vue'
import {
  getDqnLog,
  getDqnMetrics,
  listDqnJobs,
  stopDqnJob,
  type DqnJob,
  type DqnMetricPoint,
  type DqnMetrics,
} from '@/api/dqn'
import { ApiError } from '@/api/client'

const jobs = ref<DqnJob[]>([])
const selected = ref('')
const metrics = ref<DqnMetrics | null>(null)
const logText = ref('')
const error = ref<string | null>(null)
const stopping = ref(false)
const logEl = ref<HTMLPreElement | null>(null)

let poll: ReturnType<typeof setInterval> | null = null

const activeJob = computed(() => jobs.value.find((j) => j.game === selected.value) ?? null)

function downsample(points: DqnMetricPoint[], max = 240): DqnMetricPoint[] {
  if (points.length <= max) return points
  const step = Math.ceil(points.length / max)
  const out: DqnMetricPoint[] = []
  for (let i = 0; i < points.length; i += step) out.push(points[i]!)
  const last = points[points.length - 1]!
  if (out[out.length - 1] !== last) out.push(last)
  return out
}

function seriesPath(
  points: DqnMetricPoint[],
  key: 'mean_ep_reward' | 'loss',
  width: number,
  height: number,
  pad = 8,
): { d: string; min: number; max: number } {
  const vals = points
    .map((p) => p[key])
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (!vals.length || points.length < 2) return { d: '', min: 0, max: 1 }
  let min = Math.min(...vals)
  let max = Math.max(...vals)
  if (min === max) {
    min -= 1
    max += 1
  }
  const xs = points.map((p) => Number(p.decisions ?? 0))
  const x0 = Math.min(...xs)
  const x1 = Math.max(...xs) || 1
  const innerW = width - pad * 2
  const innerH = height - pad * 2
  const coords = points
    .map((p) => {
      const yv = p[key]
      if (typeof yv !== 'number' || !Number.isFinite(yv)) return null
      const x = pad + ((Number(p.decisions ?? 0) - x0) / (x1 - x0 || 1)) * innerW
      const y = pad + (1 - (yv - min) / (max - min)) * innerH
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .filter((c): c is string => c != null)
  return { d: coords.length ? `M ${coords.join(' L ')}` : '', min, max }
}

const trainPts = computed(() => downsample(metrics.value?.train ?? []))
const evalPts = computed(() => downsample(metrics.value?.eval ?? []))

const rewardChart = computed(() => {
  const w = 640
  const h = 180
  const train = seriesPath(trainPts.value, 'mean_ep_reward', w, h)
  const ev = seriesPath(evalPts.value, 'mean_ep_reward', w, h)
  const min = Math.min(train.min, ev.min)
  const max = Math.max(train.max, ev.max)
  return { w, h, train: train.d, eval: ev.d, min, max }
})

const lossChart = computed(() => {
  const w = 640
  const h = 140
  const loss = seriesPath(trainPts.value, 'loss', w, h)
  return { w, h, d: loss.d, min: loss.min, max: loss.max }
})

async function refresh() {
  error.value = null
  try {
    const { jobs: next } = await listDqnJobs()
    jobs.value = next
    if (!selected.value && next.length) {
      const running = next.find((j) => j.running)
      selected.value = running?.game ?? next[0]!.game
    } else if (selected.value && !next.some((j) => j.game === selected.value) && next.length) {
      selected.value = next[0]!.game
    }
    if (selected.value) {
      metrics.value = await getDqnMetrics(selected.value)
      logText.value = await getDqnLog(selected.value)
      await nextTick()
      if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load DQN jobs'
  }
}

async function onStop() {
  if (!selected.value) return
  stopping.value = true
  error.value = null
  try {
    await stopDqnJob(selected.value)
    await refresh()
  } catch (err) {
    error.value =
      err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'Stop failed'
  } finally {
    stopping.value = false
  }
}

watch(selected, () => {
  void refresh()
})

onMounted(() => {
  void refresh()
  poll = setInterval(() => void refresh(), 2500)
})

onUnmounted(() => {
  if (poll) clearInterval(poll)
})

function fmt(n: number | null | undefined, digits = 3) {
  if (n == null || Number.isNaN(n)) return '—'
  return Number(n).toFixed(digits)
}
</script>

<template>
  <AppShell title="Teachers">
    <section class="hero">
      <h1>DQN teacher monitor</h1>
      <p class="lede">
        Live train + validation curves from <span class="mono">train-dqn</span>. Parses the text
        log for in-flight jobs; new runs also write
        <span class="mono">*.metrics.jsonl</span>.
      </p>
    </section>

    <p v-if="error" class="err">{{ error }}</p>

    <UiCard v-if="!jobs.length" class="empty">
      <p>No DQN jobs found under <span class="mono">runs/teachers/</span> or
        <span class="mono">logs/train-dqn-*.log</span>.</p>
    </UiCard>

    <template v-else>
      <div class="toolbar">
        <label class="pick">
          <span>Game</span>
          <select v-model="selected">
            <option v-for="j in jobs" :key="j.game" :value="j.game">
              {{ j.game }}{{ j.running ? ' · live' : '' }}
            </option>
          </select>
        </label>
        <span v-if="activeJob" class="pill" :data-live="activeJob.running">
          {{ activeJob.running ? `running · pid ${activeJob.pid}` : 'idle' }}
        </span>
        <UiButton
          v-if="activeJob?.running"
          variant="ghost"
          :disabled="stopping"
          @click="onStop"
        >
          {{ stopping ? 'Stopping…' : 'Stop' }}
        </UiButton>
      </div>

      <div class="stats">
        <div class="stat">
          <span class="k">Decisions</span>
          <span class="v mono">{{ activeJob?.last_decisions ?? '—' }}</span>
        </div>
        <div class="stat">
          <span class="k">Train reward</span>
          <span class="v mono">{{ fmt(activeJob?.last_train_reward) }}</span>
        </div>
        <div class="stat">
          <span class="k">Eval reward</span>
          <span class="v mono">{{ fmt(activeJob?.last_eval_reward) }}</span>
        </div>
        <div class="stat">
          <span class="k">Loss</span>
          <span class="v mono">{{ fmt(activeJob?.last_loss, 4) }}</span>
        </div>
      </div>

      <div class="charts">
        <UiCard class="chart-card">
          <header class="chart-head">
            <h2>Episode return</h2>
            <span class="legend">
              <i class="swatch train" /> train
              <i class="swatch eval" /> eval
            </span>
          </header>
          <svg
            class="chart"
            :viewBox="`0 0 ${rewardChart.w} ${rewardChart.h}`"
            preserveAspectRatio="none"
            role="img"
            aria-label="Train and eval episode return"
          >
            <path v-if="rewardChart.train" class="line train" :d="rewardChart.train" />
            <path v-if="rewardChart.eval" class="line eval" :d="rewardChart.eval" />
            <text v-if="!rewardChart.train && !rewardChart.eval" x="16" y="32" class="empty-svg">
              waiting for log points…
            </text>
          </svg>
          <div class="range mono">
            {{ fmt(rewardChart.min) }} → {{ fmt(rewardChart.max) }}
          </div>
        </UiCard>

        <UiCard class="chart-card">
          <header class="chart-head">
            <h2>TD loss</h2>
          </header>
          <svg
            class="chart"
            :viewBox="`0 0 ${lossChart.w} ${lossChart.h}`"
            preserveAspectRatio="none"
            role="img"
            aria-label="Training loss"
          >
            <path v-if="lossChart.d" class="line loss" :d="lossChart.d" />
            <text v-if="!lossChart.d" x="16" y="32" class="empty-svg">waiting for loss…</text>
          </svg>
          <div class="range mono">{{ fmt(lossChart.min, 4) }} → {{ fmt(lossChart.max, 4) }}</div>
        </UiCard>
      </div>

      <UiCard class="log-card">
        <header class="chart-head">
          <h2>Training log</h2>
          <span v-if="activeJob?.log_path" class="path mono">{{ activeJob.log_path }}</span>
        </header>
        <pre ref="logEl" class="log">{{ logText || 'No log yet.' }}</pre>
      </UiCard>
    </template>
  </AppShell>
</template>

<style scoped>
.hero {
  margin-bottom: var(--space-6);
  animation: rise 0.45s var(--ease-standard) both;
}

h1 {
  margin: 0;
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  letter-spacing: var(--tracking-display);
  line-height: var(--leading-tight);
}

.lede {
  max-width: 52ch;
  margin: var(--space-3) 0 0;
  color: var(--fg-2);
  font-size: var(--text-base);
}

.mono {
  font-family: var(--font-mono);
  font-size: 0.92em;
}

.err {
  color: var(--danger);
  margin-bottom: var(--space-4);
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-3);
  margin-bottom: var(--space-5);
}

.pick {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--meta);
  font-size: var(--text-sm);
}

.pick select {
  background: var(--surface);
  color: var(--fg);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-pill);
  padding: 8px 14px;
}

.pill {
  font-size: var(--text-xs);
  color: var(--meta);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-pill);
  padding: 6px 12px;
}

.pill[data-live='true'] {
  color: var(--accent);
  border-color: color-mix(in oklab, var(--accent), transparent 50%);
  box-shadow: 0 0 0 1px color-mix(in oklab, var(--accent), transparent 80%);
}

.stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--space-3);
  margin-bottom: var(--space-5);
}

.stat {
  padding: var(--space-4);
  border-radius: var(--radius-md);
  background: var(--surface);
  box-shadow: var(--elev-ring);
  animation: rise 0.5s var(--ease-standard) both;
}

.stat:nth-child(2) {
  animation-delay: 40ms;
}
.stat:nth-child(3) {
  animation-delay: 80ms;
}
.stat:nth-child(4) {
  animation-delay: 120ms;
}

.stat .k {
  display: block;
  color: var(--meta);
  font-size: var(--text-xs);
  margin-bottom: var(--space-2);
}

.stat .v {
  font-size: var(--text-xl);
  color: var(--fg);
}

.charts {
  display: grid;
  gap: var(--space-4);
  margin-bottom: var(--space-5);
}

.chart-card,
.log-card {
  padding: var(--space-4);
}

.chart-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
}

.chart-head h2 {
  margin: 0;
  font-size: var(--text-sm);
  font-weight: 600;
  color: var(--fg-2);
}

.legend {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--meta);
  font-size: var(--text-xs);
}

.swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 999px;
}

.swatch.train {
  background: var(--accent);
}
.swatch.eval {
  background: #34d399;
}

.chart {
  width: 100%;
  height: 180px;
  display: block;
  background:
    linear-gradient(180deg, rgba(0, 153, 255, 0.06), transparent 40%),
    repeating-linear-gradient(
      90deg,
      transparent,
      transparent 39px,
      rgba(255, 255, 255, 0.03) 40px
    );
  border-radius: var(--radius-sm);
}

.line {
  fill: none;
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}

.line.train {
  stroke: var(--accent);
}
.line.eval {
  stroke: #34d399;
  stroke-dasharray: 4 3;
}
.line.loss {
  stroke: #f59e0b;
}

.empty-svg {
  fill: var(--meta);
  font-size: 12px;
  font-family: var(--font-body);
}

.range,
.path {
  margin-top: var(--space-2);
  color: var(--meta);
  font-size: var(--text-xs);
}

.log {
  margin: 0;
  max-height: 320px;
  overflow: auto;
  padding: var(--space-3);
  background: #050505;
  border-radius: var(--radius-sm);
  color: var(--fg-2);
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.45;
  white-space: pre-wrap;
}

.empty {
  padding: var(--space-6);
  color: var(--fg-2);
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

@media (max-width: 720px) {
  .stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
