<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import AppShell from '@/components/shell/AppShell.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCard from '@/components/ui/UiCard.vue'
import {
  getEvolveLog,
  getEvolveMetrics,
  listEvolveJobs,
  stopEvolveJob,
  type EvolveJob,
  type EvolveMetricPoint,
  type EvolveMetrics,
} from '@/api/evolve'
import { ApiError } from '@/api/client'

const jobs = ref<EvolveJob[]>([])
const selected = ref('')
const metrics = ref<EvolveMetrics | null>(null)
const logText = ref('')
const error = ref<string | null>(null)
const stopping = ref(false)
const logEl = ref<HTMLPreElement | null>(null)

let poll: ReturnType<typeof setInterval> | null = null

const activeJob = computed(() => jobs.value.find((j) => j.run_id === selected.value) ?? null)

function downsample(points: EvolveMetricPoint[], max = 240): EvolveMetricPoint[] {
  if (points.length <= max) return points
  const step = Math.ceil(points.length / max)
  const out: EvolveMetricPoint[] = []
  for (let i = 0; i < points.length; i += step) out.push(points[i]!)
  const last = points[points.length - 1]!
  if (out[out.length - 1] !== last) out.push(last)
  return out
}

function seriesPath(
  points: EvolveMetricPoint[],
  xKey: 'step' | 'generation',
  yKey: 'loss' | 'primary',
  width: number,
  height: number,
  pad = 8,
): { d: string; min: number; max: number } {
  const vals = points
    .map((p) => p[yKey])
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (!vals.length || points.length < 2) return { d: '', min: 0, max: 1 }
  let min = Math.min(...vals)
  let max = Math.max(...vals)
  if (min === max) {
    min -= 1
    max += 1
  }
  const xs = points.map((p) => Number(p[xKey] ?? 0))
  const x0 = Math.min(...xs)
  const x1 = Math.max(...xs) || 1
  const innerW = width - pad * 2
  const innerH = height - pad * 2
  const coords = points
    .map((p) => {
      const yv = p[yKey]
      if (typeof yv !== 'number' || !Number.isFinite(yv)) return null
      const x = pad + ((Number(p[xKey] ?? 0) - x0) / (x1 - x0 || 1)) * innerW
      const y = pad + (1 - (yv - min) / (max - min)) * innerH
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .filter((c): c is string => c != null)
  return { d: coords.length ? `M ${coords.join(' L ')}` : '', min, max }
}

const trainPts = computed(() => downsample(metrics.value?.train ?? []))
const evalPts = computed(() => downsample(metrics.value?.eval ?? []))

const lossChart = computed(() => {
  const w = 640
  const h = 160
  const loss = seriesPath(trainPts.value, 'step', 'loss', w, h)
  return { w, h, d: loss.d, min: loss.min, max: loss.max }
})

const primaryChart = computed(() => {
  const w = 640
  const h = 160
  const prim = seriesPath(evalPts.value, 'generation', 'primary', w, h)
  return { w, h, d: prim.d, min: prim.min, max: prim.max }
})

async function refresh() {
  error.value = null
  try {
    const { jobs: next } = await listEvolveJobs()
    jobs.value = next
    if (!selected.value && next.length) {
      const running = next.find((j) => j.running)
      selected.value = running?.run_id ?? next[0]!.run_id
    } else if (selected.value && !next.some((j) => j.run_id === selected.value) && next.length) {
      selected.value = next[0]!.run_id
    }
    if (selected.value) {
      metrics.value = await getEvolveMetrics(selected.value)
      logText.value = await getEvolveLog(selected.value)
      await nextTick()
      if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load evolve jobs'
  }
}

async function onStop() {
  if (!selected.value) return
  stopping.value = true
  error.value = null
  try {
    await stopEvolveJob(selected.value)
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
  <AppShell title="Evolve">
    <section class="hero">
      <h1>SFT / evolve monitor</h1>
      <p class="lede">
        Live phase, eval primary, and train loss for <span class="mono">slm-rl evolve</span> /
        reject_sft warm-starts. Tails run logs and
        <span class="mono">train.metrics.jsonl</span> when present.
      </p>
    </section>

    <p v-if="error" class="err">{{ error }}</p>

    <UiCard v-if="!jobs.length" class="empty">
      <p>
        No evolve runs found under <span class="mono">runs/&lt;run_id&gt;/</span> or
        <span class="mono">logs/evolve-*.log</span>.
      </p>
    </UiCard>

    <template v-else>
      <div class="toolbar">
        <label class="pick">
          <span>Run</span>
          <select v-model="selected">
            <option v-for="j in jobs" :key="j.run_id" :value="j.run_id">
              {{ j.run_id }}{{ j.running ? ' · live' : '' }}
            </option>
          </select>
        </label>
        <span v-if="activeJob" class="pill" :data-live="activeJob.running">
          {{ activeJob.running ? `running · pid ${activeJob.pid}` : 'idle' }}
          · {{ activeJob.phase || '—' }}
          <template v-if="activeJob.phase_generation != null">
            gen {{ activeJob.phase_generation }}
          </template>
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
          <span class="k">Game / model</span>
          <span class="v small">
            {{ activeJob?.game || '—' }}
            <span class="meta">{{ activeJob?.backend || '' }}</span>
          </span>
        </div>
        <div class="stat">
          <span class="k">Phase</span>
          <span class="v">{{ activeJob?.phase || '—' }}</span>
        </div>
        <div class="stat">
          <span class="k">Primary</span>
          <span class="v mono">{{ fmt(activeJob?.last_primary) }}</span>
        </div>
        <div class="stat">
          <span class="k">Champion gen</span>
          <span class="v mono">{{ activeJob?.champion ?? '—' }}</span>
        </div>
      </div>

      <div class="charts">
        <UiCard class="chart-card">
          <header class="chart-head">
            <h2>Eval primary (per generation)</h2>
          </header>
          <svg
            class="chart"
            :viewBox="`0 0 ${primaryChart.w} ${primaryChart.h}`"
            preserveAspectRatio="none"
            role="img"
            aria-label="Eval primary by generation"
          >
            <path v-if="primaryChart.d" class="line primary" :d="primaryChart.d" />
            <text v-if="!primaryChart.d" x="16" y="32" class="empty-svg">
              waiting for eval results…
            </text>
          </svg>
          <div class="range mono">{{ fmt(primaryChart.min) }} → {{ fmt(primaryChart.max) }}</div>
        </UiCard>

        <UiCard class="chart-card">
          <header class="chart-head">
            <h2>SFT train loss</h2>
          </header>
          <svg
            class="chart"
            :viewBox="`0 0 ${lossChart.w} ${lossChart.h}`"
            preserveAspectRatio="none"
            role="img"
            aria-label="Training loss"
          >
            <path v-if="lossChart.d" class="line loss" :d="lossChart.d" />
            <text v-if="!lossChart.d" x="16" y="32" class="empty-svg">
              waiting for train steps…
            </text>
          </svg>
          <div class="range mono">{{ fmt(lossChart.min, 4) }} → {{ fmt(lossChart.max, 4) }}</div>
        </UiCard>
      </div>

      <UiCard v-if="activeJob?.generations?.length" class="gens-card">
        <header class="chart-head">
          <h2>Generations</h2>
        </header>
        <table class="gens">
          <thead>
            <tr>
              <th>Gen</th>
              <th>Primary</th>
              <th>SFT pairs</th>
              <th>Loss</th>
              <th>Gate</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="g in activeJob.generations" :key="g.generation">
              <td class="mono">{{ g.generation }}</td>
              <td class="mono">{{ fmt(g.primary) }}</td>
              <td class="mono">{{ g.num_pairs ?? '—' }}</td>
              <td class="mono">{{ fmt(g.loss, 4) }}</td>
              <td>
                <span v-if="g.promoted === true" class="ok">promoted</span>
                <span v-else-if="g.promoted === false" class="no">rejected</span>
                <span v-else class="meta">—</span>
              </td>
            </tr>
          </tbody>
        </table>
      </UiCard>

      <UiCard class="log-card">
        <header class="chart-head">
          <h2>Evolve log</h2>
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
  max-width: 56ch;
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
  max-width: min(420px, 70vw);
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

.stat .v.small {
  font-size: var(--text-sm);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.meta {
  color: var(--meta);
}

.charts {
  display: grid;
  gap: var(--space-4);
  margin-bottom: var(--space-5);
}

.chart-card,
.log-card,
.gens-card {
  padding: var(--space-4);
  margin-bottom: var(--space-4);
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

.chart {
  width: 100%;
  height: 160px;
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

.line.primary {
  stroke: var(--accent);
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

.gens {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--text-sm);
}

.gens th,
.gens td {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border-soft);
}

.gens th {
  color: var(--meta);
  font-weight: 500;
}

.ok {
  color: var(--success);
}
.no {
  color: var(--danger);
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
