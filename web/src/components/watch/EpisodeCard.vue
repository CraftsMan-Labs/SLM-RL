<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { EpisodeState } from '@/composables/useWatchStream'
import type { WatchEvent } from '@/api/watch'
import UiButton from '@/components/ui/UiButton.vue'

const props = defineProps<{
  episode: EpisodeState
}>()

const emit = defineEmits<{
  watch: [episodeId: string]
}>()

const logEl = ref<HTMLOListElement | null>(null)

const inPlay = computed(() => !props.episode.terminated && !props.episode.truncated)

watch(
  () => props.episode.steps.length,
  async () => {
    if (!inPlay.value) return
    await nextTick()
    const el = logEl.value
    if (el) el.scrollTop = el.scrollHeight
  },
)

const outcomeClass = computed(() => {
  if (inPlay.value) return 'playing'
  if (props.episode.outcome === 'win') return 'win'
  if (props.episode.outcome === 'loss' || props.episode.truncated) return 'loss'
  return ''
})

const latest = computed(() => props.episode.steps[0] ?? null)

/** Stream order: oldest → newest (steps[] is newest-first). */
const chronSteps = computed(() => [...props.episode.steps].reverse())

/** DQN / heuristic teachers — real input is vector_obs, not the text prompt. */
const isTeacher = computed(() => {
  const id = props.episode.modelId ?? ''
  return id.startsWith('teacher:') || id.startsWith('solver')
})

function flagKeys(flags: Record<string, unknown> | null | undefined): string[] {
  if (!flags || typeof flags !== 'object') return []
  return Object.keys(flags)
}

function stepKey(s: WatchEvent, i: number): string {
  return `${s.step_idx ?? 'x'}-${s.parsed_action ?? ''}-${i}`
}
</script>

<template>
  <article class="ep" :class="outcomeClass">
    <header class="head">
      <div class="meta">
        <span class="id">episode {{ episode.id }}</span>
        <span>gen {{ episode.generation ?? '—' }}</span>
        <span>seed {{ episode.seed ?? '—' }}</span>
        <span class="model" :title="episode.modelId ?? undefined">
          {{ episode.modelId ?? '—' }}
        </span>
        <span v-if="isTeacher" class="chip agent">teacher / DQN</span>
        <span v-else class="chip agent slm">SLM</span>
      </div>
      <div class="actions">
        <UiButton variant="secondary" class="watch-btn" @click="emit('watch', episode.id)">
          Watch screen
        </UiButton>
        <span v-if="inPlay" class="outcome playing">In play</span>
        <span v-else-if="episode.outcome" class="outcome" :class="outcomeClass">
          {{ episode.outcome }}
          <template v-if="episode.cumReward != null">
            · cum {{ episode.cumReward }}
          </template>
        </span>
        <span v-else class="outcome">Done</span>
      </div>
    </header>

    <div v-if="latest" class="latest">
      <div class="row">
        <span class="step-n">step {{ latest.step_idx ?? '—' }}</span>
        <code class="action">{{ latest.parsed_action ?? '—' }}</code>
        <span class="reward">
          reward {{ latest.reward ?? '—' }} · cum {{ latest.cum_reward ?? '—' }}
        </span>
        <span
          class="badge"
          :class="{ bad: latest.parse_status && latest.parse_status !== 'ok' }"
        >
          {{ latest.parse_status ?? '' }}
        </span>
        <span
          v-for="flag in flagKeys(latest.monitor_flags)"
          :key="flag"
          class="chip"
        >
          {{ flag }}
        </span>
      </div>
    </div>

    <details v-if="episode.steps.length" class="log-panel" :open="inPlay">
      <summary>
        {{ episode.steps.length }} recent step{{ episode.steps.length === 1 ? '' : 's' }}
        <template v-if="inPlay"> · live</template>
      </summary>
      <p v-if="isTeacher" class="note">
        Teacher chose from numeric <code>vector_obs</code>. Text below is for SLM export.
      </p>
      <ol ref="logEl" class="log" aria-label="Episode step log">
        <li v-for="(s, i) in chronSteps" :key="stepKey(s, i)" class="log-line">
          <div class="log-head">
            <span class="step-n">#{{ s.step_idx ?? i + 1 }}</span>
            <code class="action">{{ s.parsed_action ?? '—' }}</code>
            <span class="reward">r {{ s.reward ?? '—' }} · Σ {{ s.cum_reward ?? '—' }}</span>
            <span
              v-if="s.parse_status && s.parse_status !== 'ok'"
              class="badge bad"
            >
              {{ s.parse_status }}
            </span>
          </div>
          <pre
            v-if="s.completion && !isTeacher"
            class="completion"
          >{{ s.completion }}</pre>
          <details v-if="s.observed && !isTeacher" class="io">
            <summary>board / observation</summary>
            <pre>{{ s.observed }}</pre>
          </details>
          <details v-if="s.completion && isTeacher" class="io">
            <summary>logged action text</summary>
            <pre>{{ s.completion }}</pre>
          </details>
        </li>
      </ol>
    </details>
  </article>
</template>

<style scoped>
.ep {
  padding: var(--space-4);
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.02);
  box-shadow: 0 0 0 1px var(--border-soft);
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.ep.playing {
  box-shadow: 0 0 0 1px rgba(0, 153, 255, 0.35);
}

.ep.win {
  box-shadow: 0 0 0 1px rgba(22, 163, 74, 0.35);
}

.ep.loss {
  box-shadow: 0 0 0 1px rgba(220, 38, 38, 0.35);
}

.head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-4);
  align-items: flex-start;
  flex-wrap: wrap;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2) var(--space-3);
  font-size: var(--text-xs);
  color: var(--meta);
  font-family: var(--font-mono);
}

.id {
  color: var(--fg-2);
}

.model {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.actions {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.watch-btn {
  min-height: 32px;
  padding: 6px 12px;
  font-size: var(--text-xs);
}

.outcome {
  font-size: var(--text-sm);
  font-weight: 600;
  color: var(--fg-2);
  text-transform: lowercase;
}

.outcome.playing {
  color: var(--accent);
  text-transform: uppercase;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.06em;
}

.outcome.win {
  color: var(--success);
}

.outcome.loss {
  color: var(--danger);
}

.latest .row,
.log-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2);
}

.step-n {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--meta);
  min-width: 3.5rem;
}

.action {
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  color: var(--fg);
  background: var(--frosted);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
}

.reward {
  font-size: var(--text-xs);
  color: var(--muted);
  font-family: var(--font-mono);
}

.badge {
  font-size: 11px;
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  background: var(--frosted);
  color: var(--meta);
}

.badge.bad {
  color: var(--danger);
  background: rgba(220, 38, 38, 0.15);
}

.chip {
  font-size: 11px;
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  background: rgba(234, 179, 8, 0.12);
  color: var(--warn);
}

.chip.agent {
  background: rgba(255, 255, 255, 0.06);
  color: var(--meta);
}

.chip.agent.slm {
  background: rgba(0, 153, 255, 0.12);
  color: var(--accent);
}

.note {
  margin: 0 0 var(--space-2);
  font-size: var(--text-xs);
  color: var(--muted);
  line-height: 1.45;
}

.note code {
  font-family: var(--font-mono);
  font-size: 11px;
}

.log-panel {
  border-radius: var(--radius-sm);
  background: rgba(0, 0, 0, 0.22);
  box-shadow: inset 0 0 0 1px var(--border-soft);
  padding: var(--space-2) var(--space-3);
}

.log-panel > summary {
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  letter-spacing: 0.04em;
  color: var(--meta);
  user-select: none;
  padding: var(--space-1) 0;
}

.log-panel[open] > summary {
  color: var(--accent);
  margin-bottom: var(--space-2);
}

.log {
  list-style: none;
  margin: 0;
  padding: 0;
  max-height: 220px;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  scroll-behavior: smooth;
}

.log-line {
  padding: var(--space-2);
  border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.03);
}

.log-line:last-child {
  box-shadow: inset 0 0 0 1px rgba(0, 153, 255, 0.25);
}

.completion,
.io pre {
  margin: var(--space-2) 0 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.45;
  color: var(--muted);
  background: rgba(255, 255, 255, 0.03);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  max-height: 120px;
  overflow: auto;
}

.completion {
  color: var(--fg-2, var(--text));
  border-left: 2px solid rgba(0, 153, 255, 0.45);
}

.io {
  margin-top: var(--space-1);
}

.io summary {
  cursor: pointer;
  font-size: 11px;
  color: var(--meta);
}
</style>
