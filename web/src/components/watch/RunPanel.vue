<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiModal from '@/components/ui/UiModal.vue'
import LiveWatchPanel from '@/components/watch/LiveWatchPanel.vue'

defineProps<{
  open: boolean
  projectName: string
  projectStatus?: string | null
  activeJobs?: string[] | null
  busy: boolean
  runRunning: boolean
}>()

const runEpisodes = defineModel<number>('runEpisodes', { required: true })
const maxTurns = defineModel<number>('maxTurns', { required: true })

const emit = defineEmits<{
  close: []
  start: []
  stop: []
}>()
</script>

<template>
  <UiModal :open="open" size="wide" title="Quick run" @close="emit('close')">
    <template #header>
      <div class="head-row">
        <div>
          <p class="eyebrow">Run</p>
          <h2>Quick teacher play</h2>
        </div>
        <UiButton variant="ghost" :disabled="busy" @click="emit('close')">Close</UiButton>
      </div>
    </template>

    <p class="lede">
      Teacher plays on screen — <strong>no training</strong>, and it never trains a fresh DQN.
      Uses the experiment’s Hugging Face DQN (<code>dqn_url</code> / baked pack) when
      configured; otherwise the game heuristic. Opening this window does not start a run —
      use <strong>Start Run</strong> when you are ready.
    </p>

    <div class="controls">
      <div class="fields">
        <UiField
          label="Episodes"
          for-id="run-panel-ep"
          hint="How many teacher games to play in this quick run"
        >
          <input
            id="run-panel-ep"
            v-model.number="runEpisodes"
            type="number"
            min="1"
            max="200"
            :disabled="busy || runRunning"
          />
        </UiField>
        <UiField
          label="Max turns"
          for-id="run-panel-turns"
          hint="Decisions per episode (playground default 30 truncates early; Boxing full bout ≈ 2500)"
        >
          <input
            id="run-panel-turns"
            v-model.number="maxTurns"
            type="number"
            min="1"
            max="20000"
            :disabled="busy || runRunning"
          />
        </UiField>
      </div>

      <div class="actions">
        <UiButton class="start" :disabled="busy || runRunning" @click="emit('start')">
          {{ runRunning ? 'Run in progress…' : 'Start Run' }}
        </UiButton>
        <UiButton v-if="runRunning" variant="danger" :disabled="busy" @click="emit('stop')">
          Stop Run
        </UiButton>
      </div>
    </div>

    <LiveWatchPanel
      v-if="open"
      :project-name="projectName"
      :project-status="projectStatus"
      :active-jobs="activeJobs"
    />
  </UiModal>
</template>

<style scoped>
.head-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
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

.controls {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  padding-bottom: var(--space-4);
  border-bottom: 1px solid color-mix(in srgb, var(--border) 80%, transparent);
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

.actions :deep(.start.btn) {
  min-width: 10rem;
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
</style>
