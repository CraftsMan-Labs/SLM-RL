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
  canStop: boolean
  evolveRunning: boolean
}>()

const datasetUrl = defineModel<string>('datasetUrl', { required: true })
const adapterUrl = defineModel<string>('adapterUrl', { required: true })
const dqnUrl = defineModel<string>('dqnUrl', { required: true })
const generations = defineModel<number>('generations', { required: true })
const episodesPerGen = defineModel<number>('episodesPerGen', { required: true })

const emit = defineEmits<{
  close: []
  start: []
  stop: []
  'save-episodes-per-gen': []
  'evolve-dead': [detail: { phase: string; phaseGen: number | null }]
}>()
</script>

<template>
  <UiModal :open="open" size="wide" title="Self-improve loop" @close="emit('close')">
    <template #header>
      <div class="head-row">
        <div>
          <p class="eyebrow">Evolve</p>
          <h2>Self-improve loop</h2>
        </div>
        <UiButton variant="ghost" :disabled="busy" @click="emit('close')">Close</UiButton>
      </div>
    </template>

    <p class="lede">
      Configure the run, then start. Opening this window does not launch Evolve — use
      <strong>Start Evolve</strong> when you are ready.
    </p>

    <div class="controls">
      <div class="fields">
        <UiField label="Generations" for-id="evolve-gens" hint="RL generations this run">
          <input
            id="evolve-gens"
            v-model.number="generations"
            type="number"
            min="1"
            max="50"
            :disabled="busy || evolveRunning"
          />
        </UiField>
        <UiField
          label="Episodes per generation"
          for-id="evolve-ep"
          hint="LLM games per collect step. Save before starting."
        >
          <div class="ep-row">
            <input
              id="evolve-ep"
              v-model.number="episodesPerGen"
              type="number"
              min="1"
              max="200"
              :disabled="busy || canStop"
            />
            <UiButton
              variant="secondary"
              :disabled="busy || canStop"
              @click="emit('save-episodes-per-gen')"
            >
              Save
            </UiButton>
          </div>
        </UiField>
        <UiField label="Dataset URL" for-id="evolve-dataset">
          <input id="evolve-dataset" v-model="datasetUrl" type="text" autocomplete="off" />
        </UiField>
        <UiField label="SFT model URL" for-id="evolve-adapter">
          <input id="evolve-adapter" v-model="adapterUrl" type="text" autocomplete="off" />
        </UiField>
        <UiField label="DQN URL" for-id="evolve-dqn">
          <input id="evolve-dqn" v-model="dqnUrl" type="text" autocomplete="off" />
        </UiField>
      </div>

      <div class="actions">
        <UiButton
          class="start"
          :disabled="busy || evolveRunning"
          @click="emit('start')"
        >
          {{ evolveRunning ? 'Evolve running…' : 'Start Evolve' }}
        </UiButton>
        <UiButton
          v-if="evolveRunning || canStop"
          variant="danger"
          :disabled="busy"
          @click="emit('stop')"
        >
          Stop Evolve
        </UiButton>
      </div>
    </div>

    <LiveWatchPanel
      v-if="open"
      :project-name="projectName"
      :project-status="projectStatus"
      :active-jobs="activeJobs"
      @evolve-dead="emit('evolve-dead', $event)"
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

.controls {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  padding-bottom: var(--space-4);
  border-bottom: 1px solid color-mix(in srgb, var(--border) 80%, transparent);
}

.fields {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3) var(--space-4);
}

.ep-row {
  display: flex;
  gap: var(--space-2);
  align-items: center;
}

.ep-row input {
  flex: 1;
  min-width: 0;
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

input[type='text'],
input[type='number'] {
  width: 100%;
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

@media (max-width: 720px) {
  .fields {
    grid-template-columns: 1fr;
  }
}
</style>
