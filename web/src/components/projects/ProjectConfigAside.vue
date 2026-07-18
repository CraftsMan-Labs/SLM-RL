<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import type { KnobSchema } from '@/api/projects'

defineProps<{
  game?: string | null
  knobs: KnobSchema[]
  knobValues: Record<string, unknown>
  busy: boolean
  canStop: boolean
}>()

const datasetUrl = defineModel<string>('datasetUrl', { required: true })
const adapterUrl = defineModel<string>('adapterUrl', { required: true })
const dqnUrl = defineModel<string>('dqnUrl', { required: true })
const generations = defineModel<number>('generations', { required: true })
const episodesPerGen = defineModel<number>('episodesPerGen', { required: true })
const runEpisodes = defineModel<number>('runEpisodes', { required: true })

const emit = defineEmits<{
  'save-episodes-per-gen': []
}>()
</script>

<template>
  <div class="config">
    <details class="fold" open>
      <summary class="fold-sum">
        <span class="fold-copy">
          <span class="eyebrow">Workshop pack</span>
          <span class="fold-title" id="pack-heading">Paste URLs, then evolve</span>
        </span>
        <span class="chev" aria-hidden="true" />
      </summary>
      <div class="fold-body">
        <p class="hint">
          Workshop HF URLs pre-fill for all keeper games. Evolve imports the LoRA, runs RL, and
          streams episodes + the evolve log. Clear URLs for live-teacher warm-start.
        </p>
        <div class="fields">
          <UiField
            label="Dataset URL"
            for-id="dataset-url"
            hint="Pre-filled workshop dataset when available"
          >
            <input
              id="dataset-url"
              v-model="datasetUrl"
              type="text"
              placeholder="BLANK/slm-rl-space-invaders"
              autocomplete="off"
            />
          </UiField>
          <UiField
            label="SFT model URL"
            for-id="adapter-url"
            hint="Pre-filled workshop LoRA (adapter/) when available"
          >
            <input
              id="adapter-url"
              v-model="adapterUrl"
              type="text"
              placeholder="BLANK/slm-rl-boxing"
              autocomplete="off"
            />
          </UiField>
          <UiField label="DQN URL (optional)" for-id="dqn-url">
            <input
              id="dqn-url"
              v-model="dqnUrl"
              type="text"
              placeholder="BLANK/slm-rl-boxing-dqn"
              autocomplete="off"
            />
          </UiField>
          <UiField
            label="Generations"
            for-id="generations"
            hint="RL generations this Evolve. Default 2 — real GRPO + strict EvalGate; early-stops after 2 rejects."
          >
            <input
              id="generations"
              v-model.number="generations"
              type="number"
              min="1"
              max="50"
            />
          </UiField>
          <UiField
            label="Episodes per generation"
            for-id="ep-per-gen"
            hint="LLM games per Evolve round. Default 2 (workshop speed). Save, then restart Evolve."
          >
            <div class="ep-row">
              <input
                id="ep-per-gen"
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
          <UiField
            label="Quick-run episodes"
            for-id="run-ep"
            hint="Run game = teacher/solver screen, no train"
          >
            <input
              id="run-ep"
              v-model.number="runEpisodes"
              type="number"
              min="1"
              max="200"
            />
          </UiField>
        </div>
      </div>
    </details>

    <details class="fold">
      <summary class="fold-sum">
        <span class="fold-copy">
          <span class="eyebrow">Parameters</span>
          <span class="fold-title" id="params-heading">
            Frozen at create
            <span class="fold-meta">{{ knobs.length }} knobs · {{ game || 'game' }}</span>
          </span>
        </span>
        <span class="chev" aria-hidden="true" />
      </summary>
      <div class="fold-body">
        <p class="hint">
          Locked into this project at create. Different knobs need a new project.
        </p>
        <dl v-if="knobs.length" class="frozen">
          <div v-for="k in knobs" :key="k.key" class="frozen-row">
            <dt>
              <span>{{ k.label }}</span>
              <span
                v-if="k.help"
                class="info"
                tabindex="0"
                role="button"
                :aria-label="`${k.help.title}: more info`"
              >
                <span class="info-mark" aria-hidden="true">i</span>
                <span class="info-card" role="tooltip">
                  <strong>{{ k.help.title }}</strong>
                  <p>{{ k.help.body }}</p>
                </span>
              </span>
            </dt>
            <dd class="mono">{{ knobValues[k.key] ?? '—' }}</dd>
          </div>
        </dl>
        <p v-else class="hint">No knobs for this game.</p>
      </div>
    </details>
  </div>
</template>

<style scoped>
.config {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: var(--space-4) var(--space-3) var(--space-8);
}

.fold {
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.03);
  box-shadow: inset 0 0 0 1px var(--border-soft);
  overflow: clip;
}

.fold-sum {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 12px 12px 14px;
  cursor: pointer;
  list-style: none;
  user-select: none;
  transition: background var(--motion-fast) var(--ease-standard);
}

.fold-sum::-webkit-details-marker {
  display: none;
}

.fold-sum:hover {
  background: rgba(255, 255, 255, 0.04);
}

.fold-sum:focus-visible {
  outline: none;
  box-shadow: inset 0 0 0 2px color-mix(in oklab, var(--accent), transparent 40%);
}

.fold-copy {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  flex: 1;
}

.eyebrow {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
}

.fold-title {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-family: var(--font-display);
  font-size: 15px;
  font-weight: 500;
  letter-spacing: -0.03em;
  line-height: var(--leading-tight);
  color: var(--fg);
}

.fold-meta {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 400;
  letter-spacing: 0;
  color: var(--meta);
}

.chev {
  flex: 0 0 auto;
  width: 28px;
  height: 28px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
  position: relative;
  transition:
    transform var(--motion-fast) var(--ease-standard),
    background var(--motion-fast) var(--ease-standard);
}

.chev::before {
  content: '';
  position: absolute;
  inset: 0;
  margin: auto;
  width: 7px;
  height: 7px;
  border-right: 1.5px solid var(--muted);
  border-bottom: 1.5px solid var(--muted);
  transform: translateY(-2px) rotate(45deg);
}

.fold[open] > .fold-sum .chev {
  transform: rotate(180deg);
  background: rgba(255, 255, 255, 0.1);
}

.fold-body {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  padding: 0 14px 14px;
}

.hint {
  margin: 0;
  color: var(--meta);
  font-size: var(--text-sm);
  line-height: 1.45;
}

.fields {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.ep-row {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.ep-row input {
  flex: 1;
  min-width: 0;
}

@media (prefers-reduced-motion: reduce) {
  .chev {
    transition: none;
  }
}

.frozen {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.frozen-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-bottom: var(--space-3);
  box-shadow: 0 1px 0 var(--border-soft);
}

.frozen-row:last-child {
  padding-bottom: 0;
  box-shadow: none;
}

.frozen dt {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--text-xs);
  color: var(--meta);
}

.frozen dd {
  margin: 0;
  font-size: var(--text-sm);
  color: var(--fg);
}

.mono {
  font-family: var(--font-mono);
  font-size: 0.92em;
}

.frozen .info {
  position: relative;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border-radius: 999px;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.28);
  color: var(--fg-2);
  cursor: help;
  outline: none;
}

.frozen .info-mark {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 600;
  font-style: italic;
  line-height: 1;
}

.frozen .info:hover,
.frozen .info:focus-visible {
  color: var(--fg);
  box-shadow:
    inset 0 0 0 1px var(--accent),
    0 0 0 2px rgba(0, 153, 255, 0.2);
}

.frozen .info-card {
  display: none;
  position: absolute;
  left: 0;
  top: calc(100% + 8px);
  z-index: 40;
  width: min(240px, 70vw);
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  background: #14161a;
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.12),
    0 12px 32px rgba(0, 0, 0, 0.45);
  color: var(--fg);
  text-align: left;
  white-space: normal;
  pointer-events: none;
}

.frozen .info-card strong {
  display: block;
  margin-bottom: 6px;
  font-size: 12px;
  font-weight: 600;
}

.frozen .info-card p {
  margin: 0;
  font-size: 12px;
  line-height: 1.45;
  color: var(--fg-2);
  font-weight: 400;
}

.frozen .info:hover .info-card,
.frozen .info:focus-visible .info-card {
  display: block;
}
</style>
