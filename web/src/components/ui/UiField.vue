<script setup lang="ts">
defineProps<{
  label: string
  hint?: string
  forId?: string
  /** Hover/focus ⓘ card (from /api/knobs help). */
  infoTitle?: string
  info?: string
}>()
</script>

<template>
  <label class="field" :for="forId">
    <span class="label-row">
      <span class="label">{{ label }}</span>
      <span
        v-if="info"
        class="info"
        tabindex="0"
        role="button"
        :aria-label="infoTitle ? `${infoTitle}: more info` : `${label}: more info`"
        @click.prevent
      >
        <span class="info-mark" aria-hidden="true">i</span>
        <span class="info-card" role="tooltip">
          <strong v-if="infoTitle">{{ infoTitle }}</strong>
          <p>{{ info }}</p>
        </span>
      </span>
    </span>
    <slot />
    <span v-if="hint" class="hint">{{ hint }}</span>
  </label>
</template>

<style scoped>
.field {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.label-row {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.label {
  font-size: var(--text-sm);
  font-weight: 500;
  color: var(--muted);
}

.info {
  position: relative;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 999px;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.28);
  color: var(--fg-2);
  cursor: help;
  outline: none;
}

.info-mark {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  font-style: italic;
  line-height: 1;
  transform: translateY(-0.5px);
}

.info:hover,
.info:focus-visible {
  color: var(--fg);
  box-shadow:
    inset 0 0 0 1px var(--accent),
    0 0 0 2px rgba(0, 153, 255, 0.2);
}

.info-card {
  display: none;
  position: absolute;
  left: 0;
  top: calc(100% + 8px);
  z-index: 40;
  width: min(280px, 70vw);
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

.info-card strong {
  display: block;
  margin-bottom: 6px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: -0.01em;
}

.info-card p {
  margin: 0;
  font-size: 12px;
  line-height: 1.45;
  color: var(--fg-2);
  font-weight: 400;
}

.info:hover .info-card,
.info:focus-visible .info-card {
  display: block;
}

.hint {
  font-size: var(--text-xs);
  color: var(--meta);
}

:deep(input),
:deep(select),
:deep(textarea) {
  width: 100%;
  min-height: 44px;
  padding: 10px 14px;
  border: 0;
  border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.04);
  box-shadow: 0 0 0 1px var(--border-soft);
  color: var(--fg);
  transition: box-shadow var(--motion-fast) var(--ease-standard);
}

:deep(input::placeholder),
:deep(textarea::placeholder) {
  color: var(--meta);
}

:deep(input:hover),
:deep(select:hover),
:deep(textarea:hover) {
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.12);
}

:deep(input:focus-visible),
:deep(select:focus-visible),
:deep(textarea:focus-visible) {
  box-shadow: 0 0 0 1px var(--accent), var(--focus-ring);
}

:deep(select) {
  appearance: none;
  background-image: linear-gradient(45deg, transparent 50%, var(--fg-2) 50%),
    linear-gradient(135deg, var(--fg-2) 50%, transparent 50%);
  background-position:
    calc(100% - 18px) 50%,
    calc(100% - 12px) 50%;
  background-size:
    6px 6px,
    6px 6px;
  background-repeat: no-repeat;
  padding-right: 36px;
}
</style>
