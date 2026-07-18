<script setup lang="ts">
import { onMounted, onUnmounted, watch } from 'vue'

const props = withDefaults(
  defineProps<{
    open: boolean
    title?: string
    eyebrow?: string
    /** Default dialog; wide fits live evolve / theater-scale panels. */
    size?: 'default' | 'wide'
  }>(),
  { size: 'default' },
)

const emit = defineEmits<{
  close: []
}>()

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape' && props.open) emit('close')
}

watch(
  () => props.open,
  (open) => {
    document.body.style.overflow = open ? 'hidden' : ''
  },
)

onMounted(() => {
  window.addEventListener('keydown', onKey)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  document.body.style.overflow = ''
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="overlay"
      role="presentation"
      @click.self="emit('close')"
    >
      <div
        class="panel"
        :class="size"
        role="dialog"
        aria-modal="true"
        :aria-label="title || 'Dialog'"
      >
        <header v-if="eyebrow || title || $slots.header" class="head">
          <slot name="header">
            <p v-if="eyebrow" class="eyebrow">{{ eyebrow }}</p>
            <h2 v-if="title">{{ title }}</h2>
          </slot>
        </header>
        <div class="body">
          <slot />
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  z-index: 80;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: var(--space-8) var(--space-4);
  overflow: auto;
  background: rgba(0, 0, 0, 0.72);
  animation: fade-in var(--motion-fast) var(--ease-standard);
}

.panel {
  width: min(640px, 100%);
  max-height: calc(100vh - 2 * var(--space-8));
  overflow: auto;
  background: var(--surface);
  border-radius: var(--radius-md);
  box-shadow:
    0 0 0 1px rgba(0, 153, 255, 0.28),
    var(--elev-raised);
  padding: var(--space-6);
  animation: rise var(--motion-base) var(--ease-standard);
}

.panel.wide {
  width: min(1100px, 100%);
  max-height: calc(100vh - var(--space-6));
}

.head {
  margin-bottom: var(--space-5);
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

.body {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

@keyframes fade-in {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (prefers-reduced-motion: reduce) {
  .overlay,
  .panel {
    animation: none;
  }
}
</style>
