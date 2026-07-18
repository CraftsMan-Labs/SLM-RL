<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  status: 'connecting' | 'open' | 'closed'
}>()

const label = computed(() => {
  if (props.status === 'open') return 'live'
  if (props.status === 'connecting') return 'connecting'
  // ponytail: "idle" not "offline" — closed before Start Evolve is normal.
  return 'idle'
})
</script>

<template>
  <span class="pill" :class="status" role="status">
    <span class="dot" aria-hidden="true" />
    {{ label }}
  </span>
</template>

<style scoped>
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: var(--radius-pill);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--meta);
  background: var(--frosted);
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}

.open {
  color: var(--accent);
  box-shadow: 0 0 0 1px rgba(0, 153, 255, 0.35);
}

.open .dot {
  animation: pulse 1.4s var(--ease-standard) infinite;
}

.connecting {
  color: var(--muted);
}

.closed {
  color: var(--muted);
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.35;
  }
}

@media (prefers-reduced-motion: reduce) {
  .open .dot {
    animation: none;
  }
}
</style>
