<script setup lang="ts">
defineProps<{
  eyebrow?: string
  title?: string
}>()
</script>

<template>
  <section class="card">
    <header v-if="eyebrow || title || $slots.header" class="head">
      <slot name="header">
        <p v-if="eyebrow" class="eyebrow">{{ eyebrow }}</p>
        <h2 v-if="title">{{ title }}</h2>
      </slot>
    </header>
    <div class="body">
      <slot />
    </div>
  </section>
</template>

<style scoped>
.card {
  background: var(--surface);
  border-radius: var(--radius-md);
  box-shadow: var(--elev-ring);
  padding: var(--space-6);
  transition: box-shadow var(--motion-base) var(--ease-standard);
}

.card:hover {
  box-shadow:
    0 0 0 1px rgba(0, 153, 255, 0.28),
    var(--elev-raised);
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
  font-size: var(--text-xl);
  letter-spacing: -0.03em;
}

.body {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}
</style>
