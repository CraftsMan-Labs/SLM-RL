<script setup lang="ts">
withDefaults(
  defineProps<{
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
    type?: 'button' | 'submit'
    disabled?: boolean
    block?: boolean
  }>(),
  {
    variant: 'primary',
    type: 'button',
    disabled: false,
    block: false,
  },
)
</script>

<template>
  <button
    class="btn"
    :class="[variant, { block }]"
    :type="type"
    :disabled="disabled"
  >
    <slot />
  </button>
</template>

<style scoped>
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  min-height: 44px;
  padding: 10px 18px;
  border: 0;
  border-radius: var(--radius-pill);
  cursor: pointer;
  transition:
    transform var(--motion-fast) var(--ease-standard),
    background var(--motion-fast) var(--ease-standard),
    opacity var(--motion-fast) var(--ease-standard);
}

.btn:active:not(:disabled) {
  transform: scale(0.97);
}

.btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.btn.block {
  width: 100%;
}

.primary {
  background: #fff;
  color: #000;
  font-weight: 600;
}

.primary:hover:not(:disabled) {
  background: #f2f2f2;
}

.secondary {
  background: var(--frosted);
  color: var(--fg);
}

.secondary:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.16);
}

.ghost {
  background: transparent;
  color: var(--fg);
}

.ghost:hover:not(:disabled) {
  background: var(--frosted);
}

.danger {
  background: rgba(220, 38, 38, 0.16);
  color: #fecaca;
  font-weight: 600;
}

.danger:hover:not(:disabled) {
  background: rgba(220, 38, 38, 0.28);
}
</style>
