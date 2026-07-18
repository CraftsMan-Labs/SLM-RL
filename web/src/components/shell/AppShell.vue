<script setup lang="ts">
import { computed, onUnmounted, ref, useSlots, watch } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { useProfile } from '@/composables/useProfile'

defineProps<{
  title?: string
}>()

const ASIDE_KEY = 'slm-rl.aside-collapsed'

const { profile } = useProfile()
const route = useRoute()
const slots = useSlots()
const hasAside = computed(() => Boolean(slots.aside))
const navOpen = ref(false)
const asideCollapsed = ref(
  typeof localStorage !== 'undefined' && localStorage.getItem(ASIDE_KEY) === '1',
)

watch(
  () => route.fullPath,
  () => {
    navOpen.value = false
  },
)

function toggleNav() {
  navOpen.value = !navOpen.value
}

function closeNav() {
  navOpen.value = false
}

function toggleAside() {
  asideCollapsed.value = !asideCollapsed.value
  try {
    localStorage.setItem(ASIDE_KEY, asideCollapsed.value ? '1' : '0')
  } catch {
    /* ignore quota / private mode */
  }
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && navOpen.value) {
    navOpen.value = false
  }
}

watch(navOpen, (open) => {
  if (typeof document === 'undefined') return
  document.body.style.overflow = open ? 'hidden' : ''
  if (open) {
    window.addEventListener('keydown', onKeydown)
  } else {
    window.removeEventListener('keydown', onKeydown)
  }
})

onUnmounted(() => {
  if (typeof document === 'undefined') return
  document.body.style.overflow = ''
  window.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <div
    class="shell"
    :class="{
      'has-aside': hasAside,
      'nav-open': navOpen,
      'aside-collapsed': hasAside && asideCollapsed,
    }"
  >
    <header class="top">
      <div class="brand-row">
        <button
          v-if="hasAside"
          type="button"
          class="menu-btn"
          :aria-expanded="navOpen"
          aria-controls="app-sidebar"
          @click="toggleNav"
        >
          <span class="sr-only">{{ navOpen ? 'Close sidebar' : 'Open sidebar' }}</span>
          <span class="menu-icon" aria-hidden="true" />
        </button>
        <button
          v-if="hasAside"
          type="button"
          class="rail-btn"
          :aria-expanded="!asideCollapsed"
          aria-controls="app-sidebar"
          :title="asideCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
          @click="toggleAside"
        >
          <span class="sr-only">{{ asideCollapsed ? 'Expand sidebar' : 'Collapse sidebar' }}</span>
          <span class="rail-chev" aria-hidden="true" />
        </button>
        <RouterLink class="brand" to="/projects">SLM-RL</RouterLink>
        <span v-if="title" class="sep">/</span>
        <span v-if="title" class="page-title">{{ title }}</span>
      </div>
      <div class="meta">
        <nav class="top-nav" aria-label="Primary">
          <RouterLink class="nav-link" to="/projects">Projects</RouterLink>
          <RouterLink class="nav-link" to="/teachers">Teachers</RouterLink>
          <RouterLink class="nav-link" to="/evolve">Evolve</RouterLink>
        </nav>
        <span v-if="profile" class="who">{{ profile.name }}</span>
      </div>
    </header>

    <div class="body">
      <div
        v-if="hasAside"
        class="scrim"
        :class="{ show: navOpen }"
        aria-hidden="true"
        @click="closeNav"
      />
      <aside
        v-if="hasAside"
        id="app-sidebar"
        class="aside"
        :class="{ open: navOpen, collapsed: asideCollapsed }"
        aria-label="Workspace controls"
      >
        <button
          type="button"
          class="aside-edge"
          :aria-expanded="!asideCollapsed"
          :title="asideCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
          @click="toggleAside"
        >
          <span class="sr-only">{{ asideCollapsed ? 'Expand sidebar' : 'Collapse sidebar' }}</span>
          <span class="edge-chev" aria-hidden="true" />
        </button>
        <div class="aside-inner">
          <div class="aside-nav">
            <p class="aside-label">Navigate</p>
            <RouterLink class="aside-link" to="/projects" @click="closeNav">Projects</RouterLink>
            <RouterLink class="aside-link" to="/teachers" @click="closeNav">Teachers</RouterLink>
            <RouterLink class="aside-link" to="/evolve" @click="closeNav">Evolve</RouterLink>
          </div>
          <div class="aside-slot">
            <slot name="aside" />
          </div>
        </div>
      </aside>
      <main class="main">
        <slot />
      </main>
    </div>
  </div>
</template>

<style scoped>
.shell {
  min-height: 100dvh;
  background:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 153, 255, 0.12), transparent 55%),
    var(--bg);
}

.top {
  position: sticky;
  top: 0;
  z-index: 30;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-4) var(--space-6);
  background: rgba(0, 0, 0, 0.85);
  box-shadow: 0 1px 0 var(--border-soft);
  backdrop-filter: blur(8px);
}

.brand-row {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  min-width: 0;
}

.brand {
  font-family: var(--font-display);
  font-size: var(--text-lg);
  letter-spacing: -0.04em;
  color: var(--fg);
  text-decoration: none;
}

.sep {
  color: var(--meta);
}

.page-title {
  color: var(--fg-2);
  font-size: var(--text-sm);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.meta {
  display: flex;
  align-items: center;
  gap: var(--space-4);
}

.top-nav {
  display: flex;
  align-items: center;
  gap: var(--space-4);
}

.nav-link {
  color: var(--muted);
  text-decoration: none;
  font-size: var(--text-sm);
  letter-spacing: -0.15px;
  transition: color var(--motion-fast) var(--ease-standard);
}

.nav-link:hover,
.nav-link.router-link-active {
  color: var(--accent);
  text-decoration: none;
}

.who {
  font-size: var(--text-sm);
  color: var(--meta);
}

.menu-btn,
.rail-btn {
  display: none;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  margin-right: var(--space-1);
  padding: 0;
  border: 0;
  border-radius: var(--radius-sm);
  background: var(--frosted);
  color: var(--fg);
  cursor: pointer;
  transition:
    background var(--motion-fast) var(--ease-standard),
    box-shadow var(--motion-fast) var(--ease-standard);
}

.menu-btn:hover,
.rail-btn:hover {
  background: rgba(255, 255, 255, 0.14);
}

.menu-btn:focus-visible,
.rail-btn:focus-visible {
  outline: none;
  box-shadow: var(--focus-ring);
}

.rail-btn {
  display: inline-flex;
}

.rail-chev,
.edge-chev {
  display: block;
  width: 7px;
  height: 7px;
  border-right: 1.5px solid currentColor;
  border-bottom: 1.5px solid currentColor;
  transform: rotate(135deg);
  transition: transform var(--motion-fast) var(--ease-standard);
}

.aside-collapsed .rail-chev,
.aside-collapsed .edge-chev {
  transform: rotate(-45deg);
}

.menu-icon,
.menu-icon::before,
.menu-icon::after {
  display: block;
  width: 16px;
  height: 1.5px;
  background: currentColor;
  border-radius: 1px;
  transition:
    transform var(--motion-base) var(--ease-standard),
    opacity var(--motion-fast) var(--ease-standard);
}

.menu-icon {
  position: relative;
}

.menu-icon::before,
.menu-icon::after {
  content: '';
  position: absolute;
  left: 0;
}

.menu-icon::before {
  top: -5px;
}

.menu-icon::after {
  top: 5px;
}

.nav-open .menu-icon {
  background: transparent;
}

.nav-open .menu-icon::before {
  top: 0;
  transform: rotate(45deg);
}

.nav-open .menu-icon::after {
  top: 0;
  transform: rotate(-45deg);
}

.body {
  display: block;
}

.has-aside .body {
  display: grid;
  grid-template-columns: var(--sidebar-width, 280px) minmax(0, 1fr);
  align-items: start;
  min-height: calc(100dvh - 65px);
  transition: grid-template-columns var(--motion-base) var(--ease-standard);
}

.aside-collapsed.has-aside .body {
  grid-template-columns: 44px minmax(0, 1fr);
}

.aside {
  position: sticky;
  top: 65px;
  z-index: 20;
  height: calc(100dvh - 65px);
  overflow: visible;
  background: var(--surface);
  box-shadow: 1px 0 0 var(--border-soft);
  animation: slide-in var(--motion-base) var(--ease-standard) both;
}

.aside-inner {
  height: 100%;
  overflow: auto;
  overscroll-behavior: contain;
  opacity: 1;
  transition: opacity var(--motion-fast) var(--ease-standard);
}

.aside.collapsed {
  overflow: hidden;
}

.aside.collapsed .aside-inner {
  opacity: 0;
  pointer-events: none;
  visibility: hidden;
}

.aside-edge {
  display: none;
  position: absolute;
  top: 50%;
  right: 0;
  z-index: 2;
  width: 28px;
  height: 44px;
  transform: translate(50%, -50%);
  align-items: center;
  justify-content: center;
  padding: 0;
  border: 0;
  border-radius: var(--radius-pill);
  background: var(--surface);
  color: var(--muted);
  cursor: pointer;
  box-shadow:
    0 0 0 1px var(--border-soft),
    0 8px 20px rgba(0, 0, 0, 0.35);
  transition:
    color var(--motion-fast) var(--ease-standard),
    background var(--motion-fast) var(--ease-standard),
    box-shadow var(--motion-fast) var(--ease-standard);
}

.aside-edge:hover {
  color: var(--fg);
  background: #111;
}

.aside-edge:focus-visible {
  outline: none;
  box-shadow: var(--focus-ring);
}

@media (min-width: 810px) {
  .aside {
    position: sticky;
  }

  .aside-edge {
    display: inline-flex;
  }

  .aside.collapsed .aside-edge {
    right: 50%;
    transform: translate(50%, -50%);
  }
}

.aside-nav {
  display: none;
  flex-direction: column;
  gap: var(--space-1);
  padding: var(--space-5) var(--space-4) var(--space-4);
  box-shadow: 0 1px 0 var(--border-soft);
}

.aside-label {
  margin: 0 0 var(--space-2);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--meta);
}

.aside-link {
  display: flex;
  align-items: center;
  min-height: 44px;
  padding: 0 var(--space-3);
  border-radius: var(--radius-sm);
  color: var(--muted);
  text-decoration: none;
  font-size: var(--text-sm);
  letter-spacing: -0.15px;
  transition:
    color var(--motion-fast) var(--ease-standard),
    background var(--motion-fast) var(--ease-standard),
    box-shadow var(--motion-fast) var(--ease-standard);
}

.aside-link:hover {
  color: var(--fg);
  background: rgba(255, 255, 255, 0.04);
  text-decoration: none;
}

.aside-link.router-link-active {
  color: var(--accent);
  box-shadow: 0 0 0 1px rgba(0, 153, 255, 0.35);
  background: rgba(0, 153, 255, 0.08);
}

.aside-slot {
  min-width: 0;
}

.scrim {
  display: none;
}

.main {
  width: min(100% - 2 * var(--space-6), var(--container-max));
  margin: 0 auto;
  padding: var(--space-8) 0 var(--space-12);
}

.has-aside .main {
  width: auto;
  max-width: none;
  margin: 0;
  padding: var(--space-6) var(--space-6) var(--space-12);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@keyframes slide-in {
  from {
    opacity: 0;
    transform: translateX(-8px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@media (max-width: 809px) {
  .top,
  .main,
  .has-aside .main {
    padding-left: var(--space-4);
    padding-right: var(--space-4);
  }

  .has-aside .menu-btn {
    display: inline-flex;
  }

  .has-aside .rail-btn,
  .aside-edge {
    display: none;
  }

  .has-aside .top-nav {
    display: none;
  }

  .has-aside .body,
  .aside-collapsed.has-aside .body {
    display: block;
    grid-template-columns: unset;
  }

  .has-aside .aside-nav {
    display: flex;
  }

  .aside,
  .aside.collapsed {
    position: fixed;
    top: 0;
    left: 0;
    z-index: 40;
    width: min(320px, 88vw);
    height: 100dvh;
    transform: translateX(-105%);
    transition: transform var(--motion-base) var(--ease-standard);
    animation: none;
    overflow: auto;
    box-shadow:
      1px 0 0 var(--border-soft),
      0 20px 48px rgba(0, 0, 0, 0.55);
  }

  .aside.collapsed .aside-inner {
    opacity: 1;
    pointer-events: auto;
  }

  .aside.open {
    transform: translateX(0);
  }

  .scrim {
    display: block;
    position: fixed;
    inset: 0;
    z-index: 35;
    background: rgba(0, 0, 0, 0.55);
    opacity: 0;
    pointer-events: none;
    transition: opacity var(--motion-base) var(--ease-standard);
  }

  .scrim.show {
    opacity: 1;
    pointer-events: auto;
  }
}

@media (min-width: 810px) and (max-width: 1199px) {
  .has-aside {
    --sidebar-width: 260px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .aside,
  .aside-inner,
  .scrim,
  .menu-icon,
  .menu-icon::before,
  .menu-icon::after,
  .rail-chev,
  .edge-chev,
  .aside-link,
  .nav-link,
  .has-aside .body {
    animation: none;
    transition: none;
  }
}
</style>
