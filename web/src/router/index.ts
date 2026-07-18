import { createRouter, createWebHistory } from 'vue-router'
import { useProfile } from '@/composables/useProfile'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'welcome',
      component: () => import('@/views/WelcomeView.vue'),
      meta: { public: true },
    },
    {
      path: '/projects',
      name: 'projects',
      component: () => import('@/views/ProjectsView.vue'),
      meta: { requiresProfile: true },
    },
    {
      path: '/projects/:name',
      name: 'project',
      component: () => import('@/views/ProjectView.vue'),
      meta: { requiresProfile: true },
    },
    {
      path: '/teachers',
      name: 'teachers',
      component: () => import('@/views/TeachersView.vue'),
      // Allow monitoring train-dqn before workshop onboarding.
      meta: { public: true },
    },
    {
      path: '/evolve',
      name: 'evolve',
      component: () => import('@/views/EvolveView.vue'),
      meta: { public: true },
    },
  ],
  scrollBehavior(to) {
    if (to.hash) return { el: to.hash, behavior: 'smooth' }
    return { top: 0 }
  },
})

router.beforeEach(async (to) => {
  const { loaded, isOnboarded, refresh } = useProfile()
  if (!loaded.value) {
    await refresh()
  }

  if (to.meta.requiresProfile && !isOnboarded.value) {
    return { name: 'welcome', hash: '#credentials' }
  }

  if (to.name === 'welcome' && isOnboarded.value && !to.hash) {
    return { name: 'projects' }
  }

  return true
})

export default router
