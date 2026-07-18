import { computed, ref } from 'vue'
import { getProfile, saveProfile, type Profile } from '@/api/profile'

const profile = ref<Profile | null>(null)
const loaded = ref(false)
const loading = ref(false)
const error = ref<string | null>(null)

export function useProfile() {
  async function refresh() {
    loading.value = true
    error.value = null
    try {
      profile.value = await getProfile()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to load profile'
      profile.value = null
    } finally {
      loading.value = false
      loaded.value = true
    }
  }

  async function save(name: string, hfToken?: string | null) {
    loading.value = true
    error.value = null
    try {
      profile.value = await saveProfile(name, hfToken)
      return profile.value
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to save profile'
      throw err
    } finally {
      loading.value = false
      loaded.value = true
    }
  }

  // Name-only signup is enough to enter the app; token only unlocks publish.
  const isOnboarded = computed(() => Boolean(profile.value?.name))
  const hasToken = computed(() => Boolean(profile.value?.has_token))

  return {
    profile,
    loaded,
    loading,
    error,
    isOnboarded,
    hasToken,
    refresh,
    save,
  }
}
