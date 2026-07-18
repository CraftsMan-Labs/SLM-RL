import { api, ApiError } from './client'

export type Profile = {
  name: string
  hf_username: string | null
  has_token: boolean
  token_masked: string | null
  created_at: string
}

export async function getProfile(): Promise<Profile | null> {
  try {
    return await api<Profile>('/api/profile')
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null
    throw err
  }
}

export async function saveProfile(name: string, hfToken?: string | null): Promise<Profile> {
  const token = hfToken?.trim() || null
  return api<Profile>('/api/profile', {
    method: 'POST',
    body: JSON.stringify({ name, hf_token: token }),
  })
}
