import { api, apiText } from './client'

export type Project = {
  name: string
  episodes: number
  mean_score: number | null
  median_score: number | null
  max_score: number | null
  action_mix: Record<string, number>
  intervention_episodes: number
  status: 'no_data' | 'running' | 'complete' | 'stopped' | string
  active_jobs?: string[]
  win_rate: number | null
  model: string | null
  backend: string | null
  game: string | null
  knob_values?: Record<string, unknown>
  /** Public HF dataset pack for evolve warm-start (e.g. BLANK/slm-rl-boxing) */
  dataset_url?: string | null
  dqn_url?: string | null
  /** Public HF model repo with adapter/ (published SFT LoRA) */
  adapter_url?: string | null
}

export type KnobSchema = {
  key: string
  label: string
  type: 'int' | 'float' | 'enum' | string
  target?: string
  default: unknown
  min?: number
  max?: number
  choices?: string[]
  help?: { title: string; body: string }
}

export type GamesResponse = {
  games: string[]
  default: string
}

export type Hardware = {
  tier: string
  model: string
  backend: string
  train: string
  presets: Array<{
    model: string
    backend: string
    experimental?: boolean
    label?: string
  }>
  host: {
    os: string
    ram_gb: number
    cuda_vram_gb: number | null
    has_mps: boolean
  }
}

export type CreateProjectBody = {
  name: string
  game: string
  knob_values?: Record<string, unknown>
  reward_code?: string
  agent?: string
  episodes?: number
  seed?: number
  model?: string | null
  backend?: string | null
  /** HF dataset/model repo with dqn.pt — required when teacher=dqn unless a local pack exists */
  dqn_url?: string | null
  /** Public HF dataset pack (demos + sft) for evolve warm-start */
  dataset_url?: string | null
  /** Public HF model repo with adapter/ (published SFT LoRA) */
  adapter_url?: string | null
  /** false = create only; run from workspace UI */
  launch?: boolean
}

export function listProjects() {
  return api<Project[]>('/api/experiments')
}

export function listGames() {
  return api<GamesResponse>('/api/games')
}

export function getKnobs(game: string) {
  return api<KnobSchema[]>(`/api/knobs?game=${encodeURIComponent(game)}`)
}

export function getHardware() {
  return api<Hardware>('/api/hardware')
}

export function getRewardTemplate() {
  return api<{ template: string }>('/api/reward-template')
}

export function createProject(body: CreateProjectBody) {
  return api<{ name: string; run_id: string; game: string; warnings: string[] }>(
    '/api/experiments',
    { method: 'POST', body: JSON.stringify({ launch: false, ...body }) },
  )
}

export function runProject(
  name: string,
  opts: { agent?: string; episodes?: number; seed?: number } = {},
) {
  return api<{ name: string; run_id: string }>(
    `/api/experiments/${encodeURIComponent(name)}/rollout`,
    {
      method: 'POST',
      body: JSON.stringify({
        agent: opts.agent ?? 'solver',
        episodes: opts.episodes ?? 30,
        seed: opts.seed ?? 20000,
      }),
    },
  )
}

export function updateProjectKnobs(name: string, knob_values: Record<string, unknown>) {
  return api<{ name: string; knob_values: Record<string, unknown> }>(
    `/api/experiments/${encodeURIComponent(name)}/knobs`,
    {
      method: 'POST',
      body: JSON.stringify({ knob_values }),
    },
  )
}

export function evolveProject(
  name: string,
  generations = 2,
  opts: {
    dataset_url?: string | null
    dqn_url?: string | null
    adapter_url?: string | null
  } = {},
) {
  return api<{ name: string; run_id: string }>(
    `/api/experiments/${encodeURIComponent(name)}/evolve`,
    {
      method: 'POST',
      body: JSON.stringify({
        generations,
        dataset_url: opts.dataset_url || null,
        dqn_url: opts.dqn_url || null,
        adapter_url: opts.adapter_url || null,
      }),
    },
  )
}

export function stopProject(name: string, kinds?: string[]) {
  return api<{ name: string; stopped: string[] }>(
    `/api/experiments/${encodeURIComponent(name)}/stop`,
    {
      method: 'POST',
      body: JSON.stringify(kinds?.length ? { kinds } : {}),
    },
  )
}

export function launchTheater(name: string, episodes = 10) {
  return api<{ name: string; run_id: string }>(
    `/api/experiments/${encodeURIComponent(name)}/theater`,
    { method: 'POST', body: JSON.stringify({ episodes }) },
  )
}

export type TheaterSideScore = {
  episodes?: number | null
  win_rate?: number | null
  mean_score?: number | null
  status?: string | null
}

export type TheaterRunStatus = {
  phase?: string | null
  side?: string | null
  episode?: number | null
  completed?: number | null
  episodes?: number | null
  generation?: number | null
  champion_generation?: number | null
  error?: string | null
  message?: string | null
}

export type TheaterScores = {
  base?: TheaterSideScore | null
  champion?: TheaterSideScore | null
  run?: TheaterRunStatus | null
}

export function getTheaterScores(name: string) {
  return api<TheaterScores>(
    `/api/experiments/${encodeURIComponent(name)}/theater-scores`,
  )
}

export function publishProject(name: string) {
  return api(`/api/experiments/${encodeURIComponent(name)}/publish`, {
    method: 'POST',
    body: JSON.stringify({}),
  })
}

export function getProjectLog(name: string, kind: 'rollout' | 'evolve' | 'theater' = 'rollout') {
  return apiText(
    `/api/experiments/${encodeURIComponent(name)}/log?kind=${encodeURIComponent(kind)}`,
  )
}

export type PackInfo = {
  slug: string
  path: string
  game: string | null
  n_episodes: number | null
  n_episodes_raw?: number | null
  selection_quantile?: number | null
  has_dqn: boolean | null
  created_at: string | null
  hf_repo?: string | null
  repo_hint: string
}

export function listPacks() {
  return api<{ packs: PackInfo[]; baking: boolean }>('/api/packs')
}

export function getBakeLog() {
  return apiText('/api/packs/log')
}

export function bakePack(body: {
  game?: string | null
  all?: boolean
  episodes?: number
  dqn_decisions?: number
  selection_quantile?: number
  device?: string
  push?: string | null
  push_prefix?: string | null
}) {
  return api<{ ok: boolean; baking: boolean; push?: string | null; push_prefix?: string | null }>(
    '/api/packs/bake',
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  )
}
