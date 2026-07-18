import { api, apiText } from '@/api/client'

export type EvolveGeneration = {
  generation: number
  primary?: number | null
  num_pairs?: number | null
  loss?: number | null
  promoted?: boolean | null
  gate_reason?: string | null
  has_adapter?: boolean
  has_train_metrics?: boolean
}

export type EvolveJob = {
  run_id: string
  running: boolean
  pid: number | null
  path: string | null
  log_path: string | null
  game?: string | null
  model?: string | null
  backend?: string | null
  champion?: number | null
  generations: EvolveGeneration[]
  last_primary?: number | null
  phase?: string
  phase_generation?: number
  target_generations?: number | null
  start_generation?: number | null
  end_generation?: number | null
  started_at?: number | null
  train_step?: number | null
  train_total_steps?: number | null
  train_kl?: number | null
  train_entropy?: number | null
  train_reward?: number | null
}

export type EvolveMetricPoint = {
  split?: string
  step?: number
  epoch?: number
  loss?: number | null
  generation?: number
  primary?: number | null
  learning_rate?: number | null
  num_pairs?: number | null
  ts?: string
}

export type EvolveMetrics = {
  run_id: string
  log_path: string | null
  path: string | null
  /** True when the evolve OS process / playground job slot is alive. */
  running?: boolean
  pid?: number | null
  /** Current champion generation from registry.json (latest promoted). */
  champion?: number | null
  /** Unix seconds — evolve.log ctime (workshop ETA). */
  started_at?: number | null
  target_generations?: number | null
  start_generation?: number | null
  end_generation?: number | null
  train_step?: number | null
  train_total_steps?: number | null
  train_kl?: number | null
  train_entropy?: number | null
  train_reward?: number | null
  /** Last exception from evolve.log traceback when the process is dead. */
  crash_error?: string | null
  train: EvolveMetricPoint[]
  eval: EvolveMetricPoint[]
  generations: EvolveGeneration[]
  points: EvolveMetricPoint[]
  phase?: string
  phase_generation?: number
}

export function listEvolveJobs() {
  return api<{ jobs: EvolveJob[] }>('/api/evolve/jobs')
}

export function getEvolveMetrics(runId: string) {
  return api<EvolveMetrics>(`/api/evolve/${encodeURIComponent(runId)}/metrics`)
}

export function getEvolveLog(runId: string) {
  return apiText(`/api/evolve/${encodeURIComponent(runId)}/log`)
}

export function stopEvolveJob(runId: string) {
  return api<{ ok: boolean; run_id: string; pid: number }>(
    `/api/evolve/${encodeURIComponent(runId)}/stop`,
    { method: 'POST' },
  )
}
