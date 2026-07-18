import { api, apiText } from '@/api/client'

export type DqnJob = {
  game: string
  running: boolean
  pid: number | null
  checkpoint: string | null
  metrics_path: string | null
  log_path: string | null
  checkpoint_bytes?: number
  checkpoint_mtime?: number
  last_decisions?: number | null
  last_train_reward?: number | null
  last_eval_reward?: number | null
  last_loss?: number | null
}

export type DqnMetricPoint = {
  split?: string
  decisions?: number
  episodes?: number
  eps?: number
  mean_ep_reward?: number | null
  loss?: number | null
  ts?: string
  done?: boolean
}

export type DqnMetrics = {
  game: string
  metrics_path: string | null
  log_path: string | null
  train: DqnMetricPoint[]
  eval: DqnMetricPoint[]
  points: DqnMetricPoint[]
}

export function listDqnJobs() {
  return api<{ jobs: DqnJob[] }>('/api/dqn/jobs')
}

export function getDqnMetrics(game: string) {
  return api<DqnMetrics>(`/api/dqn/${encodeURIComponent(game)}/metrics`)
}

export function getDqnLog(game: string) {
  return apiText(`/api/dqn/${encodeURIComponent(game)}/log`)
}

export function stopDqnJob(game: string) {
  return api<{ ok: boolean; game: string; pid: number }>(
    `/api/dqn/${encodeURIComponent(game)}/stop`,
    { method: 'POST' },
  )
}
