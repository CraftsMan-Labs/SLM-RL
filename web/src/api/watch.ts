/** URL helpers for the playground watch/theater streaming endpoints. */

export type WatchEvent = {
  episode_id?: string | null
  generation?: number | null
  step_idx?: number | null
  parsed_action?: string | null
  completion?: string | null
  parse_status?: string | null
  reward?: number | null
  cum_reward?: number | null
  terminated?: boolean | null
  truncated?: boolean | null
  outcome?: string | null
  monitor_flags?: Record<string, unknown> | null
  model_id?: string | null
  seed?: number | null
  observed?: string | null
}

export function watchEventsUrl(name: string, gen?: number | null): string {
  const base = `/watch/${encodeURIComponent(name)}/events`
  if (gen === undefined || gen === null) return base
  return `${base}?gen=${encodeURIComponent(String(gen))}`
}

export function watchFramesUrl(
  name: string,
  episodeId: string,
  gen?: number | null,
  /** Bump on each Watch click so the browser opens a fresh stream from step 0. */
  restartToken?: number | string | null,
): string {
  const params = new URLSearchParams({ episode: episodeId })
  if (gen !== undefined && gen !== null) {
    params.set('gen', String(gen))
  }
  if (restartToken !== undefined && restartToken !== null && restartToken !== '') {
    params.set('t', String(restartToken))
  }
  return `/watch/${encodeURIComponent(name)}/frames?${params.toString()}`
}

/** Full live-viewer page for one theater side (base | champion | play). */
export function theaterViewerUrl(name: string, side: 'base' | 'champion' | 'play') {
  return `/theater/${encodeURIComponent(name)}/${side}/`
}

export type TheaterSide = 'base' | 'champion' | 'play'

export function theaterEventsUrl(name: string, side: TheaterSide) {
  return `/theater/${encodeURIComponent(name)}/${side}/events`
}

export function theaterFramesUrl(
  name: string,
  side: TheaterSide,
  episodeId: string,
  gen?: number | null,
  restartToken?: number | string | null,
) {
  const params = new URLSearchParams({ episode: episodeId })
  if (gen !== undefined && gen !== null) {
    params.set('gen', String(gen))
  }
  if (restartToken !== undefined && restartToken !== null && restartToken !== '') {
    params.set('t', String(restartToken))
  }
  return `/theater/${encodeURIComponent(name)}/${side}/frames?${params.toString()}`
}
