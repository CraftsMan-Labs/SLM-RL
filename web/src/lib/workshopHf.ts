/** Canonical workshop Hub org + games with published packs. */
export const WORKSHOP_HF_ORG = 'BLANK'

const WORKSHOP_GAMES = new Set([
  'boxing',
  'space-invaders',
  'freeway',
  'demon-attack',
])

/** Dataset pack + LoRA model share the same repo id: `{org}/slm-rl-{game}`. */
export function workshopPackUrl(game: string): string {
  if (!WORKSHOP_GAMES.has(game)) return ''
  return `${WORKSHOP_HF_ORG}/slm-rl-${game}`
}

/** Dedicated DQN teacher: `{org}/slm-rl-{game}-dqn`. */
export function workshopDqnUrl(game: string): string {
  if (!WORKSHOP_GAMES.has(game)) return ''
  return `${WORKSHOP_HF_ORG}/slm-rl-${game}-dqn`
}
