export class ApiError extends Error {
  status: number
  body: unknown

  constructor(status: number, message: string, body?: unknown) {
    super(message)
    this.status = status
    this.body = body
  }
}

async function parseBody(res: Response): Promise<unknown> {
  const text = await res.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

export async function api<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const res = await fetch(path, { ...init, headers })
  const body = await parseBody(res)

  if (!res.ok) {
    const message =
      typeof body === 'object' && body && 'error' in body
        ? String((body as { error: unknown }).error)
        : res.statusText || `HTTP ${res.status}`
    throw new ApiError(res.status, message, body)
  }

  return body as T
}

export async function apiText(path: string): Promise<string> {
  const res = await fetch(path)
  if (!res.ok) {
    throw new ApiError(res.status, res.statusText)
  }
  return res.text()
}
