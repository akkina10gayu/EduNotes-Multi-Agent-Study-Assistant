import { createClient } from '@/lib/supabase/client'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1'

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const supabase = createClient()
  const { data: { session } } = await supabase.auth.getSession()

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(session ? { 'Authorization': `Bearer ${session.access_token}` } : {}),
      ...options.headers,
    },
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `API error: ${response.status}`)
  }

  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('text/plain')) {
    return response.text() as Promise<T>
  }

  return response.json()
}

export async function apiClientFormData<T>(
  endpoint: string,
  formData: FormData
): Promise<T> {
  const supabase = createClient()
  const { data: { session } } = await supabase.auth.getSession()

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: {
      ...(session ? { 'Authorization': `Bearer ${session.access_token}` } : {}),
    },
    body: formData,
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `API error: ${response.status}`)
  }

  return response.json()
}
