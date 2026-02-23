import { apiClient, apiClientFormData } from './client'
import type { GenerateNotesRequest, GenerateNotesResponse } from '@/types'

export async function generateNotes(params: GenerateNotesRequest): Promise<GenerateNotesResponse> {
  return apiClient<GenerateNotesResponse>('/generate-notes', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export async function processPdf(formData: FormData): Promise<GenerateNotesResponse> {
  return apiClientFormData<GenerateNotesResponse>('/process-pdf', formData)
}

export async function getTopics(): Promise<{ success: boolean; topics: string[]; total_topics: number }> {
  return apiClient('/topics')
}
