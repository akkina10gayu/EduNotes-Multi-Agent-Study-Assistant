import { apiClient } from './client'
import type { Document, SearchResult } from '@/types'

export async function listDocuments(keyword?: string): Promise<{ success: boolean; documents: Document[]; count: number }> {
  const params = keyword ? `?keyword=${encodeURIComponent(keyword)}` : ''
  return apiClient(`/documents${params}`)
}

export async function searchDocumentsSemantic(query: string, k?: number): Promise<{ success: boolean; documents: Document[]; count: number }> {
  const params = new URLSearchParams({ query })
  if (k) params.set('k', String(k))
  return apiClient(`/documents/search?${params}`)
}

export async function getDocument(docId: string): Promise<{ success: boolean; document: Document }> {
  return apiClient(`/documents/${docId}`)
}

export async function searchKnowledgeBase(query: string, k: number = 5, threshold: number = 0.7): Promise<{ success: boolean; results: SearchResult[]; count: number }> {
  return apiClient('/search-kb', {
    method: 'POST',
    body: JSON.stringify({ query, k, threshold }),
  })
}

export async function updateKnowledgeBase(documents: Array<{ content: string; title?: string; topic?: string; source?: string; url?: string }>): Promise<{ success: boolean; documents_added: number }> {
  return apiClient('/update-kb', {
    method: 'POST',
    body: JSON.stringify({ documents }),
  })
}

export async function deleteDocument(docId: string): Promise<{ success: boolean; message: string }> {
  return apiClient(`/documents/${docId}`, { method: 'DELETE' })
}
