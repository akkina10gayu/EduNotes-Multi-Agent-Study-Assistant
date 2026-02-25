'use client'

import { useState, useEffect } from 'react'
import { listDocuments, searchDocumentsSemantic, getDocument, deleteDocument } from '@/lib/api/kb'
import type { Document } from '@/types'
import LoadingSpinner from '@/components/ui/LoadingSpinner'
import DocumentViewer from './DocumentViewer'
import { formatDate } from '@/lib/utils/formatters'

export default function DocumentList() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [searchKeyword, setSearchKeyword] = useState('')
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingDoc, setLoadingDoc] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const fetchDocuments = async (keyword?: string) => {
    setLoading(true)
    try {
      const data = keyword
        ? await searchDocumentsSemantic(keyword)
        : await listDocuments()
      setDocuments(data.documents || [])
    } catch (e) {
      console.error('Failed to load documents', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchDocuments() }, [])

  const handleSearch = () => {
    if (searchKeyword.trim()) {
      fetchDocuments(searchKeyword.trim())
    } else {
      fetchDocuments()
    }
  }

  const handleDelete = async (e: React.MouseEvent, docId: string) => {
    e.stopPropagation()
    if (!confirm('Delete this document? This cannot be undone.')) return
    setDeletingId(docId)
    try {
      await deleteDocument(docId)
      setDocuments(prev => prev.filter(d => d.id !== docId))
      if (selectedDoc?.id === docId) setSelectedDoc(null)
    } catch (err) {
      console.error('Failed to delete document', err)
    } finally {
      setDeletingId(null)
    }
  }

  const handleSelectDoc = async (doc: Document) => {
    setLoadingDoc(true)
    try {
      const data = await getDocument(doc.id)
      setSelectedDoc(data.document)
    } catch (e) {
      console.error('Failed to load document', e)
    } finally {
      setLoadingDoc(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="flex gap-2">
        <input
          value={searchKeyword}
          onChange={(e) => setSearchKeyword(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="Search documents..."
          className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]"
        />
        <button onClick={handleSearch} className="px-4 py-2 bg-[#6CA0DC] text-white rounded-lg text-sm hover:bg-[#5a8ec4]">Search</button>
      </div>

      {loading ? (
        <LoadingSpinner message="Loading documents..." />
      ) : (
        <>
          <p className="text-sm text-gray-500">{documents.length} document{documents.length !== 1 ? 's' : ''} found</p>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {documents.map(doc => (
              <div
                key={doc.id}
                onClick={() => handleSelectDoc(doc)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer flex items-center justify-between gap-2 ${
                  selectedDoc?.id === doc.id ? 'bg-[#6CA0DC]/20 border border-[#6CA0DC]/30' : 'bg-gray-800 hover:bg-gray-700 border border-transparent'
                }`}
              >
                <div className="min-w-0 flex-1">
                  <p className="text-white font-medium truncate">{doc.title}</p>
                  <p className="text-xs text-gray-500">{doc.topic} | {formatDate(doc.created_at)}</p>
                </div>
                <button
                  onClick={(e) => handleDelete(e, doc.id)}
                  disabled={deletingId === doc.id}
                  className="shrink-0 px-2 py-1 text-xs text-red-400 hover:bg-red-900/30 rounded opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity"
                  style={{ opacity: deletingId === doc.id ? 0.5 : undefined }}
                  title="Delete document"
                >
                  {deletingId === doc.id ? '...' : 'Delete'}
                </button>
              </div>
            ))}
          </div>

          {loadingDoc && <LoadingSpinner message="Loading document..." size="sm" />}

          {selectedDoc && !loadingDoc && <DocumentViewer document={selectedDoc} />}
        </>
      )}
    </div>
  )
}
