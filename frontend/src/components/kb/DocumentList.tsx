'use client'

import { useState, useEffect } from 'react'
import { listDocuments, searchDocumentsSemantic, getDocument } from '@/lib/api/kb'
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
              <button
                key={doc.id}
                onClick={() => handleSelectDoc(doc)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                  selectedDoc?.id === doc.id ? 'bg-[#6CA0DC]/20 border border-[#6CA0DC]/30' : 'bg-gray-800 hover:bg-gray-700 border border-transparent'
                }`}
              >
                <p className="text-white font-medium">{doc.title}</p>
                <p className="text-xs text-gray-500">{doc.topic} | {formatDate(doc.created_at)}{doc.word_count ? ` | ${doc.word_count} words` : ''}</p>
              </button>
            ))}
          </div>

          {loadingDoc && <LoadingSpinner message="Loading document..." size="sm" />}

          {selectedDoc && !loadingDoc && <DocumentViewer document={selectedDoc} />}
        </>
      )}
    </div>
  )
}
