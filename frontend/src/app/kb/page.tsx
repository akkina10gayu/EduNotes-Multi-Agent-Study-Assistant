'use client'

import { useState } from 'react'
import DocumentList from '@/components/kb/DocumentList'
import SemanticSearch from '@/components/kb/SemanticSearch'
import AddDocumentForm from '@/components/kb/AddDocumentForm'

export default function KBPage() {
  const [mode, setMode] = useState<'browse' | 'search'>('browse')

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Knowledge Base</h1>

      {/* Mode toggle */}
      <div className="flex gap-2">
        {(['browse', 'search'] as const).map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-lg text-sm transition-colors ${
              mode === m ? 'bg-[#6CA0DC]/20 text-[#6CA0DC]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {m === 'browse' ? 'Browse Documents' : 'Search Vector DB'}
          </button>
        ))}
      </div>

      {/* Content */}
      {mode === 'browse' ? <DocumentList /> : <SemanticSearch />}

      {/* Add Document section */}
      <AddDocumentForm />
    </div>
  )
}
