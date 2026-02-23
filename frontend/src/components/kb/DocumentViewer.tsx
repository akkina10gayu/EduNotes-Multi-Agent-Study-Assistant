'use client'

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Document } from '@/types'
import { formatDate } from '@/lib/utils/formatters'

interface DocumentViewerProps {
  document: Document
}

export default function DocumentViewer({ document: doc }: DocumentViewerProps) {
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(doc.content || '')
      setSuccessMsg('Copied!')
      setTimeout(() => setSuccessMsg(null), 3000)
    } catch { setSuccessMsg('Failed to copy') }
  }

  const handleDownload = () => {
    const blob = new Blob([doc.content || ''], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = globalThis.document.createElement('a')
    a.href = url
    a.download = `${doc.title?.replace(/\s+/g, '_') || 'document'}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3 text-xs text-gray-400">
        <span>Topic: <span className="text-gray-300">{doc.topic}</span></span>
        <span>Date: <span className="text-gray-300">{formatDate(doc.created_at)}</span></span>
        {doc.word_count && <span>Words: <span className="text-gray-300">{doc.word_count}</span></span>}
      </div>

      {successMsg && (
        <div className="bg-green-900/20 border border-green-800 rounded-lg p-2 text-sm text-green-400">{successMsg}</div>
      )}

      <div className="notes-container">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {doc.content || 'No content available'}
        </ReactMarkdown>
      </div>

      <div className="flex gap-2">
        <button onClick={handleDownload} className="px-3 py-1.5 text-sm bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700">Download</button>
        <button onClick={handleCopy} className="px-3 py-1.5 text-sm bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700">Copy</button>
      </div>
    </div>
  )
}
