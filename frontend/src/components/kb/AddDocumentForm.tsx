'use client'

import { useState } from 'react'
import { updateKnowledgeBase } from '@/lib/api/kb'

export default function AddDocumentForm() {
  const [title, setTitle] = useState('')
  const [topic, setTopic] = useState('general')
  const [source, setSource] = useState('manual')
  const [content, setContent] = useState('')
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const handleSubmit = async () => {
    if (!title.trim() || !content.trim()) {
      setMessage({ type: 'error', text: 'Title and content are required' })
      return
    }
    setSaving(true)
    setMessage(null)
    try {
      await updateKnowledgeBase([{ content: content.trim(), title: title.trim(), topic: topic.trim(), source: source.trim() }])
      setMessage({ type: 'success', text: 'Document added to Knowledge Base!' })
      setTitle('')
      setContent('')
    } catch (e) {
      setMessage({ type: 'error', text: e instanceof Error ? e.message : 'Failed to add document' })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <h2 className="text-lg font-semibold text-white">Add Document</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div>
          <label className="text-sm text-gray-400">Title *</label>
          <input value={title} onChange={(e) => setTitle(e.target.value)} className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]" />
        </div>
        <div>
          <label className="text-sm text-gray-400">Topic</label>
          <input value={topic} onChange={(e) => setTopic(e.target.value)} className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]" />
        </div>
        <div>
          <label className="text-sm text-gray-400">Source</label>
          <input value={source} onChange={(e) => setSource(e.target.value)} className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]" />
        </div>
      </div>

      <div>
        <label className="text-sm text-gray-400">Content *</label>
        <textarea value={content} onChange={(e) => setContent(e.target.value)} rows={6} className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm resize-y focus:outline-none focus:border-[#6CA0DC]" placeholder="Paste document content here..." />
      </div>

      {message && (
        <div className={`text-sm p-3 rounded-lg ${message.type === 'success' ? 'bg-green-900/20 text-green-400 border border-green-800' : 'bg-red-900/20 text-red-400 border border-red-800'}`}>
          {message.text}
        </div>
      )}

      <button onClick={handleSubmit} disabled={saving} className="px-6 py-2 bg-[#6CA0DC] text-white rounded-lg text-sm hover:bg-[#5a8ec4] disabled:opacity-50">
        {saving ? 'Adding...' : 'Add Document'}
      </button>
    </div>
  )
}
