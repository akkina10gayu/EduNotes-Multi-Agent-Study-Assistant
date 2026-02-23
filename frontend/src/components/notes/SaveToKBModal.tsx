'use client'

import { useState } from 'react'
import { updateKnowledgeBase } from '@/lib/api/kb'
import type { GenerateNotesResponse } from '@/types'

interface SaveToKBModalProps {
  note: GenerateNotesResponse
  currentNotes: string
  onClose: () => void
  onSuccess: () => void
}

export default function SaveToKBModal({ note, currentNotes, onClose, onSuccess }: SaveToKBModalProps) {
  const [title, setTitle] = useState(note.query?.slice(0, 100) || 'Untitled')
  const [topic, setTopic] = useState(note.query_type === 'topic' ? note.query || 'general' : 'general')
  const [source, setSource] = useState(note.query_type || 'manual')
  const [url, setUrl] = useState(note.query_type === 'url' ? note.query || '' : '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSave = async () => {
    if (!title.trim() || !topic.trim()) {
      setError('Title and topic are required')
      return
    }
    setSaving(true)
    setError(null)
    try {
      await updateKnowledgeBase([{
        content: currentNotes,
        title: title.trim(),
        topic: topic.trim(),
        source: source.trim(),
        url: url.trim(),
      }])
      onSuccess()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-md w-full mx-4 space-y-4">
        <h3 className="text-lg font-semibold text-white">Save to Knowledge Base</h3>

        <div className="space-y-3">
          <div>
            <label className="text-sm text-gray-400">Title *</label>
            <input value={title} onChange={(e) => setTitle(e.target.value)} className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]" />
          </div>
          <div>
            <label className="text-sm text-gray-400">Topic *</label>
            <input value={topic} onChange={(e) => setTopic(e.target.value)} className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]" />
          </div>
          <div>
            <label className="text-sm text-gray-400">Source</label>
            <input value={source} onChange={(e) => setSource(e.target.value)} className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]" />
          </div>
          <div>
            <label className="text-sm text-gray-400">URL</label>
            <input value={url} onChange={(e) => setUrl(e.target.value)} className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]" />
          </div>
        </div>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <div className="flex justify-end gap-3">
          <button onClick={onClose} className="px-4 py-2 text-sm text-gray-300 bg-gray-800 rounded-lg hover:bg-gray-700">Cancel</button>
          <button onClick={handleSave} disabled={saving} className="px-4 py-2 text-sm text-white bg-[#6CA0DC] rounded-lg hover:bg-[#5a8ec4] disabled:opacity-50">
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}
