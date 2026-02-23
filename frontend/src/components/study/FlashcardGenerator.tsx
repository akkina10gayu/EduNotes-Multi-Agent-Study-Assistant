'use client'

import { useState, useEffect } from 'react'
import { generateFlashcards, listFlashcardSets, getFlashcardSet, deleteFlashcardSet } from '@/lib/api/study'
import type { FlashcardSet } from '@/types'
import LoadingSpinner from '@/components/ui/LoadingSpinner'

interface Props { onSetLoaded: (set: FlashcardSet) => void }

export default function FlashcardGenerator({ onSetLoaded }: Props) {
  const [topic, setTopic] = useState('')
  const [content, setContent] = useState('')
  const [numCards, setNumCards] = useState(10)
  const [generating, setGenerating] = useState(false)
  const [sets, setSets] = useState<FlashcardSet[]>([])
  const [selectedSetId, setSelectedSetId] = useState('')
  const [error, setError] = useState<string | null>(null)

  const loadSets = async () => {
    try {
      const data = await listFlashcardSets()
      setSets(data.sets || [])
    } catch { /* ignore */ }
  }

  useEffect(() => { loadSets() }, [])

  const handleGenerate = async () => {
    if (!topic.trim() || !content.trim()) { setError('Topic and content are required'); return }
    setGenerating(true); setError(null)
    try {
      const data = await generateFlashcards(content.trim(), topic.trim(), numCards)
      if (data.success && data.flashcard_set) {
        onSetLoaded(data.flashcard_set)
        loadSets()
      } else { setError('Failed to generate flashcards') }
    } catch (e) { setError(e instanceof Error ? e.message : 'Error') }
    finally { setGenerating(false) }
  }

  const handleLoadSet = async () => {
    if (!selectedSetId) return
    try {
      const data = await getFlashcardSet(selectedSetId)
      if (data.flashcard_set) onSetLoaded(data.flashcard_set)
    } catch (e) { setError(e instanceof Error ? e.message : 'Error loading set') }
  }

  const handleDeleteSet = async (id: string) => {
    try { await deleteFlashcardSet(id); loadSets() } catch { /* ignore */ }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <h3 className="font-semibold text-white">Generate Flashcards</h3>

      <input value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="Topic" className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]" />

      <div className="flex items-center gap-2 text-sm">
        <span className="text-gray-500">Cards:</span>
        <input type="range" min={5} max={20} value={numCards} onChange={(e) => setNumCards(Number(e.target.value))} className="flex-1 accent-[#6CA0DC]" />
        <span className="text-gray-400 w-6">{numCards}</span>
      </div>

      <textarea value={content} onChange={(e) => setContent(e.target.value)} rows={4} placeholder="Paste content to generate flashcards from..." className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm resize-none focus:outline-none focus:border-[#6CA0DC]" />

      <button onClick={handleGenerate} disabled={generating} className="w-full py-2 bg-[#6CA0DC] text-white rounded-lg text-sm hover:bg-[#5a8ec4] disabled:opacity-50">
        {generating ? 'Generating...' : 'Generate Flashcards'}
      </button>

      {generating && <LoadingSpinner size="sm" message="Generating flashcards..." />}
      {error && <p className="text-sm text-red-400">{error}</p>}

      {/* Load Existing */}
      {sets.length > 0 && (
        <div className="border-t border-gray-800 pt-4 space-y-2">
          <h4 className="text-sm font-medium text-gray-400">Load Existing Set</h4>
          <div className="flex gap-2">
            <select value={selectedSetId} onChange={(e) => setSelectedSetId(e.target.value)} className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm">
              <option value="">Select a set...</option>
              {sets.map(s => <option key={s.id} value={s.id}>{s.name} ({s.card_count} cards)</option>)}
            </select>
            <button onClick={handleLoadSet} disabled={!selectedSetId} className="px-3 py-2 bg-gray-700 text-white rounded-lg text-sm hover:bg-gray-600 disabled:opacity-50">Load</button>
          </div>
          {sets.map(s => (
            <div key={s.id} className="flex items-center justify-between text-xs text-gray-500 px-1">
              <span>{s.name}</span>
              <button onClick={() => handleDeleteSet(s.id)} className="text-red-500 hover:text-red-400">Delete</button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
