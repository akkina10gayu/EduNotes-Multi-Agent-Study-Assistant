'use client'

import { useState, useEffect } from 'react'
import { listFlashcardSets, exportFlashcardsToAnki, exportAllFlashcardsToAnki } from '@/lib/api/study'
import type { FlashcardSet } from '@/types'

export default function AnkiExport() {
  const [sets, setSets] = useState<FlashcardSet[]>([])
  const [selectedSetId, setSelectedSetId] = useState('')

  useEffect(() => {
    listFlashcardSets().then(data => setSets(data.sets || [])).catch(() => {})
  }, [])

  const handleExportSet = async () => {
    if (!selectedSetId) return
    try {
      const text = await exportFlashcardsToAnki(selectedSetId)
      downloadText(text as unknown as string, 'flashcards_anki.txt')
    } catch (e) { console.error('Export failed', e) }
  }

  const handleExportAll = async () => {
    try {
      const text = await exportAllFlashcardsToAnki()
      downloadText(text as unknown as string, 'all_flashcards_anki.txt')
    } catch (e) { console.error('Export all failed', e) }
  }

  const downloadText = (text: string, filename: string) => {
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = filename; a.click()
    URL.revokeObjectURL(url)
  }

  const totalCards = sets.reduce((sum, s) => sum + (s.card_count || 0), 0)

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
      <h3 className="font-semibold text-white text-sm">Export to Anki</h3>
      <div className="flex gap-2">
        <select value={selectedSetId} onChange={(e) => setSelectedSetId(e.target.value)} className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm">
          <option value="">Select a set...</option>
          {sets.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
        <button onClick={handleExportSet} disabled={!selectedSetId} className="px-3 py-2 bg-gray-700 text-white rounded-lg text-sm hover:bg-gray-600 disabled:opacity-50">Export</button>
      </div>
      {totalCards > 0 && (
        <button onClick={handleExportAll} className="text-sm text-[#6CA0DC] hover:underline">
          Export All ({totalCards} cards)
        </button>
      )}
    </div>
  )
}
