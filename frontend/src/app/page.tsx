'use client'

import { useState, useEffect } from 'react'
import TopicChips from '@/components/notes/TopicChips'
import InputPanel from '@/components/notes/InputPanel'
import NoteViewer from '@/components/notes/NoteViewer'
import { getDashboardStats } from '@/lib/api/progress'
import type { GenerateNotesResponse, DashboardStats } from '@/types'

export default function HomePage() {
  const [currentNote, setCurrentNote] = useState<GenerateNotesResponse | null>(null)
  const [inputValue, setInputValue] = useState('')
  const [dashStats, setDashStats] = useState<DashboardStats | null>(null)
  const [sessionNoteCount, setSessionNoteCount] = useState(0)
  const [showWelcome, setShowWelcome] = useState(false)

  // Check for first-time visit
  useEffect(() => {
    if (!localStorage.getItem('edunotes-welcome-dismissed')) {
      setShowWelcome(true)
    }
  }, [])

  // Load dashboard stats
  useEffect(() => {
    getDashboardStats().then(setDashStats).catch(() => {})
  }, [currentNote])

  const handleNoteGenerated = (note: GenerateNotesResponse) => {
    setCurrentNote(note)
    setSessionNoteCount(prev => prev + 1)
    // Dispatch event for sidebar history
    window.dispatchEvent(new CustomEvent('edunotes:note-generated', {
      detail: { query: note.query, query_type: note.query_type, notes: note.notes, timestamp: Date.now() }
    }))
  }

  // Listen for history note view events from sidebar
  useEffect(() => {
    const handler = (e: CustomEvent) => {
      setCurrentNote({ success: true, query: e.detail.query, query_type: e.detail.query_type, notes: e.detail.notes, sources_used: 0, from_kb: false })
    }
    window.addEventListener('edunotes:view-history-note', handler as EventListener)
    return () => window.removeEventListener('edunotes:view-history-note', handler as EventListener)
  }, [])

  return (
    <div className="space-y-6">
      {/* Welcome Banner */}
      {showWelcome && (
        <div className="bg-[#6CA0DC]/10 border border-[#6CA0DC]/30 rounded-xl p-4 flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-[#6CA0DC]">Welcome to EduNotes!</h2>
            <p className="text-sm text-gray-300 mt-1">
              Enter a topic, paste a URL, upload a PDF, or paste text to generate study notes.
              Use Research Mode for academic papers with figure analysis.
            </p>
          </div>
          <button
            onClick={() => { setShowWelcome(false); localStorage.setItem('edunotes-welcome-dismissed', 'true') }}
            className="text-sm text-[#6CA0DC] hover:underline whitespace-nowrap ml-4"
          >
            Got it!
          </button>
        </div>
      )}

      {/* Quick Stats */}
      {dashStats && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Notes', value: sessionNoteCount, sub: 'this session' },
            { label: 'Flashcard Sets', value: dashStats.flashcard_sets },
            { label: 'Quizzes', value: dashStats.total_quizzes },
            { label: 'Streak', value: `${dashStats.current_streak} days` },
          ].map(({ label, value, sub }) => (
            <div key={label} className="bg-gray-900 border border-gray-800 rounded-lg p-3 text-center">
              <p className="text-xl font-bold text-white">{value}</p>
              <p className="text-xs text-gray-400">{label}</p>
              {sub && <p className="text-xs text-gray-500">{sub}</p>}
            </div>
          ))}
        </div>
      )}

      {/* Topic Chips */}
      <TopicChips onSelect={(topic) => setInputValue(topic)} />

      {/* Input Panel */}
      <InputPanel
        inputValue={inputValue}
        setInputValue={setInputValue}
        onNoteGenerated={handleNoteGenerated}
      />

      {/* Note Viewer */}
      {currentNote && currentNote.success && (
        <NoteViewer
          note={currentNote}
          onClose={() => setCurrentNote(null)}
        />
      )}

      {/* Error display */}
      {currentNote && !currentNote.success && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
          <p className="text-red-400">{currentNote.error || currentNote.message || 'Failed to generate notes'}</p>
        </div>
      )}
    </div>
  )
}
