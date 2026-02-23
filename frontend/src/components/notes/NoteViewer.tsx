'use client'

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { GenerateNotesResponse } from '@/types'
import NoteEditor from './NoteEditor'
import ResearchResults from './ResearchResults'
import SaveToKBModal from './SaveToKBModal'

interface NoteViewerProps {
  note: GenerateNotesResponse
  onClose: () => void
}

export default function NoteViewer({ note, onClose }: NoteViewerProps) {
  const [currentNotes, setCurrentNotes] = useState(note.notes)
  const [previousNotes, setPreviousNotes] = useState<string | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(currentNotes)
      setSuccessMsg('Copied to clipboard!')
      setTimeout(() => setSuccessMsg(null), 3000)
    } catch {
      setSuccessMsg('Failed to copy')
    }
  }

  const handleDownload = () => {
    const blob = new Blob([currentNotes], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${note.query?.slice(0, 30) || 'notes'}.md`
    a.click()
    URL.revokeObjectURL(url)
    setSuccessMsg('Downloaded!')
    setTimeout(() => setSuccessMsg(null), 3000)
  }

  const handleSaveEdit = (newNotes: string) => {
    setPreviousNotes(currentNotes)
    setCurrentNotes(newNotes)
    setIsEditing(false)
    setSuccessMsg('Notes updated!')
    setTimeout(() => setSuccessMsg(null), 3000)
  }

  const handleUndo = () => {
    if (previousNotes) {
      setCurrentNotes(previousNotes)
      setPreviousNotes(null)
      setSuccessMsg('Undo successful!')
      setTimeout(() => setSuccessMsg(null), 3000)
    }
  }

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Generated Notes</h2>
        <button onClick={onClose} className="text-gray-400 hover:text-white text-sm">Close</button>
      </div>

      {/* Metadata bar */}
      <div className="flex flex-wrap gap-4 text-xs text-gray-400">
        <span>Type: <span className="text-gray-300">{note.query_type}</span></span>
        <span>Sources: <span className="text-gray-300">{note.sources_used}</span></span>
        <span>Source: <span className="text-gray-300">{note.from_kb ? 'Knowledge Base' : 'Web'}</span></span>
        {note.message && <span className="text-gray-500">{note.message}</span>}
      </div>

      {/* Success message */}
      {successMsg && (
        <div className="bg-green-900/20 border border-green-800 rounded-lg p-2 text-sm text-green-400 flex items-center justify-between">
          {successMsg}
          <button onClick={() => setSuccessMsg(null)} className="text-green-600 hover:text-green-400">dismiss</button>
        </div>
      )}

      {/* Note content or editor */}
      {isEditing ? (
        <NoteEditor
          initialNotes={currentNotes}
          onSave={handleSaveEdit}
          onCancel={() => setIsEditing(false)}
        />
      ) : (
        <div className="notes-container">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {currentNotes}
          </ReactMarkdown>
        </div>
      )}

      {/* Action buttons */}
      {!isEditing && (
        <div className="flex flex-wrap gap-2">
          <button onClick={handleDownload} className="px-3 py-1.5 text-sm bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700">
            Download .md
          </button>
          <button onClick={handleCopy} className="px-3 py-1.5 text-sm bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700">
            Copy
          </button>
          <button onClick={() => setShowSaveModal(true)} className="px-3 py-1.5 text-sm bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700">
            Save to KB
          </button>
          <button onClick={() => setIsEditing(true)} className="px-3 py-1.5 text-sm bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700">
            Edit
          </button>
          {previousNotes && (
            <button onClick={handleUndo} className="px-3 py-1.5 text-sm bg-yellow-900/30 text-yellow-400 rounded-lg hover:bg-yellow-900/50">
              Undo Last Edit
            </button>
          )}
        </div>
      )}

      {/* Research results */}
      {note.vision_data && <ResearchResults visionData={note.vision_data} />}
      {note.related_papers && note.related_papers.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-[#6CA0DC] mb-2">Related Papers</h3>
          {note.related_papers.map((paper, i) => (
            <div key={i} className="mb-3 text-sm">
              <a href={paper.url} target="_blank" rel="noopener noreferrer" className="text-[#6CA0DC] hover:underline font-medium">{paper.title}</a>
              <p className="text-gray-500 text-xs">{paper.authors?.join(', ')}</p>
              <p className="text-gray-400 text-xs mt-1">{paper.summary?.slice(0, 200)}...</p>
            </div>
          ))}
        </div>
      )}

      {/* Save to KB Modal */}
      {showSaveModal && (
        <SaveToKBModal
          note={note}
          currentNotes={currentNotes}
          onClose={() => setShowSaveModal(false)}
          onSuccess={() => { setShowSaveModal(false); setSuccessMsg('Saved to Knowledge Base!'); setTimeout(() => setSuccessMsg(null), 3000) }}
        />
      )}
    </div>
  )
}
