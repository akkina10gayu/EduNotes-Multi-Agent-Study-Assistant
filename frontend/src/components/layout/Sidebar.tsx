'use client'

import { useState, useEffect, useCallback } from 'react'
import { getHealth, getSystemStats, getDashboardStats } from '@/lib/api/progress'
import { truncate, getQueryTypeIcon } from '@/lib/utils/formatters'
import { usePathname } from 'next/navigation'

interface NoteHistoryItem {
  query: string
  query_type: string
  notes: string
  timestamp: number
}

export default function Sidebar() {
  const pathname = usePathname()
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null)
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large'>('medium')
  const [systemStats, setSystemStats] = useState<Record<string, unknown> | null>(null)
  const [dashStats, setDashStats] = useState<{ kb_documents: number; topics_count: number } | null>(null)
  const [noteHistory, setNoteHistory] = useState<NoteHistoryItem[]>([])
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({})
  const [selectedHistoryNote, setSelectedHistoryNote] = useState<NoteHistoryItem | null>(null)

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  // Font size
  useEffect(() => {
    const saved = localStorage.getItem('edunotes-font-size') as 'small' | 'medium' | 'large' | null
    if (saved) setFontSize(saved)
  }, [])

  useEffect(() => {
    document.body.className = document.body.className.replace(/font-(small|medium|large)/, '')
    document.body.classList.add(`font-${fontSize}`)
    localStorage.setItem('edunotes-font-size', fontSize)
  }, [fontSize])

  // Health check polling
  useEffect(() => {
    const check = async () => {
      try {
        const data = await getHealth()
        setIsHealthy(data.status === 'healthy')
      } catch {
        setIsHealthy(false)
      }
    }
    check()
    const interval = setInterval(check, 30000)
    return () => clearInterval(interval)
  }, [])

  // Load system stats when expanded
  const loadSystemStats = useCallback(async () => {
    try {
      const [stats, dash] = await Promise.all([getSystemStats(), getDashboardStats()])
      setSystemStats(stats)
      setDashStats(dash)
    } catch (e) {
      console.error('Failed to load system stats', e)
    }
  }, [])

  // Load note history from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('edunotes-note-history')
    if (stored) {
      try { setNoteHistory(JSON.parse(stored)) } catch { /* ignore */ }
    }
  }, [])

  // Listen for new notes via custom event
  useEffect(() => {
    const handler = (e: CustomEvent<NoteHistoryItem>) => {
      setNoteHistory(prev => {
        const updated = [e.detail, ...prev].slice(0, 6)
        localStorage.setItem('edunotes-note-history', JSON.stringify(updated))
        return updated
      })
    }
    window.addEventListener('edunotes:note-generated', handler as EventListener)
    return () => window.removeEventListener('edunotes:note-generated', handler as EventListener)
  }, [])

  // Dispatch selected history note
  useEffect(() => {
    if (selectedHistoryNote) {
      window.dispatchEvent(new CustomEvent('edunotes:view-history-note', { detail: selectedHistoryNote }))
      setSelectedHistoryNote(null)
    }
  }, [selectedHistoryNote])

  if (pathname.startsWith('/auth')) return null

  return (
    <aside className="w-64 border-r border-gray-800 bg-gray-900/30 min-h-[calc(100vh-3.5rem)] p-4 space-y-4 hidden lg:block">
      {/* API Health */}
      <div className="flex items-center gap-2 text-sm">
        <div className={`w-2 h-2 rounded-full ${isHealthy === null ? 'bg-gray-500' : isHealthy ? 'bg-green-500' : 'bg-red-500'}`} />
        <span className="text-gray-400">
          API: {isHealthy === null ? 'Checking...' : isHealthy ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      {/* Font Size */}
      <div>
        <p className="text-xs text-gray-500 mb-1">Font Size</p>
        <div className="flex gap-1">
          {(['small', 'medium', 'large'] as const).map(size => (
            <button
              key={size}
              onClick={() => setFontSize(size)}
              className={`px-2 py-1 text-xs rounded ${fontSize === size ? 'bg-[#6CA0DC]/20 text-[#6CA0DC]' : 'text-gray-400 hover:bg-gray-800'}`}
            >
              {size.charAt(0).toUpperCase() + size.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* System Stats */}
      <div>
        <button onClick={() => { toggleSection('stats'); if (!expandedSections.stats) loadSystemStats() }} className="w-full flex items-center justify-between text-sm text-gray-400 hover:text-white">
          <span>System Stats</span>
          <span className="text-xs">{expandedSections.stats ? '▼' : '▶'}</span>
        </button>
        {expandedSections.stats && (
          <div className="mt-2 space-y-1 text-xs text-gray-500">
            {systemStats ? (
              <>
                <p>LLM: {(systemStats as { llm?: { provider?: string; model?: string } }).llm?.provider || 'unknown'} / {(systemStats as { llm?: { provider?: string; model?: string } }).llm?.model || 'unknown'}</p>
                <p>KB Documents: {dashStats?.kb_documents ?? '?'}</p>
                <p>Topics: {dashStats?.topics_count ?? '?'}</p>
                <p>Agents: All active</p>
              </>
            ) : <p>Loading...</p>}
          </div>
        )}
      </div>

      {/* Note History */}
      <div>
        <button onClick={() => toggleSection('history')} className="w-full flex items-center justify-between text-sm text-gray-400 hover:text-white">
          <span>Note History</span>
          <span className="text-xs">{expandedSections.history ? '▼' : '▶'}</span>
        </button>
        {expandedSections.history && (
          <div className="mt-2 space-y-1">
            {noteHistory.length === 0 ? (
              <p className="text-xs text-gray-500">No notes yet</p>
            ) : (
              noteHistory.map((note, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedHistoryNote(note)}
                  className="w-full text-left px-2 py-1 rounded text-xs text-gray-400 hover:bg-gray-800 hover:text-white flex items-center gap-1"
                >
                  <span>{getQueryTypeIcon(note.query_type)}</span>
                  <span>{truncate(note.query, 35)}</span>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      {/* Help & Setup Guide */}
      <div>
        <button onClick={() => toggleSection('help')} className="w-full flex items-center justify-between text-sm text-gray-400 hover:text-white">
          <span>Help & Setup Guide</span>
          <span className="text-xs">{expandedSections.help ? '▼' : '▶'}</span>
        </button>
        {expandedSections.help && (
          <div className="mt-2 space-y-2 text-xs text-gray-500">
            <div>
              <p className="text-gray-400 font-medium">Getting Started</p>
              <p>Enter a topic, paste a URL, upload a PDF, or paste text directly to generate notes.</p>
            </div>
            <div>
              <p className="text-gray-400 font-medium">API Key Setup</p>
              <p>Get a free API key from console.groq.com and add it to your .env file.</p>
            </div>
            <div>
              <p className="text-gray-400 font-medium">Search Modes</p>
              <p>Auto: KB first, web fallback. KB Only: Local only. Web Search: Internet only. Both: Combined.</p>
            </div>
            <div>
              <p className="text-gray-400 font-medium">Research Mode</p>
              <p>Enable for academic papers -- analyzes figures, tables, and finds related references.</p>
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}
