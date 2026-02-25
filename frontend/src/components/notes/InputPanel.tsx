'use client'

import { useState, useRef, useEffect } from 'react'
import { generateNotes, processPdf } from '@/lib/api/notes'
import type { GenerateNotesResponse } from '@/types'
import LoadingSpinner from '@/components/ui/LoadingSpinner'

interface InputPanelProps {
  inputValue: string
  setInputValue: (v: string) => void
  onNoteGenerated: (note: GenerateNotesResponse) => void
}

type InputType = 'topic' | 'url' | 'text'

function detectInputType(value: string): InputType {
  if (value.startsWith('http://') || value.startsWith('https://') || value.startsWith('www.')) return 'url'
  if (value.length > 500) return 'text'
  return 'topic'
}

export default function InputPanel({ inputValue, setInputValue, onNoteGenerated }: InputPanelProps) {
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [researchMode, setResearchMode] = useState(false)
  const [searchMode, setSearchMode] = useState(() => {
    if (typeof window !== 'undefined') return localStorage.getItem('edunotes-search-mode') || 'auto'
    return 'auto'
  })
  const [outputFormat, setOutputFormat] = useState(() => {
    if (typeof window !== 'undefined') return localStorage.getItem('edunotes-output-format') || 'paragraph_summary'
    return 'paragraph_summary'
  })
  const [summaryLength, setSummaryLength] = useState('auto')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => { localStorage.setItem('edunotes-search-mode', searchMode) }, [searchMode])
  useEffect(() => { localStorage.setItem('edunotes-output-format', outputFormat) }, [outputFormat])

  const inputType = detectInputType(inputValue)

  const handleGenerate = async () => {
    if (!inputValue.trim() && !pdfFile) return
    if (inputValue.trim() && pdfFile) {
      setError('Please use either text input OR PDF upload, not both.')
      return
    }

    setError(null)
    setLoading(true)

    try {
      if (pdfFile) {
        setProgress('Uploading PDF...')
        const formData = new FormData()
        formData.append('file', pdfFile)
        formData.append('summarization_mode', outputFormat)
        formData.append('output_length', summaryLength)
        formData.append('research_mode', String(researchMode))

        setProgress('Processing PDF...')
        const result = await processPdf(formData)
        onNoteGenerated(result)
      } else {
        setProgress('Analyzing input...')
        const result = await generateNotes({
          query: inputValue.trim(),
          summarization_mode: outputFormat,
          summary_length: summaryLength,
          search_mode: inputType === 'topic' ? searchMode : undefined,
          research_mode: researchMode,
        })
        setProgress('Generating notes...')
        onNoteGenerated(result)
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to generate notes'
      if (message.includes('429') || message.toLowerCase().includes('rate limit')) {
        setError('Rate limit reached. Please wait a moment and try again, or the system will automatically use a lighter model.')
      } else {
        setError(message)
      }
    } finally {
      setLoading(false)
      setProgress('')
    }
  }

  return (
    <div className="space-y-4 bg-gray-900 border border-gray-800 rounded-xl p-5">
      {/* Input area */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Enter a topic, paste a URL, or paste text..."
            rows={4}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white placeholder-gray-500 resize-none focus:outline-none focus:border-[#6CA0DC]"
          />
          {inputValue && (
            <p className="text-xs text-gray-500 mt-1">
              {inputType === 'url' ? 'URL detected' : inputType === 'text' ? 'Text detected' : 'Topic detected'}
            </p>
          )}
        </div>
        <div
          className="border-2 border-dashed border-gray-700 rounded-lg p-4 flex flex-col items-center justify-center cursor-pointer hover:border-[#6CA0DC]/50 transition-colors"
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); if (e.dataTransfer.files[0]?.type === 'application/pdf') setPdfFile(e.dataTransfer.files[0]) }}
        >
          <input ref={fileRef} type="file" accept=".pdf" className="hidden" onChange={(e) => setPdfFile(e.target.files?.[0] || null)} />
          {pdfFile ? (
            <div className="text-center">
              <p className="text-sm text-white">{pdfFile.name}</p>
              <p className="text-xs text-gray-500">{(pdfFile.size / 1024 / 1024).toFixed(1)} MB</p>
              <button onClick={(e) => { e.stopPropagation(); setPdfFile(null) }} className="text-xs text-red-400 mt-1 hover:underline">Remove</button>
            </div>
          ) : (
            <>
              <p className="text-sm text-gray-400">Drop PDF here or click to upload</p>
              <p className="text-xs text-gray-600 mt-1">Max 10MB</p>
            </>
          )}
        </div>
      </div>

      {/* Options row */}
      <div className="flex flex-wrap gap-4 items-center text-sm">
        {/* Research Mode */}
        <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
          <input type="checkbox" checked={researchMode} onChange={(e) => setResearchMode(e.target.checked)} className="accent-[#6CA0DC]" />
          Research Mode
        </label>

        {/* Search Mode - only for topics */}
        {inputType === 'topic' && !pdfFile && (
          <div className="flex items-center gap-2">
            <span className="text-gray-500">Search:</span>
            {['auto', 'kb_only', 'web_search', 'both'].map(mode => (
              <button
                key={mode}
                onClick={() => setSearchMode(mode)}
                className={`px-2 py-0.5 rounded text-xs ${searchMode === mode ? 'bg-[#6CA0DC]/20 text-[#6CA0DC]' : 'text-gray-400 hover:bg-gray-800'}`}
              >
                {mode === 'auto' ? 'Auto' : mode === 'kb_only' ? 'KB Only' : mode === 'web_search' ? 'Web Search' : 'Both'}
              </button>
            ))}
          </div>
        )}

        {/* Output Format */}
        <div className="flex items-center gap-2">
          <span className="text-gray-500">Format:</span>
          {[
            { value: 'paragraph_summary', label: 'Paragraph' },
            { value: 'important_points', label: 'Points' },
            { value: 'key_highlights', label: 'Highlights' },
          ].map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setOutputFormat(value)}
              className={`px-2 py-0.5 rounded text-xs ${outputFormat === value ? 'bg-[#6CA0DC]/20 text-[#6CA0DC]' : 'text-gray-400 hover:bg-gray-800'}`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Summary Length - only for paragraph */}
        {outputFormat === 'paragraph_summary' && (
          <div className="flex items-center gap-2">
            <span className="text-gray-500">Length:</span>
            {['auto', 'brief', 'medium', 'detailed'].map(len => (
              <button
                key={len}
                onClick={() => setSummaryLength(len)}
                className={`px-2 py-0.5 rounded text-xs ${summaryLength === len ? 'bg-[#6CA0DC]/20 text-[#6CA0DC]' : 'text-gray-400 hover:bg-gray-800'}`}
              >
                {len.charAt(0).toUpperCase() + len.slice(1)}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Generate button */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleGenerate}
          disabled={loading || (!inputValue.trim() && !pdfFile)}
          className="px-6 py-2 rounded-lg text-white font-medium transition-colors disabled:opacity-50"
          style={{ backgroundColor: '#6CA0DC' }}
        >
          {loading ? 'Generating...' : 'Generate Notes'}
        </button>
        {loading && <LoadingSpinner size="sm" message={progress} />}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-3 text-sm text-red-400">
          {error}
        </div>
      )}
    </div>
  )
}
