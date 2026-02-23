'use client'

import { useState } from 'react'
import { searchKnowledgeBase } from '@/lib/api/kb'
import type { SearchResult } from '@/types'
import LoadingSpinner from '@/components/ui/LoadingSpinner'

export default function SemanticSearch() {
  const [query, setQuery] = useState('')
  const [numResults, setNumResults] = useState(5)
  const [threshold, setThreshold] = useState(0.7)
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  const handleSearch = async () => {
    if (!query.trim()) return
    setLoading(true)
    try {
      const data = await searchKnowledgeBase(query.trim(), numResults, threshold)
      setResults(data.results || [])
      setSearched(true)
    } catch (e) {
      console.error('Search failed', e)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        placeholder="Enter search query..."
        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]"
      />

      <div className="flex flex-wrap gap-4 items-center text-sm">
        <div className="flex items-center gap-2">
          <span className="text-gray-500">Results:</span>
          <input type="range" min={1} max={10} value={numResults} onChange={(e) => setNumResults(Number(e.target.value))} className="w-20 accent-[#6CA0DC]" />
          <span className="text-gray-400 w-4">{numResults}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-500">Threshold:</span>
          <input type="range" min={0} max={100} step={5} value={threshold * 100} onChange={(e) => setThreshold(Number(e.target.value) / 100)} className="w-20 accent-[#6CA0DC]" />
          <span className="text-gray-400 w-8">{threshold.toFixed(2)}</span>
        </div>
        <button onClick={handleSearch} disabled={loading || !query.trim()} className="px-4 py-1.5 bg-[#6CA0DC] text-white rounded-lg text-sm hover:bg-[#5a8ec4] disabled:opacity-50">
          Search
        </button>
      </div>

      {loading && <LoadingSpinner message="Searching..." />}

      {searched && !loading && (
        <div className="space-y-3">
          <p className="text-sm text-gray-500">{results.length} result{results.length !== 1 ? 's' : ''}</p>
          {results.map((result, i) => (
            <details key={i} className="bg-gray-800 rounded-lg overflow-hidden">
              <summary className="px-4 py-3 cursor-pointer text-sm text-white hover:bg-gray-700">
                Score: {result.similarity.toFixed(3)} | {result.metadata?.title || 'Untitled'}
              </summary>
              <div className="px-4 pb-3 text-sm text-gray-300">
                <p>{result.content}</p>
                <p className="text-xs text-gray-500 mt-2">
                  {Object.entries(result.metadata || {}).map(([k, v]) => `${k}: ${v}`).join(' | ')}
                </p>
              </div>
            </details>
          ))}
        </div>
      )}
    </div>
  )
}
