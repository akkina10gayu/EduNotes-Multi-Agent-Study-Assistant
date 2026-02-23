'use client'

import { useState } from 'react'
import { cleanVisionDescription } from '@/lib/utils/markdown'

interface ResearchResultsProps {
  visionData: string
}

interface FigureEntry {
  page: number
  description: string
  image_b64: string
}

export default function ResearchResults({ visionData }: ResearchResultsProps) {
  const [expanded, setExpanded] = useState(false)

  let figures: FigureEntry[] = []
  try {
    figures = JSON.parse(visionData)
  } catch {
    return null
  }

  if (figures.length === 0) return null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-sm text-[#6CA0DC] hover:bg-gray-800/50"
      >
        <span>Figures & Tables ({figures.length} found)</span>
        <span>{expanded ? '▼' : '▶'}</span>
      </button>
      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          {figures.map((fig, i) => (
            <div key={i} className="border-t border-gray-800 pt-3">
              <p className="text-xs text-gray-500 mb-2">Page {fig.page}</p>
              {fig.image_b64 && (
                <img
                  src={`data:image/png;base64,${fig.image_b64}`}
                  alt={`Page ${fig.page}`}
                  className="max-w-full max-h-96 rounded-lg mb-2"
                />
              )}
              <div className="notes-container text-sm">
                {cleanVisionDescription(fig.description)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
