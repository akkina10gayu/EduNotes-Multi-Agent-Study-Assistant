'use client'

import { useState, useEffect } from 'react'
import { getTopics } from '@/lib/api/notes'
import { truncate } from '@/lib/utils/formatters'

const DEFAULT_TOPICS = [
  'Machine Learning', 'Deep Learning', 'Natural Language Processing',
  'Computer Vision', 'Data Structures', 'Algorithms',
  'Web Development', 'Cloud Computing', 'Cybersecurity', 'Blockchain',
]

interface TopicChipsProps {
  onSelect: (topic: string) => void
}

export default function TopicChips({ onSelect }: TopicChipsProps) {
  const [topics, setTopics] = useState<string[]>(DEFAULT_TOPICS)
  const [showAll, setShowAll] = useState(false)

  useEffect(() => {
    const fetchTopics = async () => {
      try {
        const data = await getTopics()
        if (data.topics?.length) {
          const merged = [...new Set([...data.topics, ...DEFAULT_TOPICS])]
          setTopics(merged)
        }
      } catch { /* use defaults */ }
    }
    fetchTopics()
    const interval = setInterval(fetchTopics, 300000) // 5-min cache
    return () => clearInterval(interval)
  }, [])

  const displayed = showAll ? topics.slice(0, 10) : topics.slice(0, 5)

  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {displayed.map(topic => (
          <button
            key={topic}
            onClick={() => onSelect(topic)}
            title={topic}
            className="px-3 py-1 text-sm bg-gray-800 text-gray-300 rounded-full hover:bg-[#6CA0DC]/20 hover:text-[#6CA0DC] transition-colors"
          >
            {truncate(topic, 14)}
          </button>
        ))}
        {topics.length > 5 && (
          <button
            onClick={() => setShowAll(!showAll)}
            className="px-3 py-1 text-sm text-[#6CA0DC] hover:underline"
          >
            {showAll ? 'Show less' : 'Show more'}
          </button>
        )}
      </div>
    </div>
  )
}
