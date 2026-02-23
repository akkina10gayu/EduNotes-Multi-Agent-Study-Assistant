'use client'

import { useState, useEffect } from 'react'
import { generateQuiz, listQuizzes, getQuiz, deleteQuiz } from '@/lib/api/study'
import type { Quiz } from '@/types'
import LoadingSpinner from '@/components/ui/LoadingSpinner'

interface Props { onQuizLoaded: (quiz: Quiz) => void }

export default function QuizGenerator({ onQuizLoaded }: Props) {
  const [topic, setTopic] = useState('')
  const [content, setContent] = useState('')
  const [numQuestions, setNumQuestions] = useState(5)
  const [generating, setGenerating] = useState(false)
  const [quizzes, setQuizzes] = useState<Quiz[]>([])
  const [selectedQuizId, setSelectedQuizId] = useState('')
  const [error, setError] = useState<string | null>(null)

  const loadQuizzes = async () => {
    try { const data = await listQuizzes(); setQuizzes(data.quizzes || []) } catch { /* ignore */ }
  }

  useEffect(() => { loadQuizzes() }, [])

  const handleGenerate = async () => {
    if (!topic.trim() || !content.trim()) { setError('Topic and content required'); return }
    setGenerating(true); setError(null)
    try {
      const data = await generateQuiz(content.trim(), topic.trim(), numQuestions)
      if (data.success && data.quiz) { onQuizLoaded(data.quiz); loadQuizzes() }
      else { setError('Failed to generate quiz') }
    } catch (e) { setError(e instanceof Error ? e.message : 'Error') }
    finally { setGenerating(false) }
  }

  const handleLoadQuiz = async () => {
    if (!selectedQuizId) return
    try {
      const data = await getQuiz(selectedQuizId)
      if (data.quiz) onQuizLoaded(data.quiz)
    } catch (e) { setError(e instanceof Error ? e.message : 'Error') }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <h3 className="font-semibold text-white">Generate Quiz</h3>

      <input value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="Topic" className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-[#6CA0DC]" />

      <div className="flex items-center gap-2 text-sm">
        <span className="text-gray-500">Questions:</span>
        <input type="range" min={3} max={15} value={numQuestions} onChange={(e) => setNumQuestions(Number(e.target.value))} className="flex-1 accent-[#6CA0DC]" />
        <span className="text-gray-400 w-6">{numQuestions}</span>
      </div>

      <textarea value={content} onChange={(e) => setContent(e.target.value)} rows={4} placeholder="Paste content to generate quiz from..." className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm resize-none focus:outline-none focus:border-[#6CA0DC]" />

      <button onClick={handleGenerate} disabled={generating} className="w-full py-2 bg-[#6CA0DC] text-white rounded-lg text-sm hover:bg-[#5a8ec4] disabled:opacity-50">
        {generating ? 'Generating...' : 'Generate Quiz'}
      </button>

      {generating && <LoadingSpinner size="sm" message="Generating quiz..." />}
      {error && <p className="text-sm text-red-400">{error}</p>}

      {quizzes.length > 0 && (
        <div className="border-t border-gray-800 pt-4 space-y-2">
          <h4 className="text-sm font-medium text-gray-400">Load Existing Quiz</h4>
          <div className="flex gap-2">
            <select value={selectedQuizId} onChange={(e) => setSelectedQuizId(e.target.value)} className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm">
              <option value="">Select a quiz...</option>
              {quizzes.map(q => <option key={q.id} value={q.id}>{q.title} ({q.question_count} questions)</option>)}
            </select>
            <button onClick={handleLoadQuiz} disabled={!selectedQuizId} className="px-3 py-2 bg-gray-700 text-white rounded-lg text-sm hover:bg-gray-600 disabled:opacity-50">Load</button>
          </div>
          {quizzes.map(q => (
            <div key={q.id} className="flex items-center justify-between text-xs text-gray-500 px-1">
              <span>{q.title}</span>
              <button onClick={async () => { await deleteQuiz(q.id); loadQuizzes() }} className="text-red-500 hover:text-red-400">Delete</button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
