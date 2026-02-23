'use client'

import { useState } from 'react'
import { startQuizAttempt, submitQuizAnswer, completeQuizAttempt } from '@/lib/api/study'
import type { Quiz, QuizAttemptResult } from '@/types'
import LoadingSpinner from '@/components/ui/LoadingSpinner'

interface Props { quiz: Quiz }

export default function QuizTaker({ quiz }: Props) {
  const [attemptId, setAttemptId] = useState<string | null>(null)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [results, setResults] = useState<QuizAttemptResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startQuiz = async () => {
    setLoading(true); setError(null); setResults(null); setAnswers({})
    try {
      const data = await startQuizAttempt(quiz.id)
      if (data.success) setAttemptId(data.attempt_id)
      else setError('Failed to start quiz')
    } catch (e) { setError(e instanceof Error ? e.message : 'Error') }
    finally { setLoading(false) }
  }

  const handleSubmit = async () => {
    if (!attemptId) return
    const unanswered = quiz.questions.filter(q => !answers[q.id])
    if (unanswered.length > 0) { setError(`Please answer all questions (${unanswered.length} remaining)`); return }

    setLoading(true); setError(null)
    try {
      for (const q of quiz.questions) {
        await submitQuizAnswer(quiz.id, attemptId, q.id, answers[q.id])
      }
      const result = await completeQuizAttempt(quiz.id, attemptId)
      if (result.success) setResults(result)
      else setError('Failed to complete quiz')
    } catch (e) { setError(e instanceof Error ? e.message : 'Error') }
    finally { setLoading(false) }
  }

  const scoreColor = results ? (results.score >= 80 ? 'text-green-400' : results.score >= 60 ? 'text-blue-400' : 'text-yellow-400') : ''

  if (!attemptId && !results) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 text-center space-y-3">
        <h3 className="font-semibold text-white">{quiz.title}</h3>
        <p className="text-sm text-gray-400">{quiz.questions.length} questions</p>
        <button onClick={startQuiz} disabled={loading} className="px-6 py-2 bg-[#6CA0DC] text-white rounded-lg hover:bg-[#5a8ec4] disabled:opacity-50">
          {loading ? 'Starting...' : 'Start Quiz'}
        </button>
      </div>
    )
  }

  if (results) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
        <h3 className="text-lg font-semibold text-white text-center">Quiz Results</h3>
        <p className={`text-3xl font-bold text-center ${scoreColor}`}>{Math.round(results.score)}%</p>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-green-900/20 border border-green-800 rounded-lg p-3 text-center">
            <p className="text-xl font-bold text-green-400">{results.correct_count}</p>
            <p className="text-xs text-gray-400">Correct</p>
          </div>
          <div className="bg-red-900/20 border border-red-800 rounded-lg p-3 text-center">
            <p className="text-xl font-bold text-red-400">{results.total_questions - results.correct_count}</p>
            <p className="text-xs text-gray-400">Incorrect</p>
          </div>
        </div>

        {results.detailed_results?.map((r, i) => (
          <div key={i} className={`p-3 rounded-lg border text-sm ${r.is_correct ? 'bg-green-900/10 border-green-800' : 'bg-red-900/10 border-red-800'}`}>
            <p className="text-white font-medium">{r.question}</p>
            <p className="text-xs mt-1">Your answer: <span className={r.is_correct ? 'text-green-400' : 'text-red-400'}>{r.user_answer}</span></p>
            {!r.is_correct && <p className="text-xs">Correct: <span className="text-green-400">{r.correct_answer}</span></p>}
            {r.explanation && <p className="text-xs text-gray-500 mt-1">{r.explanation}</p>}
          </div>
        ))}

        <div className="flex gap-2">
          <button onClick={() => { setAttemptId(null); setResults(null); setAnswers({}) }} className="flex-1 py-2 bg-[#6CA0DC] text-white rounded-lg text-sm hover:bg-[#5a8ec4]">Try Again</button>
          <button onClick={() => { setAttemptId(null); setResults(null); setAnswers({}) }} className="flex-1 py-2 bg-gray-800 text-gray-300 rounded-lg text-sm hover:bg-gray-700">New Quiz</button>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <h3 className="font-semibold text-white">{quiz.title}</h3>
      <p className="text-sm text-gray-400">{quiz.questions.length} questions</p>

      {quiz.questions.map((q, qi) => (
        <div key={q.id} className="space-y-2">
          <p className="text-sm text-white font-medium">{qi + 1}. {q.question}</p>
          <div className="space-y-1 pl-4">
            {q.options.map((opt, oi) => (
              <label key={oi} className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer hover:text-white">
                <input type="radio" name={q.id} value={opt} checked={answers[q.id] === opt} onChange={() => setAnswers(prev => ({ ...prev, [q.id]: opt }))} className="accent-[#6CA0DC]" />
                {opt}
              </label>
            ))}
          </div>
        </div>
      ))}

      {error && <p className="text-sm text-red-400">{error}</p>}
      {loading && <LoadingSpinner size="sm" message="Submitting..." />}

      <button onClick={handleSubmit} disabled={loading} className="w-full py-2 bg-[#6CA0DC] text-white rounded-lg hover:bg-[#5a8ec4] disabled:opacity-50">
        {loading ? 'Submitting...' : 'Submit Quiz'}
      </button>
    </div>
  )
}
