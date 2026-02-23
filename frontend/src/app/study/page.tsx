'use client'

import { useState } from 'react'
import FlashcardGenerator from '@/components/study/FlashcardGenerator'
import FlashcardReview from '@/components/study/FlashcardReview'
import QuizGenerator from '@/components/study/QuizGenerator'
import QuizTaker from '@/components/study/QuizTaker'
import AnkiExport from '@/components/study/AnkiExport'
import type { FlashcardSet, Quiz } from '@/types'

export default function StudyPage() {
  const [tab, setTab] = useState<'flashcards' | 'quizzes'>('flashcards')
  const [activeFlashcardSet, setActiveFlashcardSet] = useState<FlashcardSet | null>(null)
  const [activeQuiz, setActiveQuiz] = useState<Quiz | null>(null)

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Study Mode</h1>

      {/* Tab toggle */}
      <div className="flex gap-2">
        {(['flashcards', 'quizzes'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm transition-colors ${
              tab === t ? 'bg-[#6CA0DC]/20 text-[#6CA0DC]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {t === 'flashcards' ? 'Flashcards' : 'Quizzes'}
          </button>
        ))}
      </div>

      {tab === 'flashcards' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <FlashcardGenerator onSetLoaded={setActiveFlashcardSet} />
            <AnkiExport />
          </div>
          <div>
            {activeFlashcardSet ? (
              <FlashcardReview flashcardSet={activeFlashcardSet} />
            ) : (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 text-center text-gray-500">
                Generate or load a flashcard set to start reviewing
              </div>
            )}
          </div>
        </div>
      )}

      {tab === 'quizzes' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <QuizGenerator onQuizLoaded={setActiveQuiz} />
          <div>
            {activeQuiz ? (
              <QuizTaker quiz={activeQuiz} />
            ) : (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 text-center text-gray-500">
                Generate or load a quiz to start
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
