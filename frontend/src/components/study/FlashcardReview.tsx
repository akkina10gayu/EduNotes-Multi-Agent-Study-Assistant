'use client'

import { useState, useCallback } from 'react'
import { reviewFlashcard } from '@/lib/api/study'
import type { FlashcardSet } from '@/types'

interface Props { flashcardSet: FlashcardSet }

export default function FlashcardReview({ flashcardSet }: Props) {
  const cards = flashcardSet.cards || []
  const [currentIndex, setCurrentIndex] = useState(0)
  const [showAnswer, setShowAnswer] = useState(false)
  const [reviewed, setReviewed] = useState(0)
  const [correct, setCorrect] = useState(0)

  const card = cards[currentIndex]
  if (!card) return <p className="text-gray-500">No cards available</p>

  const handleReview = async (isCorrect: boolean) => {
    try {
      await reviewFlashcard(flashcardSet.id, card.id, isCorrect)
    } catch { /* ignore API errors for review tracking */ }
    setReviewed(r => r + 1)
    if (isCorrect) setCorrect(c => c + 1)
    setShowAnswer(false)
    if (currentIndex < cards.length - 1) setCurrentIndex(i => i + 1)
    else setCurrentIndex(0)
  }

  const handleShuffle = useCallback(() => {
    const idx = Math.floor(Math.random() * cards.length)
    setCurrentIndex(idx)
    setShowAnswer(false)
  }, [cards.length])

  return (
    <div className="space-y-4">
      {/* Progress */}
      <div className="flex items-center justify-between text-sm text-gray-400">
        <span>Card {currentIndex + 1} of {cards.length}</span>
        {reviewed > 0 && <span>{correct}/{reviewed} correct ({Math.round(correct/reviewed*100)}%)</span>}
      </div>
      <div className="w-full bg-gray-800 rounded-full h-1.5">
        <div className="bg-[#6CA0DC] h-1.5 rounded-full transition-all" style={{ width: `${((currentIndex + 1) / cards.length) * 100}%` }} />
      </div>

      {/* Card */}
      <div className={showAnswer ? 'flashcard-answer' : 'flashcard-question'}>
        <div className="text-lg font-medium text-white">
          {showAnswer ? card.back : card.front}
        </div>
      </div>

      {/* Actions */}
      {!showAnswer ? (
        <button onClick={() => setShowAnswer(true)} className="w-full py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700">Show Answer</button>
      ) : (
        <div className="flex gap-3">
          <button onClick={() => handleReview(true)} className="flex-1 py-2 bg-green-700 text-white rounded-lg hover:bg-green-600">Got it!</button>
          <button onClick={() => handleReview(false)} className="flex-1 py-2 bg-red-700 text-white rounded-lg hover:bg-red-600">Review Again</button>
        </div>
      )}

      {/* Navigation */}
      <div className="flex gap-2 justify-center">
        <button onClick={() => { setCurrentIndex(Math.max(0, currentIndex - 1)); setShowAnswer(false) }} className="px-3 py-1 text-sm text-gray-400 hover:text-white">Previous</button>
        <button onClick={handleShuffle} className="px-3 py-1 text-sm text-gray-400 hover:text-white">Shuffle</button>
        <button onClick={() => { setCurrentIndex(Math.min(cards.length - 1, currentIndex + 1)); setShowAnswer(false) }} className="px-3 py-1 text-sm text-gray-400 hover:text-white">Next</button>
      </div>
    </div>
  )
}
