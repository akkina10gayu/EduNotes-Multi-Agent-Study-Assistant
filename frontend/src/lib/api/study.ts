import { apiClient } from './client'
import type { FlashcardSet, Quiz, QuizAttemptResult } from '@/types'

// Flashcards
export async function generateFlashcards(content: string, topic: string, numCards: number = 10, setName?: string) {
  return apiClient<{ success: boolean; flashcard_set: FlashcardSet; message: string }>('/study/flashcards/generate', {
    method: 'POST',
    body: JSON.stringify({ content, topic, num_cards: numCards, set_name: setName }),
  })
}

export async function listFlashcardSets() {
  return apiClient<{ success: boolean; sets: FlashcardSet[]; total_count: number }>('/study/flashcards/sets')
}

export async function getFlashcardSet(setId: string) {
  return apiClient<{ success: boolean; flashcard_set: FlashcardSet }>(`/study/flashcards/sets/${setId}`)
}

export async function reviewFlashcard(setId: string, cardId: string, correct: boolean) {
  return apiClient<{ success: boolean; message: string }>('/study/flashcards/review', {
    method: 'POST',
    body: JSON.stringify({ set_id: setId, card_id: cardId, correct }),
  })
}

export async function deleteFlashcardSet(setId: string) {
  return apiClient<{ success: boolean; message: string }>(`/study/flashcards/sets/${setId}`, { method: 'DELETE' })
}

export async function exportFlashcardsToAnki(setId: string) {
  return apiClient<string>(`/study/flashcards/export/${setId}/anki`)
}

export async function exportAllFlashcardsToAnki() {
  return apiClient<string>('/study/flashcards/export/all/anki')
}

// Quizzes
export async function generateQuiz(content: string, topic: string, numQuestions: number = 5, title?: string) {
  return apiClient<{ success: boolean; quiz: Quiz; message: string }>('/study/quizzes/generate', {
    method: 'POST',
    body: JSON.stringify({ content, topic, num_questions: numQuestions, title }),
  })
}

export async function listQuizzes() {
  return apiClient<{ success: boolean; quizzes: Quiz[]; total_count: number }>('/study/quizzes')
}

export async function getQuiz(quizId: string) {
  return apiClient<{ success: boolean; quiz: Quiz }>(`/study/quizzes/${quizId}`)
}

export async function startQuizAttempt(quizId: string) {
  return apiClient<{ success: boolean; attempt_id: string; quiz_id: string; total_questions: number; questions: Array<{ id: string; question: string; options: string[]; question_type: string }> }>(`/study/quizzes/${quizId}/start`, { method: 'POST' })
}

export async function submitQuizAnswer(quizId: string, attemptId: string, questionId: string, answer: string) {
  return apiClient<{ success: boolean; correct: boolean; correct_answer: string; explanation?: string }>('/study/quizzes/submit-answer', {
    method: 'POST',
    body: JSON.stringify({ quiz_id: quizId, attempt_id: attemptId, question_id: questionId, answer }),
  })
}

export async function completeQuizAttempt(quizId: string, attemptId: string) {
  return apiClient<QuizAttemptResult & { success: boolean }>('/study/quizzes/complete', {
    method: 'POST',
    body: JSON.stringify({ quiz_id: quizId, attempt_id: attemptId }),
  })
}

export async function deleteQuiz(quizId: string) {
  return apiClient<{ success: boolean; message: string }>(`/study/quizzes/${quizId}`, { method: 'DELETE' })
}
