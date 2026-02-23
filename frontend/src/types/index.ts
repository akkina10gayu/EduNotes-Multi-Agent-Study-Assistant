// Notes
export interface GenerateNotesRequest {
  query: string
  summarization_mode: string
  summary_length: string
  search_mode?: string
  research_mode?: boolean
}

export interface GenerateNotesResponse {
  success: boolean
  query_type: string
  query: string
  notes: string
  sources_used: number
  from_kb: boolean
  message?: string
  error?: string
  vision_data?: string
  source_file?: string
  extracted_text?: string
  related_papers?: RelatedPaper[]
}

export interface RelatedPaper {
  title: string
  authors: string[]
  summary: string
  url: string
  published: string
}

// Knowledge Base
export interface Document {
  id: string
  title: string
  topic: string
  content?: string
  source: string
  url?: string
  word_count?: number
  created_at: string
}

export interface SearchResult {
  content: string
  metadata: Record<string, string>
  similarity: number
}

// Flashcards
export interface Flashcard {
  id: string
  front: string
  back: string
  difficulty: string
  times_reviewed: number
  times_correct: number
  last_reviewed?: string
}

export interface FlashcardSet {
  id: string
  name: string
  topic: string
  cards: Flashcard[]
  card_count: number
  created_at: string
}

// Quizzes
export interface QuizQuestion {
  id: string
  question: string
  question_type: string
  options: string[]
  correct_answer: string
  explanation?: string
}

export interface Quiz {
  id: string
  title: string
  topic: string
  questions: QuizQuestion[]
  question_count: number
  created_at: string
}

export interface QuizAttemptResult {
  score: number
  correct_count: number
  total_questions: number
  detailed_results: QuizAnswerDetail[]
}

export interface QuizAnswerDetail {
  question: string
  options: string[]
  user_answer: string
  correct_answer: string
  is_correct: boolean
  explanation?: string
}

// Progress
export interface StudyStats {
  total_notes_generated: number
  total_flashcards_reviewed: number
  total_quizzes_completed: number
  topics_studied: number
  flashcard_accuracy: number
  quiz_accuracy: number
}

export interface TopicRanking {
  topic: string
  activity_count: number
  mastery_level: string
}

export interface WeeklySummary {
  notes_generated: number
  flashcards_reviewed: number
  quizzes_completed: number
  active_days: number
}

export interface RecentActivity {
  activity_type: string
  topic: string
  summary: string
  created_at: string
}

// Dashboard
export interface DashboardStats {
  healthy: boolean
  kb_documents: number
  flashcard_sets: number
  total_quizzes: number
  current_streak: number
  topics_count: number
}

export interface HealthResponse {
  status: string
  timestamp: string
  version: string
}
