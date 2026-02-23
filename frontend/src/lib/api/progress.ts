import { apiClient } from './client'
import type { StudyStats, TopicRanking, WeeklySummary, RecentActivity, DashboardStats, HealthResponse } from '@/types'

export async function getProgress() {
  return apiClient<{
    success: boolean
    statistics: StudyStats
    topic_rankings: TopicRanking[]
    weekly_summary: WeeklySummary
    recent_activities: RecentActivity[]
    streak: { current_streak: number; longest_streak: number; last_study_date: string | null }
  }>('/study/progress')
}

export async function getStudyStats() {
  return apiClient<{
    success: boolean
    statistics: StudyStats
    topic_rankings: TopicRanking[]
    weekly_summary: WeeklySummary
  }>('/study/progress/stats')
}

export async function recordActivity(activityType: string, topic: string, details?: Record<string, unknown>) {
  return apiClient<{ success: boolean; message: string }>('/study/progress/record', {
    method: 'POST',
    body: JSON.stringify({ activity_type: activityType, topic, details: details || {} }),
  })
}

export async function resetProgress() {
  return apiClient<{ success: boolean; message: string }>('/study/progress/reset', { method: 'DELETE' })
}

export async function getDashboardStats() {
  return apiClient<DashboardStats>('/dashboard-stats')
}

export async function getHealth() {
  return apiClient<HealthResponse>('/health')
}

export async function getSystemStats() {
  return apiClient<{ knowledge_base: Record<string, unknown>; agents: Record<string, string>; llm: { provider: string; model: string; is_local: boolean } }>('/stats')
}
