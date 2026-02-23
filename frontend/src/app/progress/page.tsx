'use client'

import { useState, useEffect } from 'react'
import { getProgress } from '@/lib/api/progress'
import type { StudyStats, TopicRanking, WeeklySummary, RecentActivity } from '@/types'
import LoadingSpinner from '@/components/ui/LoadingSpinner'
import StreakDisplay from '@/components/progress/StreakDisplay'
import TopicRankings from '@/components/progress/TopicRankings'
import WeeklyActivity from '@/components/progress/WeeklyActivity'
import { formatDateTime } from '@/lib/utils/formatters'

export default function ProgressPage() {
  const [stats, setStats] = useState<StudyStats | null>(null)
  const [rankings, setRankings] = useState<TopicRanking[]>([])
  const [weekly, setWeekly] = useState<WeeklySummary | null>(null)
  const [activities, setActivities] = useState<RecentActivity[]>([])
  const [streak, setStreak] = useState({ current_streak: 0, longest_streak: 0 })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getProgress()
        if (data.success) {
          setStats(data.statistics)
          setRankings(data.topic_rankings || [])
          setWeekly(data.weekly_summary)
          setActivities(data.recent_activities || [])
          setStreak(data.streak || { current_streak: 0, longest_streak: 0 })
        }
      } catch (e) { console.error('Failed to load progress', e) }
      finally { setLoading(false) }
    }
    load()
  }, [])

  if (loading) return <LoadingSpinner message="Loading progress..." />

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Progress</h1>

      {/* Overall Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: 'Notes Generated', value: stats.total_notes_generated },
            { label: 'Flashcards Reviewed', value: stats.total_flashcards_reviewed },
            { label: 'Quizzes Completed', value: stats.total_quizzes_completed },
            { label: 'Topics Studied', value: stats.topics_studied },
          ].map(({ label, value }) => (
            <div key={label} className="bg-gray-900 border border-gray-800 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-white">{value}</p>
              <p className="text-xs text-gray-400">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Streak & Accuracy */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StreakDisplay currentStreak={streak.current_streak} longestStreak={streak.longest_streak} />
        {stats && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
            <h3 className="font-semibold text-white">Accuracy</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="text-center">
                <p className="text-xl font-bold text-[#6CA0DC]">{Math.round(stats.flashcard_accuracy)}%</p>
                <p className="text-xs text-gray-400">Flashcard Accuracy</p>
              </div>
              <div className="text-center">
                <p className="text-xl font-bold text-[#6CA0DC]">{Math.round(stats.quiz_accuracy)}%</p>
                <p className="text-xs text-gray-400">Quiz Accuracy</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Topic Rankings */}
      <TopicRankings rankings={rankings} />

      {/* This Week */}
      {weekly && <WeeklyActivity weekly={weekly} />}

      {/* Recent Activities */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
        <h3 className="font-semibold text-white">Recent Activity</h3>
        {activities.length === 0 ? (
          <p className="text-sm text-gray-500">No recent activity</p>
        ) : (
          activities.slice(0, 5).map((a, i) => (
            <div key={i} className="flex items-center justify-between text-sm border-b border-gray-800 pb-2 last:border-0">
              <span className="text-gray-300">{a.summary}</span>
              <span className="text-xs text-gray-500">{formatDateTime(a.created_at)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
