import type { WeeklySummary } from '@/types'

interface Props { weekly: WeeklySummary }

export default function WeeklyActivity({ weekly }: Props) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
      <h3 className="font-semibold text-white">This Week</h3>
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center">
          <p className="text-xl font-bold text-white">{weekly.notes_generated}</p>
          <p className="text-xs text-gray-400">Notes</p>
        </div>
        <div className="text-center">
          <p className="text-xl font-bold text-white">{weekly.flashcards_reviewed}</p>
          <p className="text-xs text-gray-400">Flashcards</p>
        </div>
        <div className="text-center">
          <p className="text-xl font-bold text-white">{weekly.active_days}/7</p>
          <p className="text-xs text-gray-400">Active Days</p>
        </div>
      </div>
    </div>
  )
}
