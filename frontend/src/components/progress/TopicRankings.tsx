import type { TopicRanking } from '@/types'

interface Props { rankings: TopicRanking[] }

export default function TopicRankings({ rankings }: Props) {
  if (rankings.length === 0) return null

  const masteryColors: Record<string, string> = {
    Beginner: 'text-gray-400',
    Intermediate: 'text-blue-400',
    Advanced: 'text-purple-400',
    Expert: 'text-yellow-400',
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
      <h3 className="font-semibold text-white">Topic Mastery</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-500 text-xs">
            <th className="text-left pb-2">Topic</th>
            <th className="text-left pb-2">Level</th>
            <th className="text-right pb-2">Activities</th>
          </tr>
        </thead>
        <tbody>
          {rankings.slice(0, 5).map((r, i) => (
            <tr key={i} className="border-t border-gray-800">
              <td className="py-2 text-white">{r.topic}</td>
              <td className={`py-2 ${masteryColors[r.mastery_level] || 'text-gray-400'}`}>{r.mastery_level}</td>
              <td className="py-2 text-right text-gray-400">{r.activity_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
