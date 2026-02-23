interface Props { currentStreak: number; longestStreak: number }

export default function StreakDisplay({ currentStreak, longestStreak }: Props) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
      <h3 className="font-semibold text-white">Study Streak</h3>
      <div className="grid grid-cols-2 gap-3">
        <div className="text-center">
          <p className="text-2xl font-bold text-[#6CA0DC]">{currentStreak}</p>
          <p className="text-xs text-gray-400">Current Streak (days)</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-gray-300">{longestStreak}</p>
          <p className="text-xs text-gray-400">Longest Streak (days)</p>
        </div>
      </div>
    </div>
  )
}
