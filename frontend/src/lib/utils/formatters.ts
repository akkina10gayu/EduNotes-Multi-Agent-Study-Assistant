/**
 * Format a date string for display.
 */
export function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

/**
 * Format a date string with time.
 */
export function formatDateTime(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

/**
 * Format a score as percentage.
 */
export function formatScore(score: number): string {
  return `${Math.round(score)}%`
}

/**
 * Format streak display with fire emoji context.
 */
export function formatStreak(current: number, best: number): string {
  if (current === 0) return '0 days'
  const delta = best - current
  return delta > 0 ? `${current} days (best: ${best})` : `${current} days`
}

/**
 * Truncate text to a max length with ellipsis.
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}

/**
 * Get icon for query type.
 */
export function getQueryTypeIcon(queryType: string): string {
  switch (queryType) {
    case 'topic': return '📚'
    case 'url': return '🔗'
    case 'text': return '📝'
    default: return '📄'
  }
}
