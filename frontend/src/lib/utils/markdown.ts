/**
 * Clean vision descriptions — heading downshift, remove "No X present" lines.
 * Same logic as the backend's _clean_vision_desc.
 */
export function cleanVisionDescription(desc: string): string {
  if (!desc) return ''

  // Replace HTML <br> tags with newlines
  desc = desc.replace(/<br\s*\/?>/gi, '\n')
  // Remove remaining HTML tags
  desc = desc.replace(/<[^>]+>/g, '')

  const lines = desc.split('\n')
  const cleaned: string[] = []

  for (let i = 0; i < lines.length; i++) {
    const stripped = lines[i].trim()

    // Skip lines about absent content
    if (/\bno\b.{0,40}\b(present|found|detected|visible|shown|appear)/i.test(stripped)) {
      // Drop preceding heading if it belongs to this empty section
      if (cleaned.length > 0 && /^#{1,6}\s/.test(cleaned[cleaned.length - 1].trim())) {
        cleaned.pop()
      }
      continue
    }

    // Heading followed by a "no X" line — skip both
    if (/^#{1,6}\s/.test(stripped) && i + 1 < lines.length) {
      const nextStripped = lines[i + 1].trim()
      if (/\bno\b.{0,40}\b(present|found|detected|visible|shown|appear)/i.test(nextStripped)) {
        i++ // skip next line too
        continue
      }
    }

    cleaned.push(lines[i])
  }

  // Collapse 3+ blank lines to 2
  return cleaned.join('\n').replace(/\n{3,}/g, '\n\n').trim()
}
