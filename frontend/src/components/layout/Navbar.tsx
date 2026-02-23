'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import { useRouter } from 'next/navigation'

const navLinks = [
  { href: '/', label: 'Generate Notes' },
  { href: '/kb', label: 'Knowledge Base' },
  { href: '/study', label: 'Study Mode' },
  { href: '/progress', label: 'Progress' },
]

export default function Navbar() {
  const pathname = usePathname()
  const router = useRouter()

  const handleSignOut = async () => {
    const supabase = createClient()
    await supabase.auth.signOut()
    router.push('/auth/login')
    router.refresh()
  }

  if (pathname.startsWith('/auth')) return null

  return (
    <nav className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 flex items-center justify-between h-14">
        <Link href="/" className="text-xl font-bold" style={{ color: '#6CA0DC' }}>
          EduNotes Study Assistant
        </Link>

        <div className="flex items-center gap-1">
          {navLinks.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                pathname === href
                  ? 'bg-[#6CA0DC]/20 text-[#6CA0DC]'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {label}
            </Link>
          ))}

          <button
            onClick={handleSignOut}
            className="ml-4 px-3 py-1.5 text-sm text-gray-400 hover:text-red-400 transition-colors"
          >
            Sign Out
          </button>
        </div>
      </div>
    </nav>
  )
}
