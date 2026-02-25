import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Navbar from '@/components/layout/Navbar'
import Sidebar from '@/components/layout/Sidebar'
import ErrorBoundary from '@/components/ui/ErrorBoundary'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'EduNotes - Multi-Agent Study Assistant',
  description: 'Transform topics, articles, PDFs into structured notes, flashcards, and quizzes',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-950 text-white min-h-screen`}>
        <Navbar />
        <div className="flex">
          <Sidebar />
          <main className="flex-1 p-6 max-w-7xl mx-auto">
            <ErrorBoundary>
              {children}
            </ErrorBoundary>
          </main>
        </div>
        <footer className="text-center text-gray-500 text-xs py-4 border-t border-gray-800">
          EduNotes v2.0 | Multi-Agent Study Assistant
        </footer>
      </body>
    </html>
  )
}
