import { Brain } from 'lucide-react'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Link from 'next/link'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AEGIS LLM Chain - Advanced AGI Platform',
  description: 'Enterprise-grade multi-LLM orchestration with cognitive reasoning, universal analysis, and distributed inference.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
          {/* Navigation */}
          <nav className="border-b border-white/10 bg-black/20 backdrop-blur-lg">
            <div className="container mx-auto px-4">
              <div className="flex h-16 items-center justify-between">
                <Link href="/" className="flex items-center space-x-3">
                  <Brain className="h-8 w-8 text-blue-400" />
                  <span className="text-xl font-bold text-white">AEGIS LLM Chain</span>
                </Link>
                <div className="flex items-center space-x-6">
                  <Link href="/" className="text-gray-300 hover:text-white transition-colors">
                    Dashboard
                  </Link>
                  <Link href="/business" className="text-gray-300 hover:text-white transition-colors">
                    Business
                  </Link>
                  <Link href="/cognitive" className="text-gray-300 hover:text-white transition-colors">
                    Cognitive
                  </Link>
                  <Link href="/analysis" className="text-gray-300 hover:text-white transition-colors">
                    Analysis
                  </Link>
                  <Link href="/generation" className="text-gray-300 hover:text-white transition-colors">
                    Generation
                  </Link>
                  <Link href="/chains" className="text-gray-300 hover:text-white transition-colors">
                    Chains
                  </Link>
                </div>
              </div>
            </div>
          </nav>

          {/* Main content */}
          <main className="container mx-auto px-4 py-8">
            {children}
          </main>

          {/* Footer */}
          <footer className="border-t border-white/10 bg-black/20 backdrop-blur-lg mt-16">
            <div className="container mx-auto px-4 py-6 text-center text-gray-400">
              <p>AEGIS LLM Chain © 2024 - Advanced AGI Platform</p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  )
}

