import { useState } from 'react'
import { ScanLine, Upload, ClipboardCheck } from 'lucide-react'
import UploadPage from './pages/UploadPage'
import ReviewPage from './pages/ReviewPage'

const NAV = [
  { id: 'upload', label: 'Upload',  icon: Upload },
  { id: 'review', label: 'Review',  icon: ClipboardCheck },
]

export default function App() {
  const [page, setPage] = useState('upload')

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#F7F9FC' }}>
      <header className="bg-white border-b border-brand-border sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ScanLine size={20} style={{ color: '#2F80ED' }} />
            <span className="font-semibold text-brand-text tracking-tight">
              ReceiptAI
            </span>
          </div>
          <nav className="flex gap-1">
            {NAV.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setPage(id)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all"
                style={page === id
                  ? { background: '#EBF4FF', color: '#2F80ED' }
                  : { color: '#6B7280' }
                }
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-4xl w-full mx-auto px-6 py-8">
        {page === 'upload' ? <UploadPage /> : <ReviewPage />}
      </main>

      <footer className="py-4 text-center text-xs" style={{ color: '#9CA3AF' }}>
        ReceiptAI · CS 4375 · ICDAR 2019 SROIE
      </footer>
    </div>
  )
}