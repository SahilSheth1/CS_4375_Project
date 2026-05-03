import { useState, useEffect, useCallback } from 'react'
import {
  ClipboardCheck, RefreshCw, Loader2,
  AlertCircle, CheckCircle2, Inbox
} from 'lucide-react'
import { getReviewQueue, getReviewStats, submitCorrection } from '../api'
import ReviewCard from '../components/ReviewCard'
import StatsPill from '../components/StatsPill'

export default function ReviewPage() {
  const [items,   setItems]   = useState([])
  const [stats,   setStats]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [saved,   setSaved]   = useState({})   // { receipt_id: true } after correction

  const fetchQueue = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [qRes, sRes] = await Promise.all([getReviewQueue(), getReviewStats()])
      setItems(qRes.data.items)
      setStats(sRes.data)
    } catch {
      setError('Could not load review queue. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchQueue() }, [fetchQueue])

  const handleSave = async (receiptId, corrections) => {
    await submitCorrection(receiptId, corrections)
    setSaved(prev => ({ ...prev, [receiptId]: true }))
    // Refresh stats after correction
    const sRes = await getReviewStats()
    setStats(sRes.data)
  }

  const pendingCount = items.filter(i => !saved[i.receipt_id]).length

  return (
    <div className="space-y-6">

      {/* Page title */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-semibold text-brand-text">Review Queue</h1>
          <p className="text-sm mt-0.5" style={{ color: '#6B7280' }}>
            Receipts where one or more fields fell below the confidence threshold.
          </p>
        </div>
        <button
          onClick={fetchQueue}
          disabled={loading}
          className="flex items-center gap-1.5 text-xs font-medium px-3 py-2 rounded-lg border border-brand-border bg-white hover:bg-slate-50 transition"
          style={{ color: '#6B7280' }}
        >
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Stats row */}
      {stats && (
        <div className="flex flex-wrap gap-3 fade-up">
          <StatsPill
            label="Total Processed"
            value={stats.total_processed}
            color="#2F80ED"
          />
          <StatsPill
            label="Auto-accepted"
            value={stats.auto_accepted}
            color="#2BBBAD"
          />
          <StatsPill
            label="Pending Review"
            value={stats.pending_review}
            color="#F59E0B"
          />
          <StatsPill
            label="Review Rate"
            value={`${(stats.review_rate * 100).toFixed(1)}%`}
            color="#8B5CF6"
          />
          <StatsPill
            label="Human Reviewed"
            value={stats.human_reviewed}
            color="#10B981"
          />
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg px-4 py-3 text-sm"
          style={{ background: '#FEF2F2', color: '#DC2626' }}>
          <AlertCircle size={15} className="shrink-0" />
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center gap-3 py-24">
          <Loader2 size={28} className="animate-spin" style={{ color: '#2F80ED' }} />
          <p className="text-xs" style={{ color: '#9CA3AF' }}>Loading queue…</p>
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && items.length === 0 && (
        <div className="flex flex-col items-center justify-center gap-3 py-24 rounded-xl border border-dashed border-brand-border bg-white">
          <Inbox size={32} style={{ color: '#D1D9E6' }} />
          <p className="text-sm font-medium text-brand-text">Queue is empty</p>
          <p className="text-xs" style={{ color: '#9CA3AF' }}>
            All receipts have been auto-accepted or reviewed.
          </p>
        </div>
      )}

      {/* Review cards */}
      {!loading && items.length > 0 && (
        <div className="space-y-4">
          {/* Pending */}
          {items.filter(i => !saved[i.receipt_id]).length > 0 && (
            <div className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-wide"
                style={{ color: '#6B7280' }}>
                Pending · {pendingCount}
              </p>
              {items
                .filter(i => !saved[i.receipt_id])
                .map((item, idx) => (
                  <div key={item.receipt_id}
                    className={`fade-up fade-up-delay-${Math.min(idx + 1, 4)}`}>
                    <ReviewCard
                      item={item}
                      onSave={handleSave}
                    />
                  </div>
                ))}
            </div>
          )}

          {/* Confirmed */}
          {Object.keys(saved).length > 0 && (
            <div className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-wide"
                style={{ color: '#2BBBAD' }}>
                Confirmed · {Object.keys(saved).length}
              </p>
              {items
                .filter(i => saved[i.receipt_id])
                .map(item => (
                  <div key={item.receipt_id}
                    className="rounded-xl border bg-white px-4 py-3 flex items-center gap-3"
                    style={{ borderColor: '#A7F3D0' }}>
                    <CheckCircle2 size={16} style={{ color: '#10B981' }} />
                    <span className="text-sm font-mono text-brand-text flex-1 truncate">
                      {item.receipt_id}
                    </span>
                    <span className="text-xs" style={{ color: '#6B7280' }}>
                      Confirmed
                    </span>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}