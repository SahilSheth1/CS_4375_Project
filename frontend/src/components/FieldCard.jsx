import { AlertTriangle, CheckCircle2 } from 'lucide-react'

export default function FieldCard({ label, value, confidence, needsReview, index = 0 }) {
  const pct = Math.round(confidence * 100)

  const barColor = needsReview
    ? '#EF4444'
    : pct >= 85
      ? '#2BBBAD'
      : '#F59E0B'

  const bgColor = needsReview ? '#FFF8F8' : '#FFFFFF'
  const borderColor = needsReview ? '#FECACA' : '#E5E9F0'

  return (
    <div
      className="rounded-xl border px-4 py-3 space-y-2 transition-all"
      style={{
        background: bgColor,
        borderColor,
        animation: `fadeUp 0.3s ease both`,
        animationDelay: `${index * 60}ms`,
      }}
    >
      <div className="flex items-center justify-between">
        <span
          className="text-xs font-semibold uppercase tracking-wider"
          style={{ color: '#9CA3AF' }}
        >
          {label}
        </span>
        {needsReview ? (
          <span
            className="flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full"
            style={{ background: '#FEE2E2', color: '#DC2626' }}
          >
            <AlertTriangle size={11} />
            Review
          </span>
        ) : (
          <span
            className="flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full"
            style={{ background: '#CCFBF1', color: '#0F766E' }}
          >
            <CheckCircle2 size={11} />
            Accepted
          </span>
        )}
      </div>

      <p className="text-sm font-medium font-mono leading-snug break-all text-brand-text">
        {value || <span style={{ color: '#D1D9E6' }}>—</span>}
      </p>

      <div className="space-y-1">
        <div className="flex justify-between items-center">
          <span className="text-xs" style={{ color: '#9CA3AF' }}>
            Confidence
          </span>
          <span
            className="text-xs font-semibold font-mono tabular-nums"
            style={{ color: barColor }}
          >
            {pct}%
          </span>
        </div>
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: '#F0F4FF' }}>
          <div
            className="h-full rounded-full"
            style={{
              width: `${pct}%`,
              background: barColor,
              animation: `growBar 0.6s ease both`,
              animationDelay: `${index * 60 + 150}ms`,
            }}
          />
        </div>
      </div>

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes growBar {
          from { width: 0%; }
        }
      `}</style>
    </div>
  )
}