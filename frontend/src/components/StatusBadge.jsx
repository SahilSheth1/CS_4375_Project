import { CheckCircle2, Clock } from 'lucide-react'

export default function StatusBadge({ autoAccepted }) {
  return autoAccepted ? (
    <span className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-full"
      style={{ background: '#D1FAF8', color: '#0F9D93' }}>
      <CheckCircle2 size={12} />
      Auto-accepted
    </span>
  ) : (
    <span className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-full"
      style={{ background: '#FEF3C7', color: '#D97706' }}>
      <Clock size={12} />
      Needs review
    </span>
  )
}