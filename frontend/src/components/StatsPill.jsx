export default function StatsPill({ label, value, color }) {
  return (
    <div className="flex items-center gap-3 bg-white rounded-xl border border-brand-border px-4 py-3 min-w-[120px]">
      <div className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
      <div>
        <p className="text-xs" style={{ color: '#9CA3AF' }}>{label}</p>
        <p className="text-sm font-semibold text-brand-text">{value}</p>
      </div>
    </div>
  )
}