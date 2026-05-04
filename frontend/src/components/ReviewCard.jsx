import { useState } from 'react'
import { AlertTriangle, Save, Loader2, ChevronDown, ChevronUp } from 'lucide-react'
import FieldCard from './FieldCard'

const FIELD_LABELS = {
  vendor: 'Vendor',
  date: 'Date',
  total: 'Total',
  address: 'Address',
}

export default function ReviewCard({ item, onSave }) {
  const flaggedFields = Object.entries(item.fields)
    .filter(([, f]) => f.needs_review)
    .map(([name]) => name)

  const [edits, setEdits] = useState(
    Object.fromEntries(flaggedFields.map(f => [f, item.fields[f].value]))
  )
  const [saving, setSaving] = useState(false)
  const [expanded, setExpanded] = useState(true)

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave(item.receipt_id, edits)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-xl border bg-white overflow-hidden"
      style={{ borderColor: '#FECACA' }}>

      {/* Card header */}
      <div className="px-4 py-3 flex items-center justify-between border-b"
        style={{ borderColor: '#FEE2E2', background: '#FFF7F7' }}>
        <div className="flex items-center gap-2">
          <AlertTriangle size={14} style={{ color: '#EF4444' }} />
          <span className="text-xs font-mono font-medium text-brand-text truncate max-w-xs">
            {item.receipt_id}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2 py-0.5 rounded-full font-medium"
            style={{ background: '#FEE2E2', color: '#EF4444' }}>
            {flaggedFields.length} field{flaggedFields.length !== 1 ? 's' : ''} flagged
          </span>
          <button onClick={() => setExpanded(e => !e)}
            className="p-1 rounded hover:bg-red-50 transition"
            style={{ color: '#9CA3AF' }}>
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="p-4 space-y-4">

          {/* All fields */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {Object.entries(item.fields).map(([field, data]) => (
              data.needs_review ? (
                <div key={field}
                  className="rounded-lg border p-3 space-y-2"
                  style={{ borderColor: '#FECACA', background: '#FFF7F7' }}>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wide"
                      style={{ color: '#6B7280' }}>
                      {FIELD_LABELS[field] ?? field}
                    </span>
                    <span className="text-xs font-medium"
                      style={{ color: '#EF4444' }}>
                      {Math.round(data.confidence * 100)}% confidence
                    </span>
                  </div>
                  <input
                    value={edits[field] ?? ''}
                    onChange={e => setEdits(prev => ({ ...prev, [field]: e.target.value }))}
                    className="w-full text-sm font-mono rounded-md px-3 py-1.5 border outline-none transition"
                    style={{
                      borderColor: '#FECACA',
                      background: '#FFFFFF',
                      color: '#1F2937',
                    }}
                    onFocus={e => e.target.style.borderColor = '#2F80ED'}
                    onBlur={e => e.target.style.borderColor = '#FECACA'}
                  />
                  <p className="text-xs" style={{ color: '#9CA3AF' }}>
                    Model predicted: <span className="font-mono">{data.value || '—'}</span>
                  </p>
                </div>
              ) : (
                <FieldCard
                  key={field}
                  label={FIELD_LABELS[field] ?? field}
                  value={data.value}
                  confidence={data.confidence}
                  needsReview={false}
                />
              )
            ))}
          </div>

          {/* Save button */}
          <div className="flex justify-end pt-1">
            <button
              onClick={handleSave}
              disabled={saving}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white transition-all"
              style={{
                background: saving ? '#93C5FD' : '#2F80ED',
                cursor: saving ? 'not-allowed' : 'pointer',
              }}
            >
              {saving
                ? <><Loader2 size={14} className="animate-spin" /> Saving…</>
                : <><Save size={14} /> Confirm & Save</>
              }
            </button>
          </div>
        </div>
      )}
    </div>
  )
}