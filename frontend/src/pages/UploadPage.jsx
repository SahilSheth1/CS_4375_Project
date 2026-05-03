import { useState, useRef, useCallback } from 'react'
import {
  UploadCloud, FileImage, X, Loader2, ChevronRight, AlertCircle
} from 'lucide-react'
import { uploadReceipt } from '../api'
import FieldCard from '../components/FieldCard'
import StatusBadge from '../components/StatusBadge'

const FIELD_LABELS = {
  vendor:  'Vendor',
  date:    'Date',
  total:   'Total',
  address: 'Address',
}

// Fields that need review get floated to the top so the user sees them first
function sortedFields(fields) {
  return Object.entries(fields).sort(([, a], [, b]) => {
    if (a.needs_review && !b.needs_review) return -1
    if (!a.needs_review && b.needs_review) return  1
    return a.confidence - b.confidence   // lower confidence first within each group
  })
}

function OverallConfidence({ fields }) {
  const values   = Object.values(fields)
  const avg      = values.reduce((s, f) => s + f.confidence, 0) / values.length
  const pct      = Math.round(avg * 100)
  const flagged  = values.filter(f => f.needs_review).length
  const color    = flagged > 0 ? '#D97706' : '#0F766E'
  const bgColor  = flagged > 0 ? '#FEF3C7' : '#CCFBF1'

  return (
    <div
      className="rounded-xl border px-4 py-3 flex items-center justify-between"
      style={{ background: bgColor, borderColor: flagged > 0 ? '#FDE68A' : '#99F6E4' }}
    >
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider" style={{ color }}>
          Overall Confidence
        </p>
        <p className="text-lg font-bold font-mono mt-0.5" style={{ color }}>
          {pct}%
        </p>
      </div>
      <div className="text-right">
        <p className="text-xs" style={{ color }}>
          {flagged === 0
            ? 'All fields accepted'
            : `${flagged} field${flagged > 1 ? 's' : ''} flagged`}
        </p>
        <p className="text-xs mt-0.5 font-mono" style={{ color: '#9CA3AF' }}>
          {values.length} fields extracted
        </p>
      </div>
    </div>
  )
}

export default function UploadPage() {
  const [file,     setFile]     = useState(null)
  const [preview,  setPreview]  = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)
  const inputRef = useRef()

  const handleFile = (f) => {
    if (!f) return
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(f.type)) {
      setError('Please upload a JPEG or PNG image.')
      return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const clearFile = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const res = await uploadReceipt(file)
      setResult(res.data)
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Could not connect to the API. Is the backend running?'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">

      <div>
        <h1 className="text-xl font-semibold text-brand-text">Upload Receipt</h1>
        <p className="text-sm mt-0.5" style={{ color: '#6B7280' }}>
          Upload a receipt image and the model will extract key fields automatically.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* ── Left: drop zone ── */}
        <div className="space-y-4">
          {!file ? (
            <div
              onDrop={onDrop}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onClick={() => inputRef.current?.click()}
              className="cursor-pointer rounded-xl border-2 border-dashed transition-all flex flex-col items-center justify-center gap-3 py-16 px-8 text-center"
              style={{
                borderColor: dragging ? '#2F80ED' : '#D1D9E6',
                background:  dragging ? '#EBF4FF' : '#FFFFFF',
              }}
            >
              <div
                className="rounded-full p-4"
                style={{ background: dragging ? '#DBEAFE' : '#F0F4FF' }}
              >
                <UploadCloud size={28} style={{ color: '#2F80ED' }} />
              </div>
              <div>
                <p className="font-medium text-brand-text text-sm">
                  {dragging ? 'Drop it here' : 'Drag & drop your receipt'}
                </p>
                <p className="text-xs mt-1" style={{ color: '#9CA3AF' }}>
                  or click to browse · JPEG, PNG
                </p>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png"
                className="hidden"
                onChange={(e) => handleFile(e.target.files[0])}
              />
            </div>
          ) : (
            <div className="rounded-xl overflow-hidden border border-brand-border bg-white shadow-sm">
              <div className="relative">
                <img
                  src={preview}
                  alt="Receipt preview"
                  className="w-full object-contain max-h-72"
                />
                <button
                  onClick={clearFile}
                  className="absolute top-2 right-2 rounded-full p-1 bg-white shadow border border-brand-border hover:bg-slate-50 transition"
                >
                  <X size={14} style={{ color: '#6B7280' }} />
                </button>
              </div>
              <div className="px-4 py-3 flex items-center gap-2 border-t border-brand-border">
                <FileImage size={15} style={{ color: '#2F80ED' }} />
                <span className="text-xs font-medium text-brand-text truncate flex-1">
                  {file.name}
                </span>
                <span className="text-xs" style={{ color: '#9CA3AF' }}>
                  {(file.size / 1024).toFixed(0)} KB
                </span>
              </div>
            </div>
          )}

          {error && (
            <div
              className="flex items-start gap-2 rounded-lg px-4 py-3 text-sm"
              style={{ background: '#FEF2F2', color: '#DC2626' }}
            >
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              {error}
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium text-white transition-all"
            style={{
              background: !file || loading ? '#93C5FD' : '#2F80ED',
              cursor:     !file || loading ? 'not-allowed' : 'pointer',
            }}
          >
            {loading
              ? <><Loader2 size={15} className="animate-spin" /> Analysing…</>
              : <><ChevronRight size={15} /> Extract Fields</>
            }
          </button>
        </div>

        {/* ── Right: results ── */}
        <div className="space-y-3">

          {/* Empty state */}
          {!result && !loading && (
            <div
              className="h-full flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-brand-border py-16"
              style={{ background: '#FFFFFF' }}
            >
              <FileImage size={28} style={{ color: '#D1D9E6' }} />
              <p className="text-xs" style={{ color: '#9CA3AF' }}>
                Extracted fields will appear here
              </p>
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="h-full flex flex-col items-center justify-center gap-3 rounded-xl border border-brand-border py-16 bg-white">
              <Loader2 size={28} className="animate-spin" style={{ color: '#2F80ED' }} />
              <p className="text-xs" style={{ color: '#6B7280' }}>Running model…</p>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="space-y-3">

              {/* Receipt ID + status */}
              <div
                className="flex items-center justify-between rounded-xl px-4 py-3 bg-white border border-brand-border"
                style={{ animation: 'fadeUp 0.25s ease both' }}
              >
                <div>
                  <p className="text-xs font-medium" style={{ color: '#6B7280' }}>
                    Receipt ID
                  </p>
                  <p className="text-xs font-mono mt-0.5 text-brand-text">
                    {result.receipt_id}
                  </p>
                </div>
                <StatusBadge autoAccepted={result.auto_accepted} />
              </div>

              {/* Overall confidence summary */}
              <div style={{ animation: 'fadeUp 0.25s ease both', animationDelay: '60ms' }}>
                <OverallConfidence fields={result.fields} />
              </div>

              {/* Field cards — flagged fields sorted to top */}
              {sortedFields(result.fields).map(([field, data], i) => (
                <FieldCard
                  key={field}
                  label={FIELD_LABELS[field] ?? field}
                  value={data.value}
                  confidence={data.confidence}
                  needsReview={data.needs_review}
                  index={i}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}