// api.js — all calls to the FastAPI backend
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
})

export const uploadReceipt = (file) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/upload/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const getReviewQueue = () => api.get('/review/')

export const getReceipt = (receiptId) => api.get(`/review/${receiptId}`)

export const submitCorrection = (receiptId, corrections) =>
  api.post(`/review/${receiptId}`, { corrections })

export const getReviewStats = () => api.get('/review/stats/summary')