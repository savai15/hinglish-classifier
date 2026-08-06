import React, { useState, useEffect, useCallback, useRef, createContext, useContext } from 'react'
import { BrowserRouter, Routes, Route, NavLink, Outlet, useNavigate, useLocation, useSearchParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import Papa from 'papaparse'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Area, AreaChart,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  CartesianGrid, Legend,
} from 'recharts'
import {
  LayoutDashboard, MessageSquareWarning, Layers, History,
  BarChart3, Code2, Search, Bell, Settings, ChevronLeft,
  ChevronRight, Upload, Download, Trash2, Copy, Send,
  AlertTriangle, CheckCircle2, Clock, Zap, ArrowUpRight,
  ArrowDownRight, TrendingUp, Database, Brain, Gauge,
  FileText, Filter, RefreshCw, Eye, EyeOff, X, Menu,
  Keyboard, Sparkles, Target, Activity, PieChart as PieChartIcon,
  ThumbsUp, ThumbsDown, Bot, ClipboardList, Check, SquareX,
  Loader2, CornerDownLeft, Sun, Moon,
} from 'lucide-react'

const API = '/api'
const COLORS = ['#6366f1', '#8b5cf6', '#22d3ee', '#14b8a6', '#f59e0b', '#f97316', '#ef4444', '#ec4899', '#e879f9']
const URGENCY_COLORS = { High: '#ef4444', Medium: '#f59e0b', Low: '#22d3ee' }
const CAT_SHORT = {
  Account_Technical: 'Account', Customer_Service: 'Support', Delivery_Issue: 'Delivery',
  Order_Status: 'Order', Payment_Invoice: 'Payment', Pricing_Discount: 'Pricing',
  Product_Quality: 'Quality', Returns_Refunds: 'Returns', Wrong_Damaged_Product: 'Damaged',
}
const CATEGORIES = Object.keys(CAT_SHORT)
const RETRAIN_THRESHOLD = 20

// ============================================================================
// ERROR BOUNDARY
// ============================================================================

class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false, error: null } }
  static getDerivedStateFromError(error) { return { hasError: true, error } }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-[60vh] flex items-center justify-center">
          <div className="dark:bg-[#131825] bg-white border dark:border-white/10 border-gray-200 rounded-xl p-8 max-w-md text-center">
            <AlertTriangle className="mx-auto mb-4 text-red-400" size={48} />
            <h2 className="text-xl font-bold dark:text-white text-gray-900 mb-2">Something went wrong</h2>
            <p className="dark:text-gray-400 text-gray-500 mb-4">{this.state.error?.message || 'An unexpected error occurred'}</p>
            <button onClick={() => this.setState({ hasError: false, error: null })} className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors">
              Try Again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

// ============================================================================
// THEME SYSTEM
// ============================================================================

const ThemeContext = createContext()

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark')
  useEffect(() => {
    localStorage.setItem('theme', theme)
    document.documentElement.classList.toggle('dark', theme === 'dark')
    document.documentElement.classList.toggle('light', theme === 'light')
  }, [theme])
  return <ThemeContext.Provider value={{ theme, toggleTheme: () => setTheme(t => t === 'dark' ? 'light' : 'dark') }}>{children}</ThemeContext.Provider>
}

function useTheme() { return useContext(ThemeContext) }

// ============================================================================
// BRANDING SYSTEM
// ============================================================================

const BrandingContext = createContext()

function BrandingProvider({ children }) {
  const [branding, setBranding] = useState(() => {
    try { return JSON.parse(localStorage.getItem('branding')) || { companyName: 'HinglishAI', accentColor: '#6366f1' } }
    catch { return { companyName: 'HinglishAI', accentColor: '#6366f1' } }
  })

  useEffect(() => {
    localStorage.setItem('branding', JSON.stringify(branding))
    document.documentElement.style.setProperty('--accent', branding.accentColor)
  }, [branding])

  return <BrandingContext.Provider value={{ branding, setBranding }}>{children}</BrandingContext.Provider>
}

function useBranding() { return useContext(BrandingContext) }

// ============================================================================
// SKELETON COMPONENTS
// ============================================================================

function Skeleton({ className = '', count = 1 }) {
  return (
    <div className={`animate-pulse ${className}`}>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="bg-white/5 rounded" style={{ height: '100%', width: '100%' }} />
      ))}
    </div>
  )
}

function CardSkeleton() {
  return <Skeleton className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5 h-32" />
}

function TableSkeleton({ rows = 5, cols = 4 }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex gap-4">
          {Array.from({ length: cols }).map((_, j) => (
            <Skeleton key={j} className="h-10 flex-1 rounded-lg" />
          ))}
        </div>
      ))}
    </div>
  )
}

function ChartSkeleton() {
  return <Skeleton className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5 h-64" />
}

function KeyboardShortcutsModal({ open, onClose }) {
  const { theme } = useTheme()
  const shortcuts = [
    { category: 'Navigation', items: [
      { keys: ['G', 'D'], desc: 'Go to Dashboard' },
      { keys: ['G', 'C'], desc: 'Go to Classify' },
      { keys: ['G', 'B'], desc: 'Go to Batch' },
      { keys: ['G', 'H'], desc: 'Go to History' },
      { keys: ['G', 'A'], desc: 'Go to Analytics' },
      { keys: ['G', 'I'], desc: 'Go to AI Assistant' },
      { keys: ['G', 'R'], desc: 'Go to Review Queue' },
      { keys: ['G', 'P'], desc: 'Go to API Playground' },
    ]},
    { category: 'Actions', items: [
      { keys: ['Ctrl', 'K'], desc: 'Search' },
      { keys: ['Ctrl', 'Enter'], desc: 'Classify complaint' },
      { keys: ['?'], desc: 'Show shortcuts' },
      { keys: ['Esc'], desc: 'Close modal' },
    ]},
  ]

  if (!open) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[200] flex items-center justify-center bg-black/50 backdrop-blur-sm"
        onClick={onClose}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: -10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: -10 }}
          className={`w-full max-w-lg rounded-xl border p-6 shadow-2xl ${
            theme === 'dark' ? 'bg-[#131825] border-white/10' : 'bg-white border-gray-200'
          }`}
          onClick={e => e.stopPropagation()}
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <Keyboard size={20} className="text-violet-400" />
              <h2 className="text-lg font-bold dark:text-white text-gray-900">Keyboard Shortcuts</h2>
            </div>
            <button onClick={onClose} className="p-1 rounded-lg hover:bg-white/10 text-gray-400" aria-label="Close shortcuts modal">
              <X size={18} />
            </button>
          </div>
          <div className="space-y-5">
            {shortcuts.map(group => (
              <div key={group.category}>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">{group.category}</h3>
                <div className="space-y-1.5">
                  {group.items.map(item => (
                    <div key={item.desc} className="flex items-center justify-between py-1.5">
                      <span className="text-sm text-gray-300">{item.desc}</span>
                      <div className="flex gap-1">
                        {item.keys.map(key => (
                          <kbd key={key} className="px-2 py-0.5 text-xs font-mono bg-white/10 border border-white/10 rounded text-gray-300">
                            {key}
                          </kbd>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-xs text-gray-500 text-center">Press <kbd className="px-1.5 py-0.5 bg-white/10 rounded text-gray-400">?</kbd> anytime to show this</p>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

// ============================================================================
// ONBOARDING WIZARD
// ============================================================================

function OnboardingWizard({ onComplete }) {
  const [step, setStep] = useState(0)
  const { theme } = useTheme()
  const navigate = useNavigate()

  const steps = [
    {
      title: 'Welcome to HinglishAI',
      desc: 'Classify Hinglish e-commerce complaints into 9 categories and 3 urgency levels using machine learning.',
      icon: Zap,
      color: 'from-violet-500 to-purple-500',
    },
    {
      title: 'Classify Complaints',
      desc: 'Type or paste any Hinglish complaint and get instant AI-powered classification with confidence scores.',
      icon: MessageSquareWarning,
      color: 'from-cyan-500 to-blue-500',
    },
    {
      title: 'Batch Process',
      desc: 'Upload a CSV file or paste multiple complaints to classify them all at once.',
      icon: Layers,
      color: 'from-emerald-500 to-teal-500',
    },
    {
      title: 'AI Assistant',
      desc: 'Get AI-powered resolution steps and draft customer responses using Groq.',
      icon: Bot,
      color: 'from-amber-500 to-orange-500',
    },
  ]

  const handleComplete = () => {
    localStorage.setItem('onboarding_complete', 'true')
    onComplete()
  }

  const handleSkip = () => {
    localStorage.setItem('onboarding_complete', 'true')
    onComplete()
  }

  const current = steps[step]
  const Icon = current.icon

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="fixed inset-0 z-[300] flex items-center justify-center bg-black/60 backdrop-blur-sm"
    >
      <motion.div
        key={step}
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: -20 }}
        className={`w-full max-w-md rounded-2xl border p-8 text-center ${
          theme === 'dark' ? 'bg-[#131825] border-white/10' : 'bg-white border-gray-200'
        }`}
      >
        <div className={`w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br ${current.color} flex items-center justify-center mb-6`}>
          <Icon size={32} className="text-white" />
        </div>
        <h2 className="text-xl font-bold text-white mb-2">{current.title}</h2>
        <p className="text-gray-400 text-sm mb-8">{current.desc}</p>

        <div className="flex justify-center gap-1.5 mb-6">
          {steps.map((_, i) => (
            <div key={i} className={`h-1.5 rounded-full transition-all ${i === step ? 'w-8 bg-violet-500' : 'w-2 bg-white/20'}`} />
          ))}
        </div>

        <div className="flex gap-3">
          <button onClick={handleSkip} className="flex-1 py-2.5 text-sm text-gray-400 hover:text-white transition-colors">
            Skip
          </button>
          {step < steps.length - 1 ? (
            <button onClick={() => setStep(s => s + 1)} className="flex-1 py-2.5 bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors font-medium">
              Next
            </button>
          ) : (
            <button onClick={handleComplete} className="flex-1 py-2.5 bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors font-medium">
              Get Started
            </button>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}

// ============================================================================
// TOAST SYSTEM
// ============================================================================

const ToastContext = createContext()

let toastId = 0

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((message, type = 'info', duration = 4000) => {
    const id = ++toastId
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, duration)
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <div className="fixed top-4 right-4 z-[100] flex flex-col gap-2 pointer-events-none max-w-sm">
        <AnimatePresence>
          {toasts.map(t => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, x: 80, scale: 0.95 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 80, scale: 0.95 }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              className={`pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-xl border shadow-2xl backdrop-blur-sm ${
                t.type === 'success'
                  ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
                  : t.type === 'error'
                  ? 'bg-red-500/10 border-red-500/20 text-red-300'
                  : 'bg-cyan-500/10 border-cyan-500/20 text-cyan-300'
              }`}
            >
              {t.type === 'success' && <CheckCircle2 size={16} className="flex-shrink-0" />}
              {t.type === 'error' && <AlertTriangle size={16} className="flex-shrink-0" />}
              {t.type === 'info' && <Bell size={16} className="flex-shrink-0" />}
              <p className="text-sm flex-1">{t.message}</p>
              <button onClick={() => removeToast(t.id)} className="flex-shrink-0 opacity-60 hover:opacity-100" aria-label="Dismiss notification">
                <X size={14} />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  )
}

function useToast() {
  return useContext(ToastContext)
}

// ============================================================================
// API HELPERS
// ============================================================================

async function apiFetch(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  })
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

async function predict(text) {
  return apiFetch('/predict', { method: 'POST', body: JSON.stringify({ text }) })
}

async function predictBatch(texts) {
  return apiFetch('/predict/batch', { method: 'POST', body: JSON.stringify({ texts }) })
}

async function getHistory(params = {}) {
  const q = new URLSearchParams(params).toString()
  return apiFetch(`/history?${q}`)
}

async function getStats() { return apiFetch('/stats') }
async function getTimeline(h = 24) { return apiFetch(`/analytics/timeline?hours=${h}`) }
async function getWordFrequency(cat) { return apiFetch(`/analytics/word-frequency${cat ? `?category=${cat}` : ''}`) }
async function getConfidence() { return apiFetch('/analytics/confidence') }
async function getPatterns() { return apiFetch('/analytics/patterns') }
async function getCategories() { return apiFetch('/categories') }
async function getRetrainStatus() { return apiFetch('/retrain/status') }
async function getRetrainHistory() { return apiFetch('/retrain/history') }
async function triggerRetrain() { return apiFetch('/retrain', { method: 'POST' }) }
async function getLowConfidence() { return apiFetch('/low-confidence') }

async function submitFeedback(payload) {
  return apiFetch('/feedback', { method: 'POST', body: JSON.stringify(payload) })
}

async function aiResolve(text, category, urgency) {
  return apiFetch('/ai/resolve', { method: 'POST', body: JSON.stringify({ text, category, urgency }) })
}

async function aiDraftResponse(text, category, urgency) {
  return apiFetch('/ai/draft-response', { method: 'POST', body: JSON.stringify({ text, category, urgency }) })
}

// ============================================================================
// ANIMATED NUMBER
// ============================================================================

function AnimNum({ value, duration = 800, decimals = 0 }) {
  const [display, setDisplay] = useState(0)
  const ref = useRef(null)
  useEffect(() => {
    const start = display
    const diff = value - start
    if (diff === 0) return
    const startTime = performance.now()
    function tick(now) {
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setDisplay(Number((start + diff * eased).toFixed(decimals)))
      if (progress < 1) ref.current = requestAnimationFrame(tick)
    }
    ref.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(ref.current)
  }, [value, duration, decimals])
  return <span>{display.toLocaleString(undefined, decimals > 0 ? { minimumFractionDigits: decimals, maximumFractionDigits: decimals } : {})}</span>
}

// ============================================================================
// SHARED COMPONENTS
// ============================================================================

function LoadingState() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
        <p className="text-sm dark:text-gray-500 text-gray-400">Loading...</p>
      </div>
    </div>
  )
}

function FeedbackPanel({ predictionId, text, predictedCategory, predictedUrgency, onFeedbackSubmitted }) {
  const { addToast } = useToast()
  const [status, setStatus] = useState(null)
  const [correctedCategory, setCorrectedCategory] = useState('')
  const [correctedUrgency, setCorrectedUrgency] = useState('')
  const [submitting, setSubmitting] = useState(false)

  if (status === 'submitted') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-4 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center gap-2"
      >
        <CheckCircle2 size={16} className="text-emerald-400" />
        <span className="text-sm text-emerald-300">Feedback recorded. Thank you!</span>
      </motion.div>
    )
  }

  async function handleSubmit(isCorrect) {
    if (!predictionId) {
      addToast('No prediction ID available', 'error')
      return
    }

    if (!isCorrect && !correctedCategory && !correctedUrgency) {
      addToast('Please select corrections', 'error')
      return
    }

    setSubmitting(true)
    try {
      const payload = {
        prediction_id: predictionId,
        is_correct_category: isCorrect || !!correctedCategory ? false : true,
        is_correct_urgency: isCorrect || !!correctedUrgency ? false : true,
      }
      if (!isCorrect && correctedCategory) payload.corrected_category = correctedCategory
      if (!isCorrect && correctedUrgency) payload.corrected_urgency = correctedUrgency
      if (isCorrect) {
        payload.is_correct_category = true
        payload.is_correct_urgency = true
      }
      const res = await submitFeedback(payload)
      setStatus('submitted')
      addToast(
        res.should_retrain
          ? 'Feedback recorded. Retrain threshold reached!'
          : `Feedback recorded. ${res.corrections_total}/${RETRAIN_THRESHOLD} corrections.`,
        'success'
      )
      if (onFeedbackSubmitted) onFeedbackSubmitted(res)
    } catch {
      addToast('Failed to submit feedback', 'error')
    }
    setSubmitting(false)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-4 p-4 rounded-lg dark:bg-white/5 bg-gray-100 dark:border-white/10 border-gray-200"
    >
      <p className="text-xs dark:text-gray-400 text-gray-500 mb-3">Was this classification correct?</p>
      <div className="flex items-center gap-3 mb-3">
        <button
          onClick={() => handleSubmit(true)}
          disabled={submitting}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm hover:bg-emerald-500/20 transition disabled:opacity-50"
        >
          <ThumbsUp size={14} />
          Correct
        </button>
        <button
          onClick={() => setStatus('incorrect')}
          disabled={submitting}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm hover:bg-red-500/20 transition disabled:opacity-50"
        >
          <ThumbsDown size={14} />
          Incorrect
        </button>
      </div>

      {status === 'incorrect' && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="space-y-3 mt-3 pt-3 border-t dark:border-white/5 border-gray-200"
        >
          <p className="text-xs dark:text-gray-500 text-gray-400">Provide corrections:</p>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] dark:text-gray-500 text-gray-400 uppercase tracking-wider block mb-1">Correct Category</label>
              <select
                value={correctedCategory}
                onChange={e => setCorrectedCategory(e.target.value)}
                className="w-full px-3 py-1.5 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg text-xs dark:text-white text-gray-900 outline-none focus:border-cyan-500/50"
              >
                <option value="">Keep: {CAT_SHORT[predictedCategory]}</option>
                {CATEGORIES.map(c => (
                  <option key={c} value={c}>{CAT_SHORT[c]}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] dark:text-gray-500 text-gray-400 uppercase tracking-wider block mb-1">Correct Urgency</label>
              <select
                value={correctedUrgency}
                onChange={e => setCorrectedUrgency(e.target.value)}
                className="w-full px-3 py-1.5 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg text-xs dark:text-white text-gray-900 outline-none focus:border-cyan-500/50"
              >
                <option value="">Keep: {predictedUrgency}</option>
                <option value="High">High</option>
                <option value="Medium">Medium</option>
                <option value="Low">Low</option>
              </select>
            </div>
          </div>
          <button
            onClick={() => handleSubmit(false)}
            disabled={submitting}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-violet-500 to-purple-500 text-white text-sm font-medium disabled:opacity-50 hover:shadow-lg hover:shadow-violet-500/20 transition-all duration-300"
          >
            {submitting ? <Loader2 size={12} className="animate-spin" /> : <Send size={12} />}
            Submit Correction
          </button>
        </motion.div>
      )}
    </motion.div>
  )
}

function RetrainBanner({ correctionsTotal, onRetrain }) {
  const { addToast } = useToast()
  const [retraining, setRetraining] = useState(false)
  const [result, setResult] = useState(null)

  if (correctionsTotal < RETRAIN_THRESHOLD && !result) return null

  async function handleRetrain() {
    setRetraining(true)
    try {
      const res = await triggerRetrain()
      setResult(res)
      addToast(`Retrain complete! Accuracy: ${(res.accuracy * 100).toFixed(1)}%`, 'success')
      if (onRetrain) onRetrain(res)
    } catch {
      addToast('Retrain failed. Try again later.', 'error')
    }
    setRetraining(false)
  }

  if (result) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-4 rounded-xl bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border border-emerald-500/20"
      >
        <div className="flex items-center gap-3">
          <CheckCircle2 size={20} className="text-emerald-400" />
          <div className="flex-1">
            <p className="text-sm font-medium text-emerald-300">Model Retrained Successfully</p>
            <p className="text-xs text-emerald-400/70">Accuracy: {(result.accuracy * 100).toFixed(1)}%</p>
          </div>
          <button onClick={() => setResult(null)} className="text-emerald-400/50 hover:text-emerald-400">
            <X size={14} />
          </button>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4 rounded-xl bg-gradient-to-r from-violet-500/10 to-purple-500/10 border border-violet-500/20"
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
          <RefreshCw size={18} className="text-white" />
        </div>
        <div className="flex-1">
          <p className="text-sm font-medium text-violet-300">Model Retrain Ready</p>
          <p className="text-xs text-violet-400/70">{correctionsTotal} corrections collected — retrain to improve accuracy</p>
        </div>
        <button
          onClick={handleRetrain}
          disabled={retraining}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-violet-500 to-purple-500 text-white text-sm font-medium disabled:opacity-50 hover:shadow-lg hover:shadow-violet-500/20 transition-all duration-300"
        >
          {retraining ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
          {retraining ? 'Retraining...' : 'Retrain Now'}
        </button>
      </div>
    </motion.div>
  )
}

// ============================================================================
// LAYOUT
// ============================================================================

function Layout() {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searching, setSearching] = useState(false)
  const [retrainStatus, setRetrainStatus] = useState(null)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const [showOnboarding, setShowOnboarding] = useState(() => {
    return !localStorage.getItem('onboarding_complete')
  })
  const [pendingKey, setPendingKey] = useState(null)
  const keyTimeoutRef = useRef(null)
  const searchRef = useRef(null)
  const navigate = useNavigate()
  const location = useLocation()
  const { theme, toggleTheme } = useTheme()
  const { branding } = useBranding()

  useEffect(() => {
    function handleKey(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return

      // Chord detection: G then D/C/B/etc
      if (pendingKey === 'g') {
        clearTimeout(keyTimeoutRef.current)
        setPendingKey(null)
        const routes = { d: '/', c: '/classify', b: '/batch', h: '/history', a: '/analytics', i: '/ai', r: '/review', p: '/playground' }
        if (routes[e.key]) {
          e.preventDefault()
          navigate(routes[e.key])
        }
        return
      }

      if (e.key === 'g' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        setPendingKey('g')
        keyTimeoutRef.current = setTimeout(() => setPendingKey(null), 500)
        return
      }

      // ? key opens shortcuts
      if (e.key === '?' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        setShortcutsOpen(true)
        return
      }

      // Ctrl+K for search
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        setSearchOpen(true)
      }

      // Escape closes modals
      if (e.key === 'Escape') {
        setSearchOpen(false)
        setShortcutsOpen(false)
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [pendingKey, navigate])

  useEffect(() => {
    if (!searchOpen) return
    const timer = setTimeout(async () => {
      if (searchQuery.length > 1) {
        setSearching(true)
        try {
          const data = await apiFetch(`/search?q=${encodeURIComponent(searchQuery)}&limit=10`)
          setSearchResults(data.predictions || [])
        } catch { setSearchResults([]) }
        setSearching(false)
      } else {
        setSearchResults([])
      }
    }, 300)
    return () => clearTimeout(timer)
  }, [searchQuery, searchOpen])

  useEffect(() => {
    getRetrainStatus().then(setRetrainStatus).catch(() => {})
    const interval = setInterval(() => {
      getRetrainStatus().then(setRetrainStatus).catch(() => {})
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  const nav = [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/classify', icon: MessageSquareWarning, label: 'Classify' },
    { to: '/batch', icon: Layers, label: 'Batch' },
    { to: '/history', icon: History, label: 'History' },
    { to: '/analytics', icon: BarChart3, label: 'Analytics' },
    { to: '/ai', icon: Bot, label: 'AI Assistant' },
    { to: '/review', icon: ClipboardList, label: 'Review Queue' },
    { to: '/playground', icon: Code2, label: 'API' },
    { to: '/compare', icon: Layers, label: 'Compare' },
    { to: '/settings', icon: Settings, label: 'Settings' },
  ]

  const shouldShowRetrainBadge = retrainStatus && retrainStatus.should_retrain

  return (
    <div className="flex h-screen dark:bg-[#0a0e1a] bg-gray-50 dark:text-gray-200 text-gray-800 overflow-hidden">
      {/* Sidebar */}
      <motion.aside
        animate={{ width: collapsed ? 64 : 220 }}
        className={`flex-shrink-0 dark:bg-[#0d1220] bg-white border-r dark:border-white/5 border-gray-200 flex flex-col z-20 ${
          mobileOpen ? 'fixed inset-y-0 left-0' : 'hidden lg:flex'
        }`}
      >
        <div className="h-14 flex items-center px-4 border-b dark:border-white/5 border-gray-200">
          {!collapsed && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2 min-w-0">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center flex-shrink-0">
                <Zap size={14} className="text-white" />
              </div>
              <span className="font-semibold text-sm dark:text-white text-gray-900 truncate">{branding.companyName}</span>
            </motion.div>
          )}
          {collapsed && (
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center mx-auto">
              <Zap size={14} className="text-white" />
            </div>
          )}
        </div>

        {/* Retrain Available Banner */}
        {!collapsed && shouldShowRetrainBadge && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="px-3 pt-2"
          >
            <NavLink
              to="/classify"
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-violet-500/10 border border-violet-500/20 text-violet-300 text-xs hover:bg-violet-500/20 transition"
            >
              <RefreshCw size={12} />
              <span className="truncate">Retrain Available</span>
              <span className="ml-auto bg-violet-500/20 text-violet-300 text-[10px] px-1.5 py-0.5 rounded-full font-medium">
                {retrainStatus.corrections_total}
              </span>
            </NavLink>
          </motion.div>
        )}

        <nav className="flex-1 py-3 px-2 space-y-0.5">
          {nav.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-200 group relative ${
                  isActive
                    ? 'dark:bg-white/10 bg-gray-100 dark:text-white text-gray-900'
                    : 'dark:text-gray-400 text-gray-500 dark:hover:text-white hover:text-gray-900 dark:hover:bg-white/5 hover:bg-gray-100'
                } ${collapsed ? 'justify-center' : ''}`
              }
            >
              <item.icon size={18} className="flex-shrink-0" />
              {!collapsed && <span className="truncate">{item.label}</span>}
              {collapsed && item.to === '/review' && retrainStatus && retrainStatus.remaining !== undefined && retrainStatus.remaining > 0 && (
                <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-amber-400" />
              )}
            </NavLink>
          ))}
        </nav>

        {/* Sidebar Footer */}
        <div className="p-2 border-t dark:border-white/5 border-gray-200 space-y-1">
          {!collapsed && retrainStatus && (
            <div className="px-3 py-2 rounded-lg dark:bg-white/5 bg-gray-100 text-xs dark:text-gray-500 text-gray-400 flex items-center justify-between">
              <span>Corrections</span>
              <span className="dark:text-gray-400 text-gray-500 font-medium">
                {retrainStatus.corrections_total}/{retrainStatus.threshold || RETRAIN_THRESHOLD}
              </span>
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg dark:text-gray-500 text-gray-400 dark:hover:text-white hover:text-gray-900 dark:hover:bg-white/5 hover:bg-gray-100 transition text-sm"
          >
            {collapsed ? <ChevronRight size={16} /> : <><ChevronLeft size={16} /><span className="truncate">Collapse</span></>}
          </button>
        </div>
      </motion.aside>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <header className="h-14 flex items-center justify-between px-6 border-b dark:border-white/5 border-gray-200 dark:bg-[#0d1220]/80 bg-white/80 backdrop-blur-sm z-10">
          <div className="flex items-center gap-3">
            <button
              onClick={() => { setCollapsed(!collapsed); setMobileOpen(!mobileOpen) }}
              className="lg:hidden p-2 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 transition dark:text-gray-400 text-gray-500"
              aria-label="Toggle menu"
            >
              <Menu size={18} />
            </button>
            <h1 className="text-sm font-medium dark:text-gray-300 text-gray-600">
              {nav.find(n => n.to === location.pathname)?.label || 'Hinglish Complaint Classifier'}
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 transition-colors dark:text-gray-400 text-gray-500 dark:hover:text-white hover:text-gray-900"
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button
              onClick={() => setSearchOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg dark:bg-white/5 bg-gray-100 dark:border-white/10 border-gray-200 dark:text-gray-400 text-gray-500 dark:hover:text-white hover:text-gray-900 dark:hover:border-white/20 border-gray-300 transition text-sm"
            >
              <Search size={14} />
              <span className="hidden sm:inline">Search</span>
              <kbd className="hidden sm:inline text-[10px] dark:bg-white/10 bg-gray-200 px-1.5 py-0.5 rounded dark:text-gray-500 text-gray-400">Ctrl+K</kbd>
            </button>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-[1400px] mx-auto p-6">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Search Modal */}
      <AnimatePresence>
        {searchOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] bg-black/60 backdrop-blur-sm"
            onClick={() => setSearchOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -10 }}
              className="w-full max-w-lg dark:bg-[#131825] bg-white border dark:border-white/10 border-gray-200 rounded-xl shadow-2xl overflow-hidden"
              onClick={e => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 px-4 py-3 border-b dark:border-white/5 border-gray-200">
                <Search size={16} className="dark:text-gray-500 text-gray-400" />
                <input
                  ref={searchRef}
                  autoFocus
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  placeholder="Search complaints..."
                  className="flex-1 bg-transparent text-sm dark:text-white text-gray-900 placeholder-gray-500 outline-none"
                />
                <kbd className="text-[10px] dark:bg-white/10 bg-gray-200 px-1.5 py-0.5 rounded dark:text-gray-500 text-gray-400">ESC</kbd>
              </div>
              <div className="max-h-80 overflow-y-auto">
                {searching && <div className="px-4 py-6 text-center dark:text-gray-500 text-gray-400 text-sm">Searching...</div>}
                {!searching && searchResults.length === 0 && searchQuery.length > 1 && (
                  <div className="px-4 py-6 text-center dark:text-gray-500 text-gray-400 text-sm">No results found</div>
                )}
                {searchResults.map((p, i) => (
                  <button
                    key={i}
                    className="w-full text-left px-4 py-3 dark:hover:bg-white/5 hover:bg-gray-100 transition border-b dark:border-white/5 border-gray-200 last:border-0"
                    onClick={() => { setSearchOpen(false); navigate(`/history?search=${encodeURIComponent(p.text?.slice(0, 50) || '')}`) }}
                  >
                    <p className="text-sm dark:text-white text-gray-900 truncate">{p.text}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs px-1.5 py-0.5 rounded dark:bg-white/10 bg-gray-100 dark:text-gray-400 text-gray-500">{CAT_SHORT[p.predicted_category] || p.predicted_category}</span>
                      <span className="text-xs" style={{ color: URGENCY_COLORS[p.predicted_urgency] }}>{p.predicted_urgency}</span>
                    </div>
                  </button>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <KeyboardShortcutsModal open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />

      {showOnboarding && <OnboardingWizard onComplete={() => setShowOnboarding(false)} />}
    </div>
  )
}

// ============================================================================
// PAGE: DASHBOARD
// ============================================================================

function DashboardPage() {
  const [stats, setStats] = useState(null)
  const [patterns, setPatterns] = useState(null)
  const [timeline, setTimeline] = useState(null)
  const [categories, setCategories] = useState(null)
  const [retrainHistory, setRetrainHistory] = useState(null)
  const [loading, setLoading] = useState(true)
  const [suggestions, setSuggestions] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    Promise.all([getStats(), getPatterns(), getTimeline(24), getCategories(), getRetrainHistory()])
      .then(([s, p, t, c, r]) => { setStats(s); setPatterns(p); setTimeline(t); setCategories(c); setRetrainHistory(r) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    apiFetch('/analytics/suggestions').then(setSuggestions).catch(() => {})
  }, [])

  if (loading) return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => <CardSkeleton key={i} />)}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {Array.from({ length: 3 }).map((_, i) => <ChartSkeleton key={i} />)}
      </div>
    </div>
  )

  const catData = stats?.category_distribution ? Object.entries(stats.category_distribution).map(([k, v]) => ({
    name: CAT_SHORT[k] || k, fullName: k, value: v
  })) : []

  const urgData = stats?.urgency_distribution ? Object.entries(stats.urgency_distribution).map(([k, v]) => ({
    name: k, value: v
  })) : []

  const timelineData = timeline?.timeline?.map(t => ({
    time: t.hour?.split(' ')[1] || t.hour,
    predictions: t.count,
    confidence: Math.round((t.avg_confidence || 0) * 100),
  })) || []

  const retrainLog = retrainHistory?.history || []

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Total Predictions', value: stats?.total_predictions || 0, icon: Database, color: 'from-cyan-500 to-blue-500', trend: '+12%' },
          { label: 'Category F1 Score', value: 99.69, decimals: 2, icon: Target, color: 'from-violet-500 to-purple-500', suffix: '%', trend: 'TF-IDF + SVM' },
          { label: 'Urgency F1 Score', value: 99.96, decimals: 2, icon: Gauge, color: 'from-emerald-500 to-teal-500', suffix: '%', trend: 'Combined Ensemble' },
          { label: 'Needs Review', value: stats?.low_confidence_count || 0, icon: AlertTriangle, color: 'from-amber-500 to-orange-500', trend: stats?.low_confidence_count > 0 ? 'Attention' : 'Clear' },
        ].map((card, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5 dark:hover:border-white/10 hover:border-gray-300 transition-all duration-300"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs dark:text-gray-500 text-gray-400 uppercase tracking-wider">{card.label}</p>
                <p className="text-3xl font-bold dark:text-white text-gray-900 mt-1">
                  <AnimNum value={card.value} decimals={card.decimals || 0} />{card.suffix || ''}
                </p>
              </div>
              <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${card.color} flex items-center justify-center`}>
                <card.icon size={18} className="text-white" />
              </div>
            </div>
            <div className="mt-3 flex items-center gap-1">
              <ArrowUpRight size={12} className="text-emerald-400" />
              <span className="text-xs text-emerald-400">{card.trend}</span>
              <span className="text-xs dark:text-gray-600 text-gray-400 ml-1">vs last week</span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Category Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-4">Category Distribution</h3>
          {catData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={catData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={2} dataKey="value">
                  {catData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">No data yet</div>
          )}
          <div className="flex flex-wrap gap-2 mt-2">
            {catData.slice(0, 6).map((d, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full" style={{ background: COLORS[i] }} />
                <span className="text-[10px] dark:text-gray-500 text-gray-400">{d.name}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Urgency Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-4">Urgency Breakdown</h3>
          {urgData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={urgData} barSize={40}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} />
                <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 12 }} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {urgData.map((entry, i) => (
                    <Cell key={i} fill={URGENCY_COLORS[entry.name] || COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">No data yet</div>
          )}
        </motion.div>

        {/* Prediction Timeline */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-4">Last 24 Hours</h3>
          {timelineData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={timelineData}>
                <defs>
                  <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="time" tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} />
                <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 12 }} />
                <Area type="monotone" dataKey="predictions" stroke="#22d3ee" fill="url(#colorPred)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">No predictions yet</div>
          )}
        </motion.div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Model Performance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-4">Model Performance</h3>
          <div className="space-y-3">
            {[
              { label: 'TF-IDF + SVM — Category F1', score: 99.69, color: '#6366f1' },
              { label: 'Combined Ensemble — Urgency F1', score: 99.96, color: '#22d3ee' },
              { label: 'MuRIL — Category F1', score: 99.87, color: '#8b5cf6' },
              { label: 'MuRIL — Urgency F1', score: 100.00, color: '#a78bfa' },
            ].map((m, i) => (
              <div key={i}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="dark:text-gray-400 text-gray-500">{m.label}</span>
                  <span className="dark:text-white text-gray-900 font-medium">{m.note || `F1: ${m.score.toFixed(2)}%`}</span>
                </div>
                <div className="h-2 dark:bg-white/5 bg-gray-200 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${m.score}%` }}
                    transition={{ duration: 1, delay: 0.8 + i * 0.1 }}
                    className="h-full rounded-full"
                    style={{ background: m.color }}
                  />
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Retrain History */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">Retrain History</h3>
            <RefreshCw size={14} className="dark:text-gray-500 text-gray-400" />
          </div>
          {retrainLog.length === 0 ? (
            <div className="h-32 flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">
              No retrains yet
            </div>
          ) : (
            <div className="space-y-2 max-h-[220px] overflow-y-auto">
              {retrainLog.slice(0, 5).map((entry, i) => (
                <div key={i} className="p-3 rounded-lg dark:bg-white/5 bg-gray-100 border dark:border-white/5 border-gray-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
                        <Check size={10} className="text-white" />
                      </div>
                      <div>
                        <p className="text-xs dark:text-white text-gray-900 font-medium">v{entry.model_version || i + 1}</p>
                        <p className="text-[10px] dark:text-gray-500 text-gray-400">
                          {entry.corrections_count} corrections
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      {entry.accuracy_before != null && entry.accuracy_after != null ? (
                        <>
                          <p className="text-xs text-emerald-400 font-medium">
                            {((entry.accuracy_after - entry.accuracy_before) * 100) > 0 ? '+' : ''}
                            {((entry.accuracy_after - entry.accuracy_before) * 100).toFixed(1)}%
                          </p>
                          <p className="text-[10px] dark:text-gray-500 text-gray-400">
                            {(entry.accuracy_after * 100).toFixed(1)}%
                          </p>
                        </>
                      ) : entry.accuracy != null ? (
                        <p className="text-xs text-emerald-400 font-medium">{(entry.accuracy * 100).toFixed(1)}%</p>
                      ) : null}
                    </div>
                  </div>
                  {entry.timestamp && (
                    <p className="text-[10px] dark:text-gray-600 text-gray-400 mt-1 ml-8">
                      {new Date(entry.timestamp).toLocaleString()}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </motion.div>
      </div>

      {/* Quick Actions Row */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
      >
        <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          {[
            { label: 'Classify Complaint', icon: MessageSquareWarning, to: '/classify', color: 'from-cyan-500 to-blue-500' },
            { label: 'Batch Process', icon: Layers, to: '/batch', color: 'from-violet-500 to-purple-500' },
            { label: 'AI Assistant', icon: Bot, to: '/ai', color: 'from-pink-500 to-rose-500' },
            { label: 'Review Queue', icon: ClipboardList, to: '/review', color: 'from-amber-500 to-orange-500' },
          ].map((a, i) => (
            <NavLink
              key={i}
              to={a.to}
              className="flex items-center gap-3 p-3 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 border dark:border-white/5 border-gray-200 dark:hover:border-white/10 hover:border-gray-300 transition-all duration-200 group"
            >
              <div className={`w-9 h-9 rounded-lg bg-gradient-to-br ${a.color} flex items-center justify-center`}>
                <a.icon size={16} className="text-white" />
              </div>
              <span className="text-sm dark:text-gray-300 text-gray-600 group-hover:dark:text-white group-hover:text-gray-900 transition">{a.label}</span>
            </NavLink>
          ))}
          <button
            onClick={() => window.open(`${API}/export/report`, '_blank', 'noopener,noreferrer')}
            className="flex items-center gap-3 p-3 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 border dark:border-white/5 border-gray-200 dark:hover:border-white/10 hover:border-gray-300 transition-all duration-200 group"
          >
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
              <FileText size={16} className="text-white" />
            </div>
            <span className="text-sm dark:text-gray-300 text-gray-600 group-hover:dark:text-white group-hover:text-gray-900 transition">Export Report</span>
          </button>
        </div>
      </motion.div>

      {/* Smart Suggestions */}
      {suggestions && suggestions.suggestions && suggestions.suggestions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
        >
          <h3 className="text-sm font-medium text-gray-300 mb-3">Smart Suggestions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {suggestions.suggestions.slice(0, 4).map((s, i) => (
              <div key={i} className={`rounded-xl p-4 border transition-all hover:border-white/10 ${
                s.type === 'warning' ? 'bg-amber-500/5 border-amber-500/10' :
                s.type === 'action' ? 'bg-violet-500/5 border-violet-500/10' :
                'bg-[#131825] border-white/5'
              }`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="text-sm font-medium text-white">{s.title}</p>
                    <p className="text-xs text-gray-400 mt-1">{s.description}</p>
                  </div>
                  {s.action && (
                    <button
                      onClick={() => navigate(s.action)}
                      className="ml-3 px-3 py-1 text-xs bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors whitespace-nowrap"
                    >
                      {s.action_label}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}

// ============================================================================
// PAGE: CLASSIFY
// ============================================================================

function ClassifyPage() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState([])
  const [retrainStatus, setRetrainStatus] = useState(null)
  const { addToast } = useToast()
  const [samples] = useState([
    { text: 'Mera order abhi tak nahi aaya, 3 din ho gaye!', cat: 'Delivery_Issue', urg: 'High' },
    { text: 'Refund kab milega? Paisa wapas karo turant', cat: 'Returns_Refunds', urg: 'High' },
    { text: 'Wrong product bheja hai, exchange karo', cat: 'Wrong_Damaged_Product', urg: 'Medium' },
    { text: 'Payment fail ho gaya but paisa kat gaya', cat: 'Payment_Invoice', urg: 'High' },
    { text: 'App crash ho raha hai, login nahi ho raha', cat: 'Account_Technical', urg: 'Medium' },
    { text: 'Customer care bilkul useless hai, koi response nahi', cat: 'Customer_Service', urg: 'High' },
    { text: 'Product ka quality bahut ghatiya hai', cat: 'Product_Quality', urg: 'Medium' },
    { text: 'Checkout pe price badh gaya, hidden charges hai', cat: 'Pricing_Discount', urg: 'Medium' },
    { text: 'Delivery boy rude tha, manager se baat karo', cat: 'Delivery_Issue', urg: 'Medium' },
  ])

  const loadRecent = useCallback(() => {
    getHistory({ limit: 5 }).then(d => setHistory(d.predictions || [])).catch(() => {})
  }, [])

  useEffect(() => { loadRecent() }, [result, loadRecent])

  useEffect(() => {
    getRetrainStatus().then(setRetrainStatus).catch(() => {})
  }, [])

  async function handleClassify(complaintText) {
    const t = complaintText || text
    if (!t.trim()) return
    setLoading(true)
    try {
      const r = await predict(t)
      setResult(r)
      if (!complaintText) setText('')
    } catch { addToast('Classification failed. Is the backend running?', 'error') }
    setLoading(false)
  }

  return (
    <div className="space-y-6">
      {/* Retrain Banner */}
      {retrainStatus && retrainStatus.should_retrain && (
        <RetrainBanner
          correctionsTotal={retrainStatus.corrections_total}
          onRetrain={() => {
            loadRecent()
            getRetrainStatus().then(setRetrainStatus).catch(() => {})
          }}
        />
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Input Column */}
        <div className="lg:col-span-3 space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
          >
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-3">Enter Complaint</h3>
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder="Type a Hinglish complaint... e.g., Mera order abhi tak nahi aaya!"
              className="w-full h-32 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg px-4 py-3 text-sm dark:text-white text-gray-900 placeholder-gray-500 outline-none focus:border-cyan-500/50 resize-none transition"
              onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleClassify() }}
            />
            <div className="flex items-center justify-between mt-3">
              <span className="text-xs dark:text-gray-600 text-gray-400">Ctrl+Enter to classify</span>
              <button
                onClick={() => handleClassify()}
                disabled={loading || !text.trim()}
                className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white text-sm font-medium disabled:opacity-50 hover:shadow-lg hover:shadow-cyan-500/20 transition-all duration-300"
              >
                {loading ? <RefreshCw size={14} className="animate-spin" /> : <Send size={14} />}
                {loading ? 'Classifying...' : 'Classify'}
              </button>
            </div>
          </motion.div>

          {/* Result */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 20, height: 0 }}
                animate={{ opacity: 1, y: 0, height: 'auto' }}
                exit={{ opacity: 0, y: -10, height: 0 }}
                className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5 overflow-hidden"
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">Result</h3>
                    {result.source && (
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        result.source === 'muril'
                          ? 'bg-violet-500/10 text-violet-400 border border-violet-500/20'
                          : 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                      }`}>
                        {result.source === 'muril' ? 'MuRIL' : 'Sklearn'}
                      </span>
                    )}
                  </div>
                  {result.needs_review && (
                    <span className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
                      <AlertTriangle size={12} /> Needs Review
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="p-4 rounded-lg dark:bg-white/5 bg-gray-100">
                    <p className="text-xs dark:text-gray-500 text-gray-400 mb-1">Category</p>
                    <p className="text-lg font-bold dark:text-white text-gray-900">{CAT_SHORT[result.category] || result.category}</p>
                    <p className="text-xs dark:text-gray-400 text-gray-500 mt-1">{result.category}</p>
                    <div className="mt-2 h-1.5 dark:bg-white/5 bg-gray-200 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.category_confidence * 100}%` }}
                        className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500"
                      />
                    </div>
                    <p className="text-xs dark:text-gray-500 text-gray-400 mt-1">{(result.category_confidence * 100).toFixed(1)}% confidence</p>
                  </div>
                  <div className="p-4 rounded-lg dark:bg-white/5 bg-gray-100">
                    <p className="text-xs dark:text-gray-500 text-gray-400 mb-1">Urgency</p>
                    <p className="text-lg font-bold" style={{ color: URGENCY_COLORS[result.urgency] }}>{result.urgency}</p>
                    <div className="mt-2 h-1.5 dark:bg-white/5 bg-gray-200 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.urgency_confidence * 100}%` }}
                        className="h-full rounded-full"
                        style={{ background: URGENCY_COLORS[result.urgency] }}
                      />
                    </div>
                    <p className="text-xs dark:text-gray-500 text-gray-400 mt-1">{(result.urgency_confidence * 100).toFixed(1)}% confidence</p>
                  </div>
                </div>

                {/* Probability Bars */}
                <div className="space-y-2">
                  <p className="text-xs dark:text-gray-500 text-gray-400 uppercase tracking-wider">Category Probabilities</p>
                  {Object.entries(result.category_probabilities || {})
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([cat, prob], i) => (
                      <div key={cat} className="flex items-center gap-3">
                        <span className="text-xs dark:text-gray-400 text-gray-500 w-20 truncate">{CAT_SHORT[cat] || cat}</span>
                        <div className="flex-1 h-1.5 dark:bg-white/5 bg-gray-200 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${prob * 100}%` }}
                            transition={{ duration: 0.5, delay: i * 0.05 }}
                            className="h-full rounded-full"
                            style={{ background: COLORS[i] }}
                          />
                        </div>
                        <span className="text-xs dark:text-gray-500 text-gray-400 w-10 text-right">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>

                {/* Feedback Panel */}
                <FeedbackPanel
                  predictionId={result.id}
                  text={result.text || text}
                  predictedCategory={result.category}
                  predictedUrgency={result.urgency}
                  onFeedbackSubmitted={(res) => {
                    setRetrainStatus(prev => prev ? { ...prev, corrections_total: res.corrections_total, should_retrain: res.should_retrain } : prev)
                    loadRecent()
                  }}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right Column: Samples + Recent */}
        <div className="lg:col-span-2 space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
          >
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-3">Quick Samples</h3>
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {samples.map((s, i) => (
                <button
                  key={i}
                  onClick={() => { setText(s.text); handleClassify(s.text) }}
                  className="w-full text-left p-3 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 border dark:border-white/5 border-gray-200 dark:hover:border-white/10 hover:border-gray-300 transition-all duration-200"
                >
                  <p className="text-xs dark:text-gray-300 text-gray-600 line-clamp-2">{s.text}</p>
                  <div className="flex items-center gap-2 mt-1.5">
                    <span className="text-[10px] px-1.5 py-0.5 rounded dark:bg-white/10 bg-gray-200 dark:text-gray-400 text-gray-500">{CAT_SHORT[s.cat]}</span>
                    <span className="text-[10px]" style={{ color: URGENCY_COLORS[s.urg] }}>{s.urg}</span>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>

          {/* Recent Predictions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
          >
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-3">Recent</h3>
            {history.length === 0 ? (
              <p className="text-xs dark:text-gray-600 text-gray-400 text-center py-4">No predictions yet</p>
            ) : (
              <div className="space-y-2">
                {history.slice(0, 4).map((p, i) => (
                  <div key={i} className="p-2.5 rounded-lg dark:bg-white/5 bg-gray-100 border dark:border-white/5 border-gray-200">
                    <p className="text-xs dark:text-gray-300 text-gray-600 truncate">{p.text}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[10px] px-1.5 py-0.5 rounded dark:bg-white/10 bg-gray-200 dark:text-gray-400 text-gray-500">{CAT_SHORT[p.predicted_category] || p.predicted_category}</span>
                      <span className="text-[10px]" style={{ color: URGENCY_COLORS[p.predicted_urgency] }}>{p.predicted_urgency}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// PAGE: BATCH
// ============================================================================

function BatchPage() {
  const [file, setFile] = useState(null)
  const [csvData, setCsvData] = useState([])
  const [results, setResults] = useState([])
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [textInput, setTextInput] = useState('')
  const fileRef = useRef(null)
  const { addToast } = useToast()

  function handleFileUpload(e) {
    const f = e.target.files[0]
    if (!f) return
    if (f.size > 5 * 1024 * 1024) {
      addToast('File too large. Max 5MB.', 'error')
      return
    }
    setFile(f)
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target.result
      const result = Papa.parse(text, { header: true, skipEmptyLines: true })
      const rows = result.data
      if (rows.length === 0) {
        addToast('No data found in CSV', 'error')
        return
      }
      const textCol = Object.keys(rows[0]).find(k => k.toLowerCase().includes('text')) || Object.keys(rows[0])[0]
      const parsed = rows.map(r => r[textCol]?.trim()).filter(Boolean)
      setCsvData(parsed)
      addToast(`Loaded ${parsed.length} complaints from CSV`, 'success')
    }
    reader.readAsText(f)
  }

  function handleTextUpload() {
    const lines = textInput.split('\n').filter(l => l.trim())
    setCsvData(lines)
    setTextInput('')
  }

  async function processBatch() {
    if (csvData.length === 0) return
    setProcessing(true)
    setResults([])
    const batchSize = 10
    const allResults = []

    for (let i = 0; i < csvData.length; i += batchSize) {
      const batch = csvData.slice(i, i + batchSize)
      try {
        const res = await predictBatch(batch)
        allResults.push(...(res.predictions || []))
        setResults([...allResults])
        setProgress(Math.round(((i + batch.length) / csvData.length) * 100))
      } catch {
        addToast('Batch processing failed', 'error')
        break
      }
    }
    setProcessing(false)
  }

  function exportResults(fmt = 'csv') {
    if (results.length === 0) return

    if (fmt === 'csv') {
      const header = 'text,category,category_confidence,urgency,urgency_confidence\n'
      const rows = results.map(r =>
        `"${r.text}","${r.category}",${r.category_confidence},"${r.urgency}",${r.urgency_confidence}`
      ).join('\n')
      const blob = new Blob([header + rows], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = `batch_results_${Date.now()}.csv`; a.click()
      URL.revokeObjectURL(url)
    } else {
      const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = `batch_results_${Date.now()}.json`; a.click()
      URL.revokeObjectURL(url)
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-4">Upload Data</h3>

          {/* File Upload */}
          <div
            onClick={() => fileRef.current?.click()}
            className="border-2 border-dashed dark:border-white/10 border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-cyan-500/30 hover:bg-white/[0.02] transition-all duration-300"
          >
            <input ref={fileRef} type="file" accept=".csv,.txt" onChange={handleFileUpload} className="hidden" />
            <Upload size={24} className="mx-auto dark:text-gray-600 text-gray-400 mb-2" />
            <p className="text-sm dark:text-gray-400 text-gray-500">Drop CSV file or click to upload</p>
            <p className="text-xs dark:text-gray-600 text-gray-400 mt-1">One complaint per line or first column of CSV</p>
          </div>

          {csvData.length > 0 && (
            <div className="mt-4 p-3 rounded-lg dark:bg-white/5 bg-gray-100">
              <p className="text-xs dark:text-gray-400 text-gray-500">{csvData.length} complaints loaded</p>
              <p className="text-xs dark:text-gray-600 text-gray-400 mt-1 truncate">{csvData[0]}...</p>
            </div>
          )}

          {/* Or paste text */}
          <div className="mt-4">
            <p className="text-xs dark:text-gray-500 text-gray-400 mb-2">Or paste complaints (one per line):</p>
            <textarea
              value={textInput}
              onChange={e => setTextInput(e.target.value)}
              placeholder={"Mera order nahi aaya\nRefund do\nWrong product aaya"}
              className="w-full h-24 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg px-3 py-2 text-sm dark:text-white text-gray-900 placeholder-gray-500 outline-none focus:border-cyan-500/50 resize-none transition"
            />
            <button
              onClick={handleTextUpload}
              disabled={!textInput.trim()}
              className="mt-2 px-4 py-1.5 rounded-lg dark:bg-white/10 bg-gray-200 text-sm dark:text-gray-300 text-gray-600 dark:hover:bg-white/15 hover:bg-gray-300 transition disabled:opacity-50"
            >
              Load Text
            </button>
          </div>

          {/* Process */}
          <button
            onClick={processBatch}
            disabled={processing || csvData.length === 0}
            className="w-full mt-4 flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white text-sm font-medium disabled:opacity-50 hover:shadow-lg hover:shadow-cyan-500/20 transition-all duration-300"
          >
            {processing ? <RefreshCw size={14} className="animate-spin" /> : <Zap size={14} />}
            {processing ? `Processing... ${progress}%` : `Classify ${csvData.length} Complaints`}
          </button>

          {processing && (
            <div className="mt-3 h-1.5 dark:bg-white/5 bg-gray-200 rounded-full overflow-hidden">
              <motion.div
                animate={{ width: `${progress}%` }}
                className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500"
              />
            </div>
          )}
        </motion.div>

        {/* Results */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">
              Results {results.length > 0 && <span className="dark:text-gray-500 text-gray-400">({results.length})</span>}
            </h3>
            {results.length > 0 && (
              <div className="flex items-center gap-2">
                <button onClick={() => exportResults('csv')} className="flex items-center gap-1 px-3 py-1.5 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 text-xs dark:text-gray-400 text-gray-500 transition">
                  <Download size={12} /> CSV
                </button>
                <button onClick={() => exportResults('json')} className="flex items-center gap-1 px-3 py-1.5 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 text-xs dark:text-gray-400 text-gray-500 transition">
                  <Download size={12} /> JSON
                </button>
              </div>
            )}
          </div>

          {results.length === 0 ? (
            <div className="h-64 flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">
              Upload data and click Classify
            </div>
          ) : (
            <div className="space-y-2 max-h-[500px] overflow-y-auto">
              {results.map((r, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.02 }}
                  className="p-3 rounded-lg dark:bg-white/5 bg-gray-100 border dark:border-white/5 border-gray-200"
                >
                  <p className="text-xs dark:text-gray-300 text-gray-600 truncate">{r.text}</p>
                  <div className="flex items-center gap-3 mt-1.5">
                    <span className="text-[10px] px-1.5 py-0.5 rounded dark:bg-white/10 bg-gray-200 dark:text-gray-400 text-gray-500">{CAT_SHORT[r.category] || r.category}</span>
                    <span className="text-[10px]" style={{ color: URGENCY_COLORS[r.urgency] }}>{r.urgency}</span>
                    <span className="text-[10px] dark:text-gray-600 text-gray-400">{(r.category_confidence * 100).toFixed(0)}%</span>
                    {r.needs_review && <AlertTriangle size={10} className="text-amber-400" />}
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}

// ============================================================================
// PAGE: HISTORY (with correction status + corrected-only filter)
// ============================================================================

function HistoryPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [data, setData] = useState({ predictions: [], total: 0 })
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const [filter, setFilter] = useState({
    category: searchParams.get('category') || '',
    urgency: searchParams.get('urgency') || '',
    search: searchParams.get('search') || '',
    correctedOnly: searchParams.get('corrected_only') === 'true',
  })
  const limit = 20

  const loadHistory = useCallback(async () => {
    setLoading(true)
    try {
      const params = { limit, offset: page * limit }
      if (filter.category) params.category = filter.category
      if (filter.urgency) params.urgency = filter.urgency
      if (filter.search) params.search = filter.search
      if (filter.correctedOnly) params.corrected_only = 'true'
      const d = await getHistory(params)
      setData(d)
    } catch {}
    setLoading(false)
  }, [page, filter])

  useEffect(() => { loadHistory() }, [loadHistory])

  function exportAll() {
    const params = new URLSearchParams()
    if (filter.category) params.set('category', filter.category)
    window.open(`${API}/export/csv?${params.toString()}`, '_blank', 'noopener,noreferrer')
  }

  function updateFilter(updates) {
    setFilter(f => {
      const newFilter = { ...f, ...updates }
      const params = new URLSearchParams()
      if (newFilter.category) params.set('category', newFilter.category)
      if (newFilter.urgency) params.set('urgency', newFilter.urgency)
      if (newFilter.search) params.set('search', newFilter.search)
      if (newFilter.correctedOnly) params.set('corrected_only', 'true')
      setSearchParams(params, { replace: true })
      setPage(0)
      return newFilter
    })
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-4"
      >
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <Filter size={14} className="dark:text-gray-500 text-gray-400" />
            <span className="text-xs dark:text-gray-500 text-gray-400">Filters:</span>
          </div>
          <input
            value={filter.search}
            onChange={e => updateFilter({ search: e.target.value })}
            placeholder="Search text..."
            aria-label="Search predictions"
            className="px-3 py-1.5 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg text-xs dark:text-white text-gray-900 placeholder-gray-500 focus:border-cyan-500/50 w-48"
          />
          <select
            value={filter.category}
            onChange={e => updateFilter({ category: e.target.value })}
            aria-label="Filter by category"
            className="px-3 py-1.5 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg text-xs dark:text-white text-gray-900"
          >
            <option value="">All Categories</option>
            {CATEGORIES.map(c => <option key={c} value={c}>{CAT_SHORT[c]}</option>)}
          </select>
          <select
            value={filter.urgency}
            onChange={e => updateFilter({ urgency: e.target.value })}
            aria-label="Filter by urgency"
            className="px-3 py-1.5 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg text-xs dark:text-white text-gray-900"
          >
            <option value="">All Urgency</option>
            <option value="High">High</option>
            <option value="Medium">Medium</option>
            <option value="Low">Low</option>
          </select>
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={filter.correctedOnly}
              onChange={e => updateFilter({ correctedOnly: e.target.checked })}
              className="w-3.5 h-3.5 rounded border-white/20 bg-white/5 text-cyan-500 focus:ring-cyan-500/50 focus:ring-offset-0"
            />
            <span className="text-xs dark:text-gray-400 text-gray-500">Corrected only</span>
          </label>
          <div className="flex-1" />
          <button onClick={exportAll} className="flex items-center gap-1 px-3 py-1.5 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 text-xs dark:text-gray-400 text-gray-500 transition">
            <Download size={12} /> Export CSV
          </button>
        </div>
      </motion.div>

      {/* Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl overflow-hidden"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">Prediction History</caption>
            <thead>
              <tr className="border-b dark:border-white/5 border-gray-200">
                <th scope="col" className="text-left px-4 py-3 text-xs dark:text-gray-500 text-gray-400 font-medium">Text</th>
                <th scope="col" className="text-left px-4 py-3 text-xs dark:text-gray-500 text-gray-400 font-medium">Category</th>
                <th scope="col" className="text-left px-4 py-3 text-xs dark:text-gray-500 text-gray-400 font-medium">Urgency</th>
                <th scope="col" className="text-left px-4 py-3 text-xs dark:text-gray-500 text-gray-400 font-medium">Confidence</th>
                <th scope="col" className="text-left px-4 py-3 text-xs dark:text-gray-500 text-gray-400 font-medium">Status</th>
                <th scope="col" className="text-left px-4 py-3 text-xs dark:text-gray-500 text-gray-400 font-medium">Time</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={6} className="px-4 py-8"><TableSkeleton rows={5} cols={6} /></td></tr>
              ) : data.predictions.length === 0 ? (
                <tr><td colSpan={6} className="px-4 py-8 text-center dark:text-gray-600 text-gray-400">No predictions found</td></tr>
              ) : (
                data.predictions.map((p, i) => {
                  const isCorrected = p.is_corrected || p.was_corrected || (p.corrections && p.corrections.length > 0)
                  return (
                    <tr key={i} className="border-b dark:border-white/5 border-gray-200 dark:hover:bg-white/[0.02] hover:bg-gray-50 transition">
                      <td className="px-4 py-3 text-xs dark:text-gray-300 text-gray-600 max-w-xs truncate">{p.text}</td>
                      <td className="px-4 py-3">
                        <span className="text-[10px] px-2 py-1 rounded-full dark:bg-white/10 bg-gray-100 dark:text-gray-400 text-gray-500">{CAT_SHORT[p.predicted_category] || p.predicted_category}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-xs font-medium" style={{ color: URGENCY_COLORS[p.predicted_urgency] }}>{p.predicted_urgency}</span>
                      </td>
                      <td className="px-4 py-3 text-xs dark:text-gray-400 text-gray-500">{(p.confidence_category * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3">
                        {isCorrected ? (
                          <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-red-500/10 text-red-400 border border-red-500/20">
                            <SquareX size={10} />
                            Corrected
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            <Check size={10} />
                            Correct
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-xs dark:text-gray-600 text-gray-400">{p.timestamp ? new Date(p.timestamp).toLocaleString() : '-'}</td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {data.total > limit && (
          <div className="flex items-center justify-between px-4 py-3 border-t dark:border-white/5 border-gray-200">
            <span className="text-xs dark:text-gray-600 text-gray-400">Showing {page * limit + 1}-{Math.min((page + 1) * limit, data.total)} of {data.total}</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-3 py-1 rounded dark:bg-white/5 bg-gray-100 text-xs dark:text-gray-400 text-gray-500 dark:hover:bg-white/10 hover:bg-gray-200 disabled:opacity-30 transition"
              >Prev</button>
              <button
                onClick={() => setPage(p => p + 1)}
                disabled={(page + 1) * limit >= data.total}
                className="px-3 py-1 rounded dark:bg-white/5 bg-gray-100 text-xs dark:text-gray-400 text-gray-500 dark:hover:bg-white/10 hover:bg-gray-200 disabled:opacity-30 transition"
              >Next</button>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  )
}

// ============================================================================
// PAGE: ANALYTICS
// ============================================================================

function AnalyticsPage() {
  const [confData, setConfData] = useState(null)
  const [wordData, setWordData] = useState(null)
  const [patterns, setPatterns] = useState(null)
  const [timeline, setTimeline] = useState(null)
  const [loading, setLoading] = useState(true)
  const [wordCat, setWordCat] = useState('')
  const [timeRange, setTimeRange] = useState(24)

  useEffect(() => {
    Promise.all([getConfidence(), getWordFrequency(), getPatterns(), getTimeline(timeRange)])
      .then(([c, w, p, t]) => { setConfData(c); setWordData(w); setPatterns(p); setTimeline(t) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    getWordFrequency(wordCat || undefined).then(setWordData).catch(() => {})
  }, [wordCat])

  useEffect(() => {
    getTimeline(timeRange).then(setTimeline).catch(() => {})
  }, [timeRange])

  if (loading) return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {Array.from({ length: 3 }).map((_, i) => <ChartSkeleton key={i} />)}
      </div>
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Confidence Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="lg:col-span-2 dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">Confidence Distribution</h3>
            <span className="text-xs dark:text-gray-500 text-gray-400">Avg: {((confData?.overall_avg || 0) * 100).toFixed(1)}%</span>
          </div>
          {confData?.distribution?.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={confData.distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="range" tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} />
                <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 12 }} />
                <Bar dataKey="count" fill="#22d3ee" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">No data yet</div>
          )}
        </motion.div>

        {/* Category Avg Confidence */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-4">Avg Confidence by Category</h3>
          {confData?.category_avg?.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={confData.category_avg} layout="vertical" barSize={16}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" domain={[0, 1]} tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} />
                <YAxis type="category" dataKey="category" tick={{ fill: '#6b7280', fontSize: 9 }} width={80} axisLine={false} />
                <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 12 }} />
                <Bar dataKey="avg_confidence" radius={[0, 4, 4, 0]}>
                  {confData.category_avg.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">No data yet</div>
          )}
        </motion.div>
      </div>

      {/* Word Frequency + Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">Top Words</h3>
            <select
              value={wordCat}
              onChange={e => setWordCat(e.target.value)}
              className="px-2 py-1 dark:bg-white/5 bg-gray-100 dark:border-white/10 border-gray-200 rounded text-xs dark:text-white text-gray-900 outline-none"
            >
              <option value="">All</option>
              {CATEGORIES.map(c => <option key={c} value={c}>{CAT_SHORT[c]}</option>)}
            </select>
          </div>
          {wordData?.words?.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={wordData.words.slice(0, 12)} barSize={20}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="word" tick={{ fill: '#6b7280', fontSize: 9 }} axisLine={false} angle={-45} textAnchor="end" height={60} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} />
                <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 12 }} />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">No word data yet</div>
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">Prediction Timeline</h3>
            <select
              value={timeRange}
              onChange={e => setTimeRange(Number(e.target.value))}
              className="px-2 py-1 dark:bg-white/5 bg-gray-100 dark:border-white/10 border-gray-200 rounded text-xs dark:text-white text-gray-900 outline-none"
            >
              <option value={6}>6 hours</option>
              <option value={24}>24 hours</option>
              <option value={72}>3 days</option>
              <option value={168}>7 days</option>
            </select>
          </div>
          {timeline?.timeline?.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={timeline.timeline.map(t => ({
                time: t.hour?.split(' ')[1] || t.hour,
                count: t.count,
                confidence: Math.round((t.avg_confidence || 0) * 100),
              }))}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="time" tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} />
                <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 12 }} />
                <Area type="monotone" dataKey="count" stroke="#6366f1" fill="url(#colorCount)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">No timeline data yet</div>
          )}
        </motion.div>
      </div>

      {/* Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
      >
        <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-4">Insights</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { label: 'Review Rate', value: `${((patterns?.review_rate || 0) * 100).toFixed(1)}%`, sub: `${patterns?.needs_review_count || 0} flagged`, icon: Eye, color: 'text-amber-400' },
            { label: 'Correction Rate', value: `${((patterns?.correction_rate || 0) * 100).toFixed(1)}%`, sub: `${patterns?.corrections_count || 0} corrections`, icon: RefreshCw, color: 'text-violet-400' },
            { label: 'Avg Text Length', value: `${patterns?.avg_text_length || 0}`, sub: 'characters', icon: FileText, color: 'text-cyan-400' },
            { label: 'Total Predictions', value: patterns?.total_predictions || 0, sub: 'all time', icon: Activity, color: 'text-emerald-400' },
          ].map((item, i) => (
            <div key={i} className="p-4 rounded-lg dark:bg-white/5 bg-gray-100">
              <item.icon size={16} className={item.color} />
              <p className="text-xl font-bold dark:text-white text-gray-900 mt-2">{item.value}</p>
              <p className="text-xs dark:text-gray-500 text-gray-400">{item.sub}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}

// ============================================================================
// PAGE: AI ASSISTANT
// ============================================================================

function AIAssistantPage() {
  const { addToast } = useToast()
  const [text, setText] = useState('')
  const [category, setCategory] = useState('')
  const [urgency, setUrgency] = useState('')
  const [resolveResult, setResolveResult] = useState(null)
  const [draftResult, setDraftResult] = useState(null)
  const [resolving, setResolving] = useState(false)
  const [drafting, setDrafting] = useState(false)

  async function handleResolve() {
    if (!text.trim()) return
    setResolving(true)
    setResolveResult(null)
    try {
      const res = await aiResolve(text, category || undefined, urgency || undefined)
      setResolveResult(res)
      addToast('Resolution steps generated', 'success')
    } catch {
      addToast('Failed to generate resolution steps', 'error')
    }
    setResolving(false)
  }

  async function handleDraft() {
    if (!text.trim()) return
    setDrafting(true)
    setDraftResult(null)
    try {
      const res = await aiDraftResponse(text, category || undefined, urgency || undefined)
      setDraftResult(res)
      addToast('Response draft generated', 'success')
    } catch {
      addToast('Failed to generate response draft', 'error')
    }
    setDrafting(false)
  }

  function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
      addToast('Copied to clipboard', 'success')
    }).catch(() => {
      addToast('Failed to copy', 'error')
    })
  }

  return (
    <div className="space-y-6">
      {/* Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-pink-500 to-rose-500 flex items-center justify-center">
            <Bot size={16} className="text-white" />
          </div>
          <div>
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">AI Complaint Assistant</h3>
            <p className="text-[10px] dark:text-gray-500 text-gray-400">Powered by Groq</p>
          </div>
        </div>

        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste a customer complaint to get resolution steps or draft a response..."
          className="w-full h-32 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg px-4 py-3 text-sm dark:text-white text-gray-900 placeholder-gray-500 outline-none focus:border-pink-500/50 resize-none transition"
        />

        <div className="grid grid-cols-2 gap-3 mt-3">
          <div>
            <label className="text-[10px] dark:text-gray-500 text-gray-400 uppercase tracking-wider block mb-1">Category (optional)</label>
            <select
              value={category}
              onChange={e => setCategory(e.target.value)}
              className="w-full px-3 py-1.5 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg text-xs dark:text-white text-gray-900 outline-none focus:border-pink-500/50"
            >
              <option value="">Auto-detect</option>
              {CATEGORIES.map(c => (
                <option key={c} value={c}>{CAT_SHORT[c]}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[10px] dark:text-gray-500 text-gray-400 uppercase tracking-wider block mb-1">Urgency (optional)</label>
            <select
              value={urgency}
              onChange={e => setUrgency(e.target.value)}
              className="w-full px-3 py-1.5 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg text-xs dark:text-white text-gray-900 outline-none focus:border-pink-500/50"
            >
              <option value="">Auto-detect</option>
              <option value="High">High</option>
              <option value="Medium">Medium</option>
              <option value="Low">Low</option>
            </select>
          </div>
        </div>

        <div className="flex items-center gap-3 mt-4">
          <button
            onClick={handleResolve}
            disabled={resolving || !text.trim()}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-pink-500 to-rose-500 text-white text-sm font-medium disabled:opacity-50 hover:shadow-lg hover:shadow-pink-500/20 transition-all duration-300"
          >
            {resolving ? <Loader2 size={14} className="animate-spin" /> : <CornerDownLeft size={14} />}
            {resolving ? 'Generating...' : 'Get Resolution Steps'}
          </button>
          <button
            onClick={handleDraft}
            disabled={drafting || !text.trim()}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-violet-500 to-purple-500 text-white text-sm font-medium disabled:opacity-50 hover:shadow-lg hover:shadow-violet-500/20 transition-all duration-300"
          >
            {drafting ? <Loader2 size={14} className="animate-spin" /> : <FileText size={14} />}
            {drafting ? 'Generating...' : 'Draft Response'}
          </button>
        </div>
      </motion.div>

      {/* Results */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Resolution Steps */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <CornerDownLeft size={14} className="text-pink-400" />
              <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">Resolution Steps</h3>
            </div>
            {resolveResult?.suggestions && (
              <button
                onClick={() => copyToClipboard(resolveResult.suggestions)}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 text-xs dark:text-gray-400 text-gray-500 transition"
              >
                <Copy size={12} /> Copy
              </button>
            )}
          </div>

          {resolving && (
            <div className="flex items-center justify-center h-48">
              <Loader2 size={24} className="text-pink-400 animate-spin" />
            </div>
          )}

          {!resolving && resolveResult?.suggestions && (
            <div className="p-4 rounded-lg dark:bg-white/5 bg-gray-100 border dark:border-white/10 border-gray-200">
              <p className="text-sm dark:text-gray-300 text-gray-600 whitespace-pre-wrap leading-relaxed">
                {resolveResult.suggestions}
              </p>
              {resolveResult.model && (
                <p className="text-[10px] dark:text-gray-600 text-gray-400 mt-3 pt-2 border-t dark:border-white/5 border-gray-200">
                  Model: {resolveResult.model}
                </p>
              )}
            </div>
          )}

          {!resolving && !resolveResult && (
            <div className="h-48 flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">
              Enter a complaint and click "Get Resolution Steps"
            </div>
          )}
        </motion.div>

        {/* Draft Response */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <FileText size={14} className="text-violet-400" />
              <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">Draft Response</h3>
            </div>
            {draftResult?.draft && (
              <button
                onClick={() => copyToClipboard(draftResult.draft)}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 text-xs dark:text-gray-400 text-gray-500 transition"
              >
                <Copy size={12} /> Copy
              </button>
            )}
          </div>

          {drafting && (
            <div className="flex items-center justify-center h-48">
              <Loader2 size={24} className="text-violet-400 animate-spin" />
            </div>
          )}

          {!drafting && draftResult?.draft && (
            <div className="p-4 rounded-lg dark:bg-white/5 bg-gray-100 border dark:border-white/10 border-gray-200">
              <p className="text-sm dark:text-gray-300 text-gray-600 whitespace-pre-wrap leading-relaxed">
                {draftResult.draft}
              </p>
              {draftResult.model && (
                <p className="text-[10px] dark:text-gray-600 text-gray-400 mt-3 pt-2 border-t dark:border-white/5 border-gray-200">
                  Model: {draftResult.model}
                </p>
              )}
            </div>
          )}

          {!drafting && !draftResult && (
            <div className="h-48 flex items-center justify-center dark:text-gray-600 text-gray-400 text-sm">
              Enter a complaint and click "Draft Response"
            </div>
          )}
        </motion.div>
      </div>

      {/* Powered by Groq Badge */}
      <div className="flex justify-center">
        <span className="text-[10px] dark:text-gray-600 text-gray-400 flex items-center gap-1.5">
          <Sparkles size={10} />
          Powered by Groq AI
        </span>
      </div>
    </div>
  )
}

// ============================================================================
// PAGE: REVIEW QUEUE
// ============================================================================

function ReviewQueuePage() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [count, setCount] = useState(0)

  useEffect(() => {
    loadQueue()
  }, [])

  function loadQueue() {
    setLoading(true)
    getLowConfidence()
      .then(d => { setItems(d.predictions || []); setCount(d.count || 0) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-4"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
              <ClipboardList size={16} className="text-white" />
            </div>
            <div>
              <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600">Review Queue</h3>
              <p className="text-[10px] dark:text-gray-500 text-gray-400">Low confidence predictions needing manual review</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs dark:text-gray-500 text-gray-400">{count} items</span>
            <button
              onClick={loadQueue}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg dark:bg-white/5 bg-gray-100 dark:hover:bg-white/10 hover:bg-gray-200 text-xs dark:text-gray-400 text-gray-500 transition"
            >
              <RefreshCw size={12} /> Refresh
            </button>
          </div>
        </div>
      </motion.div>

      {/* Items */}
      {loading ? (
        <TableSkeleton rows={3} cols={4} />
      ) : items.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-12"
        >
          <div className="text-center">
            <CheckCircle2 size={40} className="mx-auto text-emerald-400 mb-3" />
            <h3 className="text-sm font-medium dark:text-gray-300 text-gray-600 mb-1">All Clear!</h3>
            <p className="text-xs dark:text-gray-500 text-gray-400">No low-confidence predictions to review.</p>
          </div>
        </motion.div>
      ) : (
        <div className="space-y-3">
          {items.map((item, i) => (
            <motion.div
              key={item.id || i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-5"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <AlertTriangle size={14} className="text-amber-400" />
                  <span className="text-xs text-amber-400 font-medium">Low Confidence</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] px-2 py-0.5 rounded dark:bg-white/10 bg-gray-100 dark:text-gray-400 text-gray-500">
                    {CAT_SHORT[item.predicted_category] || item.predicted_category}
                  </span>
                  <span className="text-[10px]" style={{ color: URGENCY_COLORS[item.predicted_urgency] }}>
                    {item.predicted_urgency}
                  </span>
                  <span className="text-[10px] dark:text-gray-500 text-gray-400">
                    {((item.confidence_category || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <p className="text-sm dark:text-gray-300 text-gray-600 mb-3">{item.text}</p>

              <FeedbackPanel
                predictionId={item.id}
                text={item.text}
                predictedCategory={item.predicted_category}
                predictedUrgency={item.predicted_urgency}
                onFeedbackSubmitted={() => loadQueue()}
              />
            </motion.div>
          ))}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// PAGE: API PLAYGROUND
// ============================================================================

function PlaygroundPage() {
  const [endpoint, setEndpoint] = useState('/predict')
  const [method, setMethod] = useState('POST')
  const [body, setBody] = useState('{\n  "text": "Mera order nahi aaya"\n}')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [responseTime, setResponseTime] = useState(0)

  const endpoints = [
    { path: '/predict', method: 'POST', label: 'Classify Complaint', body: '{"text": "Mera order nahi aaya"}' },
    { path: '/predict/batch', method: 'POST', label: 'Batch Classify', body: '{"texts": ["Mera order nahi aaya", "Refund do"]}' },
    { path: '/history', method: 'GET', label: 'Get History' },
    { path: '/stats', method: 'GET', label: 'Get Stats' },
    { path: '/analytics/timeline', method: 'GET', label: 'Timeline' },
    { path: '/analytics/word-frequency', method: 'GET', label: 'Word Frequency' },
    { path: '/analytics/confidence', method: 'GET', label: 'Confidence Dist' },
    { path: '/analytics/patterns', method: 'GET', label: 'Patterns' },
    { path: '/categories', method: 'GET', label: 'Categories' },
    { path: '/feedback', method: 'POST', label: 'Submit Feedback', body: '{"prediction_id": 1, "is_correct_category": true, "is_correct_urgency": true}' },
    { path: '/retrain', method: 'POST', label: 'Trigger Retrain' },
    { path: '/retrain/status', method: 'GET', label: 'Retrain Status' },
    { path: '/retrain/history', method: 'GET', label: 'Retrain History' },
    { path: '/low-confidence', method: 'GET', label: 'Low Confidence' },
    { path: '/ai/resolve', method: 'POST', label: 'AI Resolve', body: '{"text": "Mera order nahi aaya"}' },
    { path: '/ai/draft-response', method: 'POST', label: 'AI Draft Response', body: '{"text": "Mera order nahi aaya"}' },
    { path: '/health', method: 'GET', label: 'Health Check' },
  ]

  async function sendRequest() {
    setLoading(true)
    const start = performance.now()
    try {
      let res
      if (method === 'GET') {
        res = await fetch(`${API}${endpoint}`)
      } else {
        res = await fetch(`${API}${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: body || undefined,
        })
      }
      const data = await res.json()
      setResponse({ status: res.status, data, time: Math.round(performance.now() - start) })
    } catch (e) {
      setResponse({ status: 0, data: { error: e.message }, time: Math.round(performance.now() - start) })
    }
    setLoading(false)
  }

  function selectEndpoint(ep) {
    setEndpoint(ep.path)
    setMethod(ep.method)
    if (ep.body) setBody(ep.body)
    setResponse(null)
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Endpoint List */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-4"
        >
          <h3 className="text-xs dark:text-gray-500 text-gray-400 uppercase tracking-wider mb-3">Endpoints</h3>
          <div className="space-y-1 max-h-[600px] overflow-y-auto">
            {endpoints.map((ep, i) => (
              <button
                key={i}
                onClick={() => selectEndpoint(ep)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs transition ${
                  endpoint === ep.path ? 'dark:bg-white/10 bg-gray-100 dark:text-white text-gray-900' : 'dark:text-gray-400 text-gray-500 dark:hover:bg-white/5 hover:bg-gray-100 dark:hover:text-white hover:text-gray-900'
                }`}
              >
                <span className={`font-mono ${ep.method === 'GET' ? 'text-emerald-400' : 'text-amber-400'}`}>{ep.method}</span>
                <span className="ml-2">{ep.label}</span>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Request/Response */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-3 space-y-4"
        >
          {/* Request */}
          <div className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-4">
            <div className="flex items-center gap-3 mb-3">
              <span className={`px-2 py-1 rounded text-xs font-mono ${method === 'GET' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'}`}>{method}</span>
              <span className="text-sm dark:text-gray-300 text-gray-600 font-mono">{API}{endpoint}</span>
              <div className="flex-1" />
              <button
                onClick={sendRequest}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white text-sm disabled:opacity-50 transition"
              >
                {loading ? <RefreshCw size={12} className="animate-spin" /> : <Send size={12} />}
                Send
              </button>
            </div>
            {method === 'POST' && (
              <textarea
                value={body}
                onChange={e => setBody(e.target.value)}
                className="w-full h-32 dark:bg-white/5 bg-gray-50 dark:border-white/10 border-gray-200 rounded-lg px-3 py-2 text-xs dark:text-gray-300 text-gray-600 font-mono outline-none focus:border-cyan-500/50 resize-none"
                placeholder="Request body (JSON)"
              />
            )}
          </div>

          {/* Response */}
          {response && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="dark:bg-[#131825] bg-white border dark:border-white/5 border-gray-200 rounded-xl p-4"
            >
              <div className="flex items-center gap-3 mb-3">
                <span className={`px-2 py-1 rounded text-xs font-mono ${response.status === 200 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                  {response.status || 'ERR'}
                </span>
                <span className="text-xs dark:text-gray-500 text-gray-400">{response.time}ms</span>
              </div>
              <pre className="dark:bg-white/5 bg-gray-50 rounded-lg p-3 text-xs dark:text-gray-300 text-gray-600 overflow-x-auto max-h-[400px] overflow-y-auto font-mono">
                {JSON.stringify(response.data, null, 2)}
              </pre>
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  )
}

// ============================================================================
// PAGE: MODEL COMPARISON
// ============================================================================

function ComparePage() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const { theme } = useTheme()
  const { addToast } = useToast()

  const handleCompare = async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const data = await apiFetch('/predict/compare', { method: 'POST', body: JSON.stringify({ text }) })
      setResult(data)
    } catch (err) {
      addToast('Comparison failed: ' + err.message, 'error')
    }
    setLoading(false)
  }

  const modelNames = {
    tfidf_svm: { label: 'TF-IDF + SVM', color: 'from-violet-500 to-purple-500' },
    tfidf_lr: { label: 'TF-IDF + LR', color: 'from-cyan-500 to-blue-500' },
    ensemble: { label: 'Combined Ensemble', color: 'from-emerald-500 to-teal-500' },
    muril: { label: 'MuRIL (GPU)', color: 'from-purple-500 to-pink-500' },
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
          <Layers size={20} className="text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-white">Model Comparison</h1>
          <p className="text-sm text-gray-500">Compare predictions across all 4 models side-by-side</p>
        </div>
      </div>

      <div className="bg-[#131825] border border-white/5 rounded-xl p-5">
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Enter a complaint to compare model predictions..."
          className={`w-full h-24 p-4 rounded-lg border resize-none focus:outline-none focus:ring-2 focus:ring-violet-500/50 ${
            theme === 'dark' ? 'bg-white/5 border-white/10 text-white placeholder-gray-500' : 'bg-gray-50 border-gray-200 text-gray-900 placeholder-gray-400'
          }`}
          onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleCompare() }}
        />
        <div className="mt-3 flex items-center justify-between">
          <span className="text-xs text-gray-500">Ctrl+Enter to compare</span>
          <button
            onClick={handleCompare}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-500 disabled:opacity-50 transition-colors"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Layers size={16} />}
            {loading ? 'Comparing...' : 'Compare Models'}
          </button>
        </div>
      </div>

      {result && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
          <div className={`rounded-xl p-4 border ${
            result.consensus.category_agreement && result.consensus.urgency_agreement
              ? 'bg-emerald-500/10 border-emerald-500/20'
              : 'bg-amber-500/10 border-amber-500/20'
          }`}>
            <div className="flex items-center gap-2">
              {result.consensus.category_agreement && result.consensus.urgency_agreement ? (
                <CheckCircle2 size={20} className="text-emerald-400" />
              ) : (
                <AlertTriangle size={20} className="text-amber-400" />
              )}
              <div>
                <p className={`font-medium ${result.consensus.category_agreement && result.consensus.urgency_agreement ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {result.consensus.category_agreement && result.consensus.urgency_agreement ? 'All Models Agree' : 'Models Disagree'}
                </p>
                <p className="text-xs text-gray-400">
                  Inference time: {result.inference_time_ms}ms
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
            {Object.entries(result.models).map(([key, model]) => {
              const info = modelNames[key] || { label: key, color: 'from-gray-500 to-gray-600' }
              if (model.error) return (
                <div key={key} className="bg-[#131825] border border-red-500/20 rounded-xl p-5">
                  <p className="text-red-400 font-medium">{info.label}</p>
                  <p className="text-sm text-gray-500 mt-1">{model.error}</p>
                </div>
              )
              return (
                <div key={key} className="bg-[#131825] border border-white/5 rounded-xl p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <div className={`w-3 h-3 rounded-full bg-gradient-to-r ${info.color}`} />
                    <p className="text-sm font-medium text-white">{info.label}</p>
                  </div>
                  <div className="space-y-2">
                    <div>
                      <p className="text-xs text-gray-500">Category</p>
                      <p className="text-lg font-bold text-white">{model.category?.replace(/_/g, ' ')}</p>
                      <div className="mt-1 h-2 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-violet-500 rounded-full" style={{ width: `${(model.category_confidence || 0) * 100}%` }} />
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5">{(model.category_confidence * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Urgency</p>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                          model.urgency === 'High' ? 'bg-red-500/20 text-red-400' :
                          model.urgency === 'Medium' ? 'bg-amber-500/20 text-amber-400' :
                          'bg-cyan-500/20 text-cyan-400'
                        }`}>{model.urgency}</span>
                        <span className="text-xs text-gray-500">{(model.urgency_confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                  {model.category_probabilities && Object.keys(model.category_probabilities).length > 0 && (
                    <div className="mt-3 pt-3 border-t border-white/5">
                      <p className="text-xs text-gray-500 mb-1">Top categories</p>
                      {Object.entries(model.category_probabilities)
                        .sort(([,a],[,b]) => b - a)
                        .slice(0, 3)
                        .map(([cat, prob]) => (
                          <div key={cat} className="flex items-center justify-between text-xs py-0.5">
                            <span className="text-gray-400">{cat.replace(/_/g, ' ')}</span>
                            <span className="text-gray-500">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))
                      }
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </motion.div>
      )}
    </div>
  )
}

// ============================================================================
// PAGE: SETTINGS
// ============================================================================

function SettingsPage() {
  const { branding, setBranding } = useBranding()
  const [name, setName] = useState(branding.companyName)
  const [color, setColor] = useState(branding.accentColor)
  const { addToast } = useToast()
  const { theme } = useTheme()

  const handleSave = () => {
    setBranding({ companyName: name, accentColor: color })
    addToast('Settings saved!', 'success')
  }

  const colorOptions = [
    { name: 'Indigo', value: '#6366f1' },
    { name: 'Violet', value: '#8b5cf6' },
    { name: 'Cyan', value: '#06b6d4' },
    { name: 'Emerald', value: '#10b981' },
    { name: 'Rose', value: '#f43f5e' },
    { name: 'Amber', value: '#f59e0b' },
  ]

  return (
    <div className="max-w-2xl space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-gray-500 to-gray-600 flex items-center justify-center">
          <Settings size={20} className="text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-white">Settings</h1>
          <p className="text-sm text-gray-500">Customize your experience</p>
        </div>
      </div>

      <div className="bg-[#131825] border border-white/5 rounded-xl p-6 space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Company Name</label>
          <input
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            className={`w-full px-4 py-2.5 rounded-lg border focus:outline-none focus:ring-2 focus:ring-violet-500/50 ${
              theme === 'dark' ? 'bg-white/5 border-white/10 text-white' : 'bg-gray-50 border-gray-200 text-gray-900'
            }`}
            placeholder="Your company name"
          />
          <p className="text-xs text-gray-500 mt-1">Displayed in the sidebar header</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Accent Color</label>
          <div className="flex gap-3">
            {colorOptions.map(c => (
              <button
                key={c.value}
                onClick={() => setColor(c.value)}
                className={`w-10 h-10 rounded-full border-2 transition-all ${
                  color === c.value ? 'border-white scale-110' : 'border-transparent hover:scale-105'
                }`}
                style={{ background: c.value }}
                title={c.name}
              />
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">Current: {colorOptions.find(c => c.value === color)?.name || 'Custom'}</p>
        </div>

        <div className="pt-4 border-t border-white/5 flex justify-end">
          <button
            onClick={handleSave}
            className="px-6 py-2.5 bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors font-medium"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// APP
// ============================================================================

export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <BrandingProvider>
          <ToastProvider>
            <Routes>
              <Route element={<Layout />}>
                <Route path="/" element={<ErrorBoundary><DashboardPage /></ErrorBoundary>} />
                <Route path="/classify" element={<ErrorBoundary><ClassifyPage /></ErrorBoundary>} />
                <Route path="/batch" element={<ErrorBoundary><BatchPage /></ErrorBoundary>} />
                <Route path="/history" element={<ErrorBoundary><HistoryPage /></ErrorBoundary>} />
                <Route path="/analytics" element={<ErrorBoundary><AnalyticsPage /></ErrorBoundary>} />
                <Route path="/ai" element={<ErrorBoundary><AIAssistantPage /></ErrorBoundary>} />
                <Route path="/review" element={<ErrorBoundary><ReviewQueuePage /></ErrorBoundary>} />
                <Route path="/playground" element={<ErrorBoundary><PlaygroundPage /></ErrorBoundary>} />
                <Route path="/compare" element={<ErrorBoundary><ComparePage /></ErrorBoundary>} />
                <Route path="/settings" element={<ErrorBoundary><SettingsPage /></ErrorBoundary>} />
              </Route>
            </Routes>
          </ToastProvider>
        </BrandingProvider>
      </ThemeProvider>
    </BrowserRouter>
  )
}
