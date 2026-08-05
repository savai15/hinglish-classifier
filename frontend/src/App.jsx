import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Menu, ExternalLink, X, Send, Loader2, MessageSquare,
  Package, Truck, CreditCard, RotateCcw, Monitor,
  AlertTriangle, Headphones, Star, Tag, Zap, ChevronDown, ChevronUp,
  ThumbsUp, ThumbsDown, CheckCircle, Database, BarChart3, Grid3X3,
  Brain, Layers, Cpu, Sparkles, ArrowRight
} from 'lucide-react'

const API_BASE = '/api'

const CATEGORIES = {
  Account_Technical:    { icon: Monitor,       accent: 'teal',    label: 'Account' },
  Customer_Service:     { icon: Headphones,    accent: 'violet',  label: 'Support' },
  Delivery_Issue:       { icon: Truck,         accent: 'amber',   label: 'Delivery' },
  Order_Status:         { icon: Package,       accent: 'teal',    label: 'Order' },
  Payment_Invoice:      { icon: CreditCard,    accent: 'emerald', label: 'Payment' },
  Pricing_Discount:     { icon: Tag,           accent: 'violet',  label: 'Pricing' },
  Product_Quality:      { icon: Star,          accent: 'emerald', label: 'Quality' },
  Returns_Refunds:      { icon: RotateCcw,     accent: 'pink',    label: 'Returns' },
  Wrong_Damaged_Product:{ icon: AlertTriangle,  accent: 'rose',    label: 'Damaged' },
}

const ACCENT = {
  teal:    { bg: 'bg-aurora-teal/10', text: 'text-aurora-teal',    border: 'border-aurora-teal/20',    bar: 'bg-aurora-teal',    glow: 'shadow-aurora-teal/20' },
  violet:  { bg: 'bg-aurora-violet/10', text: 'text-aurora-violet', border: 'border-aurora-violet/20', bar: 'bg-aurora-violet',  glow: 'shadow-aurora-violet/20' },
  pink:    { bg: 'bg-aurora-pink/10', text: 'text-aurora-pink',    border: 'border-aurora-pink/20',   bar: 'bg-aurora-pink',    glow: 'shadow-aurora-pink/20' },
  emerald: { bg: 'bg-aurora-emerald/10', text: 'text-aurora-emerald', border: 'border-aurora-emerald/20', bar: 'bg-aurora-emerald', glow: 'shadow-aurora-emerald/20' },
  amber:   { bg: 'bg-aurora-amber/10', text: 'text-aurora-amber',  border: 'border-aurora-amber/20',  bar: 'bg-aurora-amber',   glow: 'shadow-aurora-amber/20' },
  rose:    { bg: 'bg-aurora-rose/10', text: 'text-aurora-rose',    border: 'border-aurora-rose/20',   bar: 'bg-aurora-rose',    glow: 'shadow-aurora-rose/20' },
}

const CATEGORY_RESULT_ACCENT = {
  Order_Status:          'teal',
  Delivery_Issue:        'amber',
  Payment_Invoice:       'emerald',
  Returns_Refunds:       'pink',
  Account_Technical:     'teal',
  Wrong_Damaged_Product: 'rose',
  Customer_Service:      'violet',
  Product_Quality:       'emerald',
  Pricing_Discount:      'violet',
}

const URGENCY_ACCENT = { High: 'rose', Medium: 'amber', Low: 'emerald' }

const SAMPLES = [
  { text: 'Mera order abhi tak nahi aaya, 3 din ho gaye!', category: 'Order_Status', urgency: 'High' },
  { text: 'Refund kab milega? Paisa wapas karo jaldi!', category: 'Returns_Refunds', urgency: 'High' },
  { text: 'Wrong product bheja hai, exchange karo', category: 'Wrong_Damaged_Product', urgency: 'Medium' },
  { text: 'Payment fail ho gaya but paisa kat gaya', category: 'Payment_Invoice', urgency: 'Medium' },
  { text: 'App crash ho raha hai, login nahi ho raha', category: 'Account_Technical', urgency: 'Medium' },
  { text: 'Customer care bilkul useless hai, koi response nahi!', category: 'Customer_Service', urgency: 'High' },
  { text: 'Product ka quality bahut ghatiya hai, 2 din mein kharab!', category: 'Product_Quality', urgency: 'High' },
  { text: 'Checkout pe price badh gaya, hidden charges!', category: 'Pricing_Discount', urgency: 'High' },
  { text: 'Delivery boy rude tha, manager se baat karo', category: 'Delivery_Issue', urgency: 'Medium' },
]

const CATEGORY_LIST = [
  'Account_Technical', 'Customer_Service', 'Delivery_Issue',
  'Order_Status', 'Payment_Invoice', 'Pricing_Discount',
  'Product_Quality', 'Returns_Refunds', 'Wrong_Damaged_Product',
]

const card = {
  hidden: { opacity: 0, y: 20, scale: 0.97 },
  show: { opacity: 1, y: 0, scale: 1 },
}

function ConfidenceBar({ value, color, delay = 0 }) {
  return (
    <div className="w-full h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
      <motion.div
        className={`h-full rounded-full ${color}`}
        initial={{ width: 0 }}
        animate={{ width: `${value * 100}%` }}
        transition={{ duration: 1, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
      />
    </div>
  )
}

function ProbBar({ label, value, color }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] text-slate-400 w-28 truncate shrink-0">
        {label.replace(/_/g, ' ')}
      </span>
      <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>
      <span className="text-[10px] text-slate-500 w-10 text-right shrink-0">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  )
}

function Header({ onMenuClick }) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="px-4 sm:px-6 lg:px-8 pt-6 pb-2 relative z-10"
    >
      <div className="max-w-6xl mx-auto">
        <div className="glass-strong rounded-2xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onMenuClick}
              className="p-2 rounded-xl hover:bg-white/[0.06] transition-all duration-200 lg:hidden"
            >
              <Menu size={20} className="text-slate-400" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-aurora-teal to-aurora-violet flex items-center justify-center shadow-lg shadow-aurora-teal/20">
                <Sparkles size={18} className="text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gradient-aurora">
                  Hinglish Classifier
                </h1>
                <p className="text-[10px] text-slate-500 hidden sm:block tracking-wide uppercase">
                  AI Complaint Analysis
                </p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <a
              href="https://github.com/savai15/test"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2.5 rounded-xl hover:bg-white/[0.06] transition-all duration-200 text-slate-500 hover:text-slate-300"
            >
              <ExternalLink size={16} />
            </a>
            <button
              onClick={onMenuClick}
              className="p-2.5 rounded-xl hover:bg-white/[0.06] transition-all duration-200 text-slate-500 hover:text-slate-300 hidden lg:block"
            >
              <Menu size={16} />
            </button>
          </div>
        </div>
      </div>
    </motion.header>
  )
}

function StatsBar() {
  const stats = [
    { icon: Database, label: 'Dataset', value: '30K', sub: 'Labeled Complaints', color: 'aurora-teal' },
    { icon: BarChart3, label: 'Category F1', value: '99.7%', sub: 'TF-IDF + SVM', color: 'aurora-violet' },
    { icon: Zap, label: 'Urgency F1', value: '99.96%', sub: 'Combined Model', color: 'aurora-pink' },
    { icon: Grid3X3, label: 'Classes', value: '9 + 3', sub: 'Cat + Urgency', color: 'aurora-emerald' },
  ]
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-6 relative z-10">
      {stats.map((s, i) => (
        <motion.div
          key={s.label}
          variants={card}
          initial="hidden"
          animate="show"
          transition={{ duration: 0.5, delay: i * 0.08, ease: [0.25, 0.46, 0.45, 0.94] }}
          whileHover={{ scale: 1.03, y: -4, transition: { duration: 0.2 } }}
          className="glass-strong rounded-2xl p-4 text-center cursor-default group"
        >
          <div className={`w-8 h-8 rounded-lg bg-${s.color}/10 flex items-center justify-center mx-auto mb-2 group-hover:scale-110 transition-transform duration-300`}>
            <s.icon size={16} className={`text-${s.color}`} />
          </div>
          <p className="text-[9px] font-semibold text-slate-500 uppercase tracking-widest">{s.label}</p>
          <p className="text-xl font-bold text-slate-100 mt-1">{s.value}</p>
          <p className="text-[10px] text-slate-500 mt-0.5">{s.sub}</p>
        </motion.div>
      ))}
    </div>
  )
}

function ComplaintInput({ value, onChange, onSubmit, loading }) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSubmit() }
  }
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="glass-strong rounded-2xl p-6 relative z-10"
    >
      <div className="flex items-center gap-2.5 mb-4">
        <div className="p-2 rounded-lg bg-aurora-teal/10">
          <MessageSquare size={14} className="text-aurora-teal" />
        </div>
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Enter Complaint</h2>
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type a Hinglish complaint... e.g., Mera order abhi tak nahi aaya!"
        className="w-full h-32 bg-white/[0.03] border border-white/[0.06] rounded-xl px-4 py-3 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-aurora-teal/30 focus:border-aurora-teal/30 resize-none transition-all duration-300"
        disabled={loading}
      />
      <motion.button
        onClick={onSubmit}
        disabled={!value.trim() || loading}
        className="mt-4 w-full py-3 rounded-xl font-semibold text-sm text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-2 relative overflow-hidden group"
        style={{ background: 'linear-gradient(135deg, #22d3ee 0%, #8b5cf6 50%, #f472b6 100%)' }}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="absolute inset-0 bg-white/0 group-hover:bg-white/10 transition-all duration-300" />
        <span className="relative z-10 flex items-center gap-2">
          {loading ? (
            <><Loader2 size={16} className="animate-spin" />Classifying...</>
          ) : (
            <><Send size={16} />Classify Complaint</>
          )}
        </span>
      </motion.button>
    </motion.div>
  )
}

function SampleGrid({ onSampleClick }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
    >
      <div className="flex items-center gap-2 mb-3 px-1">
        <Zap size={12} className="text-aurora-amber" />
        <h2 className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Quick Samples</h2>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
        {SAMPLES.map((sample, i) => {
          const cat = CATEGORIES[sample.category]
          const Icon = cat.icon
          const a = ACCENT[cat.accent]
          return (
            <motion.button
              key={i}
              variants={card}
              initial="hidden"
              animate="show"
              transition={{ duration: 0.4, delay: 0.25 + i * 0.04 }}
              onClick={() => onSampleClick(sample.text)}
              className="glass rounded-xl p-3 text-left transition-all duration-300 group glass-hover"
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-start gap-2.5">
                <div className={`p-1.5 rounded-lg ${a.bg} shrink-0 group-hover:scale-110 transition-transform duration-300`}>
                  <Icon size={12} className={a.text} />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-[11px] text-slate-300 leading-relaxed line-clamp-2">{sample.text}</p>
                  <div className="flex items-center gap-1.5 mt-2">
                    <span className={`text-[9px] font-medium px-1.5 py-0.5 rounded-md ${a.bg} ${a.text}`}>{cat.label}</span>
                    <span className={`text-[9px] font-medium px-1.5 py-0.5 rounded-md ${ACCENT[URGENCY_ACCENT[sample.urgency]].bg} ${ACCENT[URGENCY_ACCENT[sample.urgency]].text}`}>
                      {sample.urgency}
                    </span>
                  </div>
                </div>
              </div>
            </motion.button>
          )
        })}
      </div>
    </motion.div>
  )
}

function PredictionResults({ result }) {
  const [showProbs, setShowProbs] = useState(false)
  const [showCleaned, setShowCleaned] = useState(false)
  const catAccent = ACCENT[CATEGORY_RESULT_ACCENT[result.category] || 'teal']
  const urgAccent = ACCENT[URGENCY_ACCENT[result.urgency] || 'amber']
  const CatIcon = (CATEGORIES[result.category] || CATEGORIES.Order_Status).icon
  const sortedCatProbs = Object.entries(result.category_probabilities).sort(([, a], [, b]) => b - a)
  const sortedUrgProbs = Object.entries(result.urgency_probabilities).sort(([, a], [, b]) => b - a)

  return (
    <div className="space-y-3">
      {/* Category Result */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
        className={`glass-strong rounded-2xl p-5 border ${catAccent.border} ${catAccent.glow} shadow-lg`}
      >
        <div className="flex items-center gap-2 mb-3">
          <div className={`p-2 rounded-lg ${catAccent.bg}`}>
            <CatIcon size={14} className={catAccent.text} />
          </div>
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Category</span>
        </div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-base font-bold text-slate-100">{result.category.replace(/_/g, ' ')}</h3>
          <span className={`text-sm font-bold ${catAccent.text}`}>
            {(result.category_confidence * 100).toFixed(1)}%
          </span>
        </div>
        <ConfidenceBar value={result.category_confidence} color={catAccent.bar} delay={0.2} />
        {result.needs_review && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-3 px-3 py-2 bg-aurora-amber/10 rounded-lg border border-aurora-amber/20"
          >
            <p className="text-[11px] text-aurora-amber">Low confidence - please verify this prediction</p>
          </motion.div>
        )}
      </motion.div>

      {/* Urgency Result */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
        className={`glass-strong rounded-2xl p-5 border ${urgAccent.border} ${urgAccent.glow} shadow-lg`}
      >
        <div className="flex items-center gap-2 mb-3">
          <div className={`p-2 rounded-lg ${urgAccent.bg}`}>
            <span className={`text-sm font-bold ${urgAccent.text}`}>
              {result.urgency === 'High' ? '!' : result.urgency === 'Medium' ? '~' : '?'}
            </span>
          </div>
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Urgency</span>
        </div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-base font-bold text-slate-100">{result.urgency}</h3>
          <span className={`text-sm font-bold ${urgAccent.text}`}>
            {(result.urgency_confidence * 100).toFixed(1)}%
          </span>
        </div>
        <ConfidenceBar value={result.urgency_confidence} color={urgAccent.bar} delay={0.3} />
      </motion.div>

      {/* All Probabilities */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.3 }}
        className="glass rounded-xl overflow-hidden"
      >
        <button
          onClick={() => setShowProbs(!showProbs)}
          className="w-full px-4 py-3 flex items-center justify-between hover:bg-white/[0.03] transition-colors duration-200"
        >
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">All Probabilities</span>
          <motion.div animate={{ rotate: showProbs ? 180 : 0 }} transition={{ duration: 0.2 }}>
            <ChevronDown size={14} className="text-slate-500" />
          </motion.div>
        </button>
        <AnimatePresence>
          {showProbs && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="overflow-hidden"
            >
              <div className="px-4 pb-4 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[9px] font-semibold text-slate-600 uppercase mb-2 tracking-wider">Categories</p>
                  <div className="space-y-1.5">
                    {sortedCatProbs.map(([label, value]) => (
                      <ProbBar key={label} label={label} value={value} color={label === result.category ? catAccent.bar : 'bg-slate-600'} />
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-[9px] font-semibold text-slate-600 uppercase mb-2 tracking-wider">Urgency</p>
                  <div className="space-y-1.5">
                    {sortedUrgProbs.map(([label, value]) => (
                      <ProbBar key={label} label={label} value={value} color={label === result.urgency ? urgAccent.bar : 'bg-slate-600'} />
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Preprocessed Text */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.35 }}
        className="glass rounded-xl overflow-hidden"
      >
        <button
          onClick={() => setShowCleaned(!showCleaned)}
          className="w-full px-4 py-3 flex items-center justify-between hover:bg-white/[0.03] transition-colors duration-200"
        >
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Preprocessed Text</span>
          <motion.div animate={{ rotate: showCleaned ? 180 : 0 }} transition={{ duration: 0.2 }}>
            <ChevronDown size={14} className="text-slate-500" />
          </motion.div>
        </button>
        <AnimatePresence>
          {showCleaned && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="px-4 pb-4">
                <p className="text-[11px] text-slate-400 bg-white/[0.03] rounded-lg p-3 font-mono leading-relaxed border border-white/[0.04]">
                  {result.cleaned_text}
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}

function FeedbackPanel({ predictionId }) {
  const [catCorrect, setCatCorrect] = useState(true)
  const [urgCorrect, setUrgCorrect] = useState(true)
  const [correctedCat, setCorrectedCat] = useState('')
  const [correctedUrg, setCorrectedUrg] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [showPanel, setShowPanel] = useState(false)

  const submit = async () => {
    try {
      await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prediction_id: predictionId,
          is_correct_category: catCorrect,
          is_correct_urgency: urgCorrect,
          corrected_category: !catCorrect ? correctedCat : null,
          corrected_urgency: !urgCorrect ? correctedUrg : null,
        }),
      })
      setSubmitted(true)
      setTimeout(() => { setSubmitted(false); setShowPanel(false); setCatCorrect(true); setUrgCorrect(true) }, 2500)
    } catch (err) { console.error('Feedback failed:', err) }
  }

  const toggleBtn = (active, type) =>
    `flex-1 py-2 rounded-lg text-[11px] font-medium transition-all duration-200 flex items-center justify-center gap-1.5 border ${
      active
        ? type === 'correct'
          ? 'bg-aurora-emerald/10 text-aurora-emerald border-aurora-emerald/20'
          : 'bg-aurora-rose/10 text-aurora-rose border-aurora-rose/20'
        : 'bg-white/[0.03] text-slate-500 border-white/[0.06] hover:bg-white/[0.06]'
    }`

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.4 }}
      className="mt-3"
    >
      {!showPanel && !submitted && (
        <motion.button
          onClick={() => setShowPanel(true)}
          className="w-full glass rounded-xl px-4 py-2.5 flex items-center justify-center gap-2 text-[11px] font-medium text-slate-500 hover:text-slate-300 hover:bg-white/[0.04] transition-all duration-200"
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
        >
          <ThumbsUp size={12} /> Was this correct?
        </motion.button>
      )}
      <AnimatePresence mode="wait">
        {submitted && (
          <motion.div
            key="thanks"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="glass-strong rounded-xl p-5 text-center"
          >
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.1 }}
            >
              <CheckCircle size={28} className="text-aurora-emerald mx-auto mb-2" />
            </motion.div>
            <p className="text-sm font-semibold text-slate-200">Thank you!</p>
            <p className="text-[11px] text-slate-500 mt-1">Your feedback helps improve the model</p>
          </motion.div>
        )}
        {showPanel && !submitted && (
          <motion.div
            key="panel"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="glass-strong rounded-xl p-5 space-y-4 overflow-hidden"
          >
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Feedback</p>
            <div>
              <p className="text-[11px] text-slate-400 mb-2">Category prediction:</p>
              <div className="flex gap-2">
                <button onClick={() => setCatCorrect(true)} className={toggleBtn(catCorrect, 'correct')}><ThumbsUp size={11} /> Correct</button>
                <button onClick={() => setCatCorrect(false)} className={toggleBtn(!catCorrect, 'wrong')}><ThumbsDown size={11} /> Wrong</button>
              </div>
              <AnimatePresence>
                {!catCorrect && (
                  <motion.select
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    value={correctedCat}
                    onChange={(e) => setCorrectedCat(e.target.value)}
                    className="mt-2 w-full bg-white/[0.03] border border-white/[0.06] rounded-lg px-3 py-2 text-[11px] text-slate-300 focus:outline-none focus:ring-2 focus:ring-aurora-teal/30"
                  >
                    <option value="" className="bg-surface-raised">Select correct category...</option>
                    {CATEGORY_LIST.map((c) => <option key={c} value={c} className="bg-surface-raised">{c.replace(/_/g, ' ')}</option>)}
                  </motion.select>
                )}
              </AnimatePresence>
            </div>
            <div>
              <p className="text-[11px] text-slate-400 mb-2">Urgency prediction:</p>
              <div className="flex gap-2">
                <button onClick={() => setUrgCorrect(true)} className={toggleBtn(urgCorrect, 'correct')}><ThumbsUp size={11} /> Correct</button>
                <button onClick={() => setUrgCorrect(false)} className={toggleBtn(!urgCorrect, 'wrong')}><ThumbsDown size={11} /> Wrong</button>
              </div>
              <AnimatePresence>
                {!urgCorrect && (
                  <motion.select
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    value={correctedUrg}
                    onChange={(e) => setCorrectedUrg(e.target.value)}
                    className="mt-2 w-full bg-white/[0.03] border border-white/[0.06] rounded-lg px-3 py-2 text-[11px] text-slate-300 focus:outline-none focus:ring-2 focus:ring-aurora-teal/30"
                  >
                    <option value="" className="bg-surface-raised">Select correct urgency...</option>
                    <option value="High" className="bg-surface-raised">High</option>
                    <option value="Medium" className="bg-surface-raised">Medium</option>
                    <option value="Low" className="bg-surface-raised">Low</option>
                  </motion.select>
                )}
              </AnimatePresence>
            </div>
            <div className="flex gap-2 pt-1">
              <button
                onClick={() => setShowPanel(false)}
                className="flex-1 py-2.5 rounded-lg text-[11px] font-medium text-slate-500 bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] transition-all duration-200"
              >
                Cancel
              </button>
              <motion.button
                onClick={submit}
                className="flex-1 py-2.5 rounded-lg text-[11px] font-semibold text-white flex items-center justify-center gap-1.5 relative overflow-hidden group"
                style={{ background: 'linear-gradient(135deg, #22d3ee 0%, #8b5cf6 100%)' }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="absolute inset-0 bg-white/0 group-hover:bg-white/10 transition-all duration-300" />
                <span className="relative z-10 flex items-center gap-1.5"><Send size={11} /> Submit</span>
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function Sidebar({ open, onClose }) {
  const features = [
    { icon: Brain, label: '9 Categories', desc: 'Complaint classification', color: 'aurora-teal' },
    { icon: Layers, label: 'Ensemble', desc: 'TF-IDF + SVM + LR', color: 'aurora-violet' },
    { icon: Cpu, label: 'MuRIL Ready', desc: 'Multilingual BERT', color: 'aurora-pink' },
    { icon: Database, label: '30K Dataset', desc: 'Labeled complaints', color: 'aurora-emerald' },
  ]
  const techStack = [
    { name: 'Python', color: 'aurora-teal' },
    { name: 'Scikit-learn', color: 'aurora-emerald' },
    { name: 'FastAPI', color: 'aurora-violet' },
    { name: 'React', color: 'aurora-pink' },
    { name: 'SQLite', color: 'aurora-amber' },
  ]

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden"
          />
          <motion.aside
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed left-0 top-0 bottom- w-80 glass-strong z-50 flex flex-col border-r border-white/[0.06]"
          >
            <div className="p-5 flex items-center justify-between border-b border-white/[0.06]">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-aurora-teal to-aurora-violet flex items-center justify-center">
                  <Sparkles size={14} className="text-white" />
                </div>
                <span className="text-sm font-semibold text-slate-200">About</span>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-white/[0.06] transition-colors duration-200"
              >
                <X size={14} className="text-slate-500" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-5 py-5 space-y-5">
              <div>
                <p className="text-[9px] font-semibold text-slate-600 uppercase tracking-widest mb-3">Model Performance</p>
                <div className="space-y-2">
                  {[
                    { label: 'Category F1', val: '99.7%', pct: '99.7%', color: 'aurora-teal' },
                    { label: 'Urgency F1', val: '99.96%', pct: '99.96%', color: 'aurora-violet' },
                  ].map((m) => (
                    <div key={m.label} className="glass rounded-lg p-3">
                      <div className="flex justify-between items-center">
                        <span className="text-[11px] text-slate-400">{m.label}</span>
                        <span className="text-xs font-bold text-slate-200">{m.val}</span>
                      </div>
                      <div className="mt-2 h-1 bg-white/[0.06] rounded-full overflow-hidden">
                        <div className={`h-full bg-${m.color} rounded-full`} style={{ width: m.pct }} />
                      </div>
                    </div>
                  ))}
                  <div className="glass rounded-lg p-3">
                    <div className="flex justify-between items-center">
                      <span className="text-[11px] text-slate-400">Training Samples</span>
                      <span className="text-xs font-bold text-slate-200">30,000</span>
                    </div>
                  </div>
                </div>
              </div>
              <div>
                <p className="text-[9px] font-semibold text-slate-600 uppercase tracking-widest mb-3">Features</p>
                <div className="grid grid-cols-2 gap-2">
                  {features.map((f) => (
                    <div key={f.label} className="glass rounded-lg p-3 text-center group hover:bg-white/[0.04] transition-colors duration-200">
                      <div className={`w-8 h-8 rounded-lg bg-${f.color}/10 flex items-center justify-center mx-auto mb-1.5 group-hover:scale-110 transition-transform duration-300`}>
                        <f.icon size={14} className={`text-${f.color}`} />
                      </div>
                      <p className="text-[10px] font-semibold text-slate-300">{f.label}</p>
                      <p className="text-[9px] text-slate-600 mt-0.5">{f.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-[9px] font-semibold text-slate-600 uppercase tracking-widest mb-3">Tech Stack</p>
                <div className="flex flex-wrap gap-1.5">
                  {techStack.map((t) => (
                    <span key={t.name} className={`text-[9px] font-medium px-2 py-0.5 rounded-md bg-${t.color}/10 text-${t.color}`}>
                      {t.name}
                    </span>
                  ))}
                </div>
              </div>
              <div className="glass rounded-lg p-4">
                <p className="text-[9px] font-semibold text-slate-600 uppercase tracking-widest mb-2">REST API</p>
                <p className="text-[10px] text-slate-400 leading-relaxed">
                  FastAPI backend at <span className="font-mono text-aurora-teal">localhost:8000</span><br />
                  Interactive docs at <span className="font-mono text-aurora-teal">/docs</span>
                </p>
              </div>
              <a
                href="https://github.com/savai15/test"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 glass rounded-lg p-3 hover:bg-white/[0.04] transition-colors duration-200 group"
              >
                <ExternalLink size={14} className="text-slate-500 group-hover:text-slate-300 transition-colors" />
                <span className="text-[11px] text-slate-400 group-hover:text-slate-300 transition-colors">View on GitHub</span>
                <ArrowRight size={10} className="text-slate-600 ml-auto group-hover:translate-x-0.5 transition-transform" />
              </a>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}

export default function App() {
  const [complaint, setComplaint] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const classify = async (text) => {
    if (!text?.trim()) return
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      const data = await res.json()
      setResult(data)
    } catch (err) { console.error('Prediction failed:', err) }
    finally { setLoading(false) }
  }

  const handleSampleClick = (text) => { setComplaint(text); classify(text) }

  return (
    <div className="min-h-screen relative">
      {/* Aurora background */}
      <div className="aurora-bg">
        <div className="aurora-orb-pink" />
      </div>

      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="relative z-10 flex flex-col min-h-screen">
        <Header onMenuClick={() => setSidebarOpen(true)} />

        <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 lg:px-8 pb-12">
          <StatsBar />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
            <div className="space-y-5">
              <ComplaintInput
                value={complaint}
                onChange={setComplaint}
                onSubmit={() => classify(complaint)}
                loading={loading}
              />
              <SampleGrid onSampleClick={handleSampleClick} />
            </div>

            <div>
              <AnimatePresence mode="wait">
                {result && (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
                  >
                    <PredictionResults result={result} />
                    <FeedbackPanel predictionId={result.id} />
                  </motion.div>
                )}
              </AnimatePresence>

              {!result && !loading && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
                  className="glass-strong rounded-2xl p-10 text-center"
                >
                  <motion.div
                    className="w-16 h-16 rounded-2xl bg-gradient-to-br from-aurora-teal/20 to-aurora-violet/20 flex items-center justify-center mx-auto mb-5 border border-white/[0.06]"
                    animate={{ y: [0, -6, 0] }}
                    transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                  >
                    <Sparkles size={28} className="text-aurora-teal" />
                  </motion.div>
                  <h3 className="text-lg font-semibold text-slate-200 mb-2">Enter a complaint to classify</h3>
                  <p className="text-[12px] text-slate-500 max-w-xs mx-auto">
                    Type a Hinglish complaint or click a sample on the left to get started
                  </p>
                </motion.div>
              )}

              {loading && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="glass-strong rounded-2xl p-10 text-center"
                >
                  <div className="flex items-center justify-center gap-2 mb-4">
                    {['bg-aurora-teal', 'bg-aurora-violet', 'bg-aurora-pink'].map((c, i) => (
                      <motion.div
                        key={c}
                        className={`w-2.5 h-2.5 rounded-full ${c}`}
                        animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2, ease: 'easeInOut' }}
                      />
                    ))}
                  </div>
                  <p className="text-[12px] text-slate-500">Analyzing complaint...</p>
                </motion.div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
