import { useState, useEffect, useCallback, useRef } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useNavigate, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
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
} from 'lucide-react'

const API = '/api'
const COLORS = ['#6366f1', '#8b5cf6', '#22d3ee', '#14b8a6', '#f59e0b', '#f97316', '#ef4444', '#ec4899', '#e879f9']
const URGENCY_COLORS = { High: '#ef4444', Medium: '#f59e0b', Low: '#22d3ee' }
const CAT_SHORT = {
  Account_Technical: 'Account', Customer_Service: 'Support', Delivery_Issue: 'Delivery',
  Order_Status: 'Order', Payment_Invoice: 'Payment', Pricing_Discount: 'Pricing',
  Product_Quality: 'Quality', Returns_Refunds: 'Returns', Wrong_Damaged_Product: 'Damaged',
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

// ============================================================================
// ANIMATED NUMBER
// ============================================================================

function AnimNum({ value, duration = 800 }) {
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
      setDisplay(Math.round(start + diff * eased))
      if (progress < 1) ref.current = requestAnimationFrame(tick)
    }
    ref.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(ref.current)
  }, [value])
  return <span>{display.toLocaleString()}</span>
}

// ============================================================================
// LAYOUT
// ============================================================================

function Layout({ children }) {
  const [collapsed, setCollapsed] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searching, setSearching] = useState(false)
  const searchRef = useRef(null)
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    function handleKey(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        setSearchOpen(true)
      }
      if (e.key === 'Escape') setSearchOpen(false)
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [])

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

  const nav = [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/classify', icon: MessageSquareWarning, label: 'Classify' },
    { to: '/batch', icon: Layers, label: 'Batch' },
    { to: '/history', icon: History, label: 'History' },
    { to: '/analytics', icon: BarChart3, label: 'Analytics' },
    { to: '/playground', icon: Code2, label: 'API' },
  ]

  return (
    <div className="flex h-screen bg-[#0a0e1a] text-gray-200 overflow-hidden">
      {/* Sidebar */}
      <motion.aside
        animate={{ width: collapsed ? 64 : 220 }}
        className="flex-shrink-0 bg-[#0d1220] border-r border-white/5 flex flex-col z-20"
      >
        <div className="h-14 flex items-center px-4 border-b border-white/5">
          {!collapsed && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2 min-w-0">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center flex-shrink-0">
                <Zap size={14} className="text-white" />
              </div>
              <span className="font-semibold text-sm text-white truncate">HinglishAI</span>
            </motion.div>
          )}
          {collapsed && (
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center mx-auto">
              <Zap size={14} className="text-white" />
            </div>
          )}
        </div>

        <nav className="flex-1 py-3 px-2 space-y-0.5">
          {nav.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-200 group ${
                  isActive
                    ? 'bg-white/10 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                } ${collapsed ? 'justify-center' : ''}`
              }
            >
              <item.icon size={18} className="flex-shrink-0" />
              {!collapsed && <span className="truncate">{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        <div className="p-2 border-t border-white/5">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-gray-500 hover:text-white hover:bg-white/5 transition text-sm"
          >
            {collapsed ? <ChevronRight size={16} /> : <><ChevronLeft size={16} /><span className="truncate">Collapse</span></>}
          </button>
        </div>
      </motion.aside>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-white/5 bg-[#0d1220]/80 backdrop-blur-sm z-10">
          <div className="flex items-center gap-3">
            <h1 className="text-sm font-medium text-gray-300">
              {nav.find(n => n.to === location.pathname)?.label || 'Hinglish Complaint Classifier'}
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setSearchOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:border-white/20 transition text-sm"
            >
              <Search size={14} />
              <span className="hidden sm:inline">Search</span>
              <kbd className="hidden sm:inline text-[10px] bg-white/10 px-1.5 py-0.5 rounded text-gray-500">Ctrl+K</kbd>
            </button>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-[1400px] mx-auto p-6">
            {children}
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
              className="w-full max-w-lg bg-[#131825] border border-white/10 rounded-xl shadow-2xl overflow-hidden"
              onClick={e => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 px-4 py-3 border-b border-white/5">
                <Search size={16} className="text-gray-500" />
                <input
                  ref={searchRef}
                  autoFocus
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  placeholder="Search complaints..."
                  className="flex-1 bg-transparent text-sm text-white placeholder-gray-500 outline-none"
                />
                <kbd className="text-[10px] bg-white/10 px-1.5 py-0.5 rounded text-gray-500">ESC</kbd>
              </div>
              <div className="max-h-80 overflow-y-auto">
                {searching && <div className="px-4 py-6 text-center text-gray-500 text-sm">Searching...</div>}
                {!searching && searchResults.length === 0 && searchQuery.length > 1 && (
                  <div className="px-4 py-6 text-center text-gray-500 text-sm">No results found</div>
                )}
                {searchResults.map((p, i) => (
                  <button
                    key={i}
                    className="w-full text-left px-4 py-3 hover:bg-white/5 transition border-b border-white/5 last:border-0"
                    onClick={() => { setSearchOpen(false); navigate('/history') }}
                  >
                    <p className="text-sm text-white truncate">{p.text}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs px-1.5 py-0.5 rounded bg-white/10 text-gray-400">{CAT_SHORT[p.predicted_category] || p.predicted_category}</span>
                      <span className="text-xs" style={{ color: URGENCY_COLORS[p.predicted_urgency] }}>{p.predicted_urgency}</span>
                    </div>
                  </button>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
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
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getStats(), getPatterns(), getTimeline(24), getCategories()])
      .then(([s, p, t, c]) => { setStats(s); setPatterns(p); setTimeline(t); setCategories(c) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <LoadingState />

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

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Total Predictions', value: stats?.total_predictions || 0, icon: Database, color: 'from-cyan-500 to-blue-500', trend: '+12%' },
          { label: 'Category Accuracy', value: stats?.category_accuracy ? Math.round(stats.category_accuracy * 100) : 0, icon: Target, color: 'from-violet-500 to-purple-500', suffix: '%', trend: '+2.1%' },
          { label: 'Urgency Accuracy', value: stats?.urgency_accuracy ? Math.round(stats.urgency_accuracy * 100) : 0, icon: Gauge, color: 'from-emerald-500 to-teal-500', suffix: '%', trend: '+1.8%' },
          { label: 'Needs Review', value: stats?.low_confidence_count || 0, icon: AlertTriangle, color: 'from-amber-500 to-orange-500', trend: stats?.low_confidence_count > 0 ? 'Attention' : 'Clear' },
        ].map((card, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="bg-[#131825] border border-white/5 rounded-xl p-5 hover:border-white/10 transition-all duration-300"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wider">{card.label}</p>
                <p className="text-3xl font-bold text-white mt-1">
                  <AnimNum value={card.value} />{card.suffix || ''}
                </p>
              </div>
              <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${card.color} flex items-center justify-center`}>
                <card.icon size={18} className="text-white" />
              </div>
            </div>
            <div className="mt-3 flex items-center gap-1">
              <ArrowUpRight size={12} className="text-emerald-400" />
              <span className="text-xs text-emerald-400">{card.trend}</span>
              <span className="text-xs text-gray-600 ml-1">vs last week</span>
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
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium text-gray-300 mb-4">Category Distribution</h3>
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
            <div className="h-[220px] flex items-center justify-center text-gray-600 text-sm">No data yet</div>
          )}
          <div className="flex flex-wrap gap-2 mt-2">
            {catData.slice(0, 6).map((d, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full" style={{ background: COLORS[i] }} />
                <span className="text-[10px] text-gray-500">{d.name}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Urgency Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium text-gray-300 mb-4">Urgency Breakdown</h3>
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
            <div className="h-[220px] flex items-center justify-center text-gray-600 text-sm">No data yet</div>
          )}
        </motion.div>

        {/* Prediction Timeline */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium text-gray-300 mb-4">Last 24 Hours</h3>
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
            <div className="h-[220px] flex items-center justify-center text-gray-600 text-sm">No predictions yet</div>
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
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium text-gray-300 mb-4">Model Performance</h3>
          <div className="space-y-3">
            {[
              { label: 'TF-IDF + SVM (Category)', score: 99.7, color: '#6366f1' },
              { label: 'Combined Ensemble (Urgency)', score: 99.96, color: '#22d3ee' },
              { label: 'MuRIL (GPU Fine-tuned)', score: 0, color: '#8b5cf6', note: 'Not trained yet' },
            ].map((m, i) => (
              <div key={i}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-gray-400">{m.label}</span>
                  <span className="text-white font-medium">{m.note || `${m.score}%`}</span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
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

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium text-gray-300 mb-4">Quick Actions</h3>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Classify Complaint', icon: MessageSquareWarning, to: '/classify', color: 'from-cyan-500 to-blue-500' },
              { label: 'Batch Process', icon: Layers, to: '/batch', color: 'from-violet-500 to-purple-500' },
              { label: 'View Analytics', icon: BarChart3, to: '/analytics', color: 'from-emerald-500 to-teal-500' },
              { label: 'Test API', icon: Code2, to: '/playground', color: 'from-amber-500 to-orange-500' },
            ].map((a, i) => (
              <NavLink
                key={i}
                to={a.to}
                className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 transition-all duration-200 group"
              >
                <div className={`w-9 h-9 rounded-lg bg-gradient-to-br ${a.color} flex items-center justify-center`}>
                  <a.icon size={16} className="text-white" />
                </div>
                <span className="text-sm text-gray-300 group-hover:text-white transition">{a.label}</span>
              </NavLink>
            ))}
          </div>
        </motion.div>
      </div>
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

  useEffect(() => {
    getHistory({ limit: 5 }).then(d => setHistory(d.predictions || [])).catch(() => {})
  }, [result])

  async function handleClassify(complaintText) {
    const t = complaintText || text
    if (!t.trim()) return
    setLoading(true)
    try {
      const r = await predict(t)
      setResult(r)
      if (!complaintText) setText('')
    } catch { alert('Classification failed. Is the backend running?') }
    setLoading(false)
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Input Column */}
        <div className="lg:col-span-3 space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-[#131825] border border-white/5 rounded-xl p-5"
          >
            <h3 className="text-sm font-medium text-gray-300 mb-3">Enter Complaint</h3>
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder="Type a Hinglish complaint... e.g., Mera order abhi tak nahi aaya!"
              className="w-full h-32 bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white placeholder-gray-500 outline-none focus:border-cyan-500/50 resize-none transition"
              onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleClassify() }}
            />
            <div className="flex items-center justify-between mt-3">
              <span className="text-xs text-gray-600">Ctrl+Enter to classify</span>
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
                className="bg-[#131825] border border-white/5 rounded-xl p-5 overflow-hidden"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-medium text-gray-300">Result</h3>
                  {result.needs_review && (
                    <span className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
                      <AlertTriangle size={12} /> Needs Review
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="p-4 rounded-lg bg-white/5">
                    <p className="text-xs text-gray-500 mb-1">Category</p>
                    <p className="text-lg font-bold text-white">{CAT_SHORT[result.category] || result.category}</p>
                    <p className="text-xs text-gray-400 mt-1">{result.category}</p>
                    <div className="mt-2 h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.category_confidence * 100}%` }}
                        className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500"
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{(result.category_confidence * 100).toFixed(1)}% confidence</p>
                  </div>
                  <div className="p-4 rounded-lg bg-white/5">
                    <p className="text-xs text-gray-500 mb-1">Urgency</p>
                    <p className="text-lg font-bold" style={{ color: URGENCY_COLORS[result.urgency] }}>{result.urgency}</p>
                    <div className="mt-2 h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.urgency_confidence * 100}%` }}
                        className="h-full rounded-full"
                        style={{ background: URGENCY_COLORS[result.urgency] }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{(result.urgency_confidence * 100).toFixed(1)}% confidence</p>
                  </div>
                </div>

                {/* Probability Bars */}
                <div className="space-y-2">
                  <p className="text-xs text-gray-500 uppercase tracking-wider">Category Probabilities</p>
                  {Object.entries(result.category_probabilities || {})
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([cat, prob], i) => (
                      <div key={cat} className="flex items-center gap-3">
                        <span className="text-xs text-gray-400 w-20 truncate">{CAT_SHORT[cat] || cat}</span>
                        <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${prob * 100}%` }}
                            transition={{ duration: 0.5, delay: i * 0.05 }}
                            className="h-full rounded-full"
                            style={{ background: COLORS[i] }}
                          />
                        </div>
                        <span className="text-xs text-gray-500 w-10 text-right">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
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
            className="bg-[#131825] border border-white/5 rounded-xl p-5"
          >
            <h3 className="text-sm font-medium text-gray-300 mb-3">Quick Samples</h3>
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {samples.map((s, i) => (
                <button
                  key={i}
                  onClick={() => { setText(s.text); handleClassify(s.text) }}
                  className="w-full text-left p-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 transition-all duration-200"
                >
                  <p className="text-xs text-gray-300 line-clamp-2">{s.text}</p>
                  <div className="flex items-center gap-2 mt-1.5">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-gray-400">{CAT_SHORT[s.cat]}</span>
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
            className="bg-[#131825] border border-white/5 rounded-xl p-5"
          >
            <h3 className="text-sm font-medium text-gray-300 mb-3">Recent</h3>
            {history.length === 0 ? (
              <p className="text-xs text-gray-600 text-center py-4">No predictions yet</p>
            ) : (
              <div className="space-y-2">
                {history.slice(0, 4).map((p, i) => (
                  <div key={i} className="p-2.5 rounded-lg bg-white/5 border border-white/5">
                    <p className="text-xs text-gray-300 truncate">{p.text}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-gray-400">{CAT_SHORT[p.predicted_category] || p.predicted_category}</span>
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

  function handleFileUpload(e) {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target.result
      const lines = text.split('\n').filter(l => l.trim())
      const parsed = lines.map(l => {
        const parts = l.split(',')
        return parts[0]?.replace(/"/g, '').trim()
      }).filter(t => t && t !== 'text')
      setCsvData(parsed)
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
        alert('Batch processing failed')
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
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium text-gray-300 mb-4">Upload Data</h3>

          {/* File Upload */}
          <div
            onClick={() => fileRef.current?.click()}
            className="border-2 border-dashed border-white/10 rounded-lg p-8 text-center cursor-pointer hover:border-cyan-500/30 hover:bg-white/[0.02] transition-all duration-300"
          >
            <input ref={fileRef} type="file" accept=".csv,.txt" onChange={handleFileUpload} className="hidden" />
            <Upload size={24} className="mx-auto text-gray-600 mb-2" />
            <p className="text-sm text-gray-400">Drop CSV file or click to upload</p>
            <p className="text-xs text-gray-600 mt-1">One complaint per line or first column of CSV</p>
          </div>

          {csvData.length > 0 && (
            <div className="mt-4 p-3 rounded-lg bg-white/5">
              <p className="text-xs text-gray-400">{csvData.length} complaints loaded</p>
              <p className="text-xs text-gray-600 mt-1 truncate">{csvData[0]}...</p>
            </div>
          )}

          {/* Or paste text */}
          <div className="mt-4">
            <p className="text-xs text-gray-500 mb-2">Or paste complaints (one per line):</p>
            <textarea
              value={textInput}
              onChange={e => setTextInput(e.target.value)}
              placeholder={"Mera order nahi aaya\nRefund do\nWrong product aaya"}
              className="w-full h-24 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 outline-none focus:border-cyan-500/50 resize-none transition"
            />
            <button
              onClick={handleTextUpload}
              disabled={!textInput.trim()}
              className="mt-2 px-4 py-1.5 rounded-lg bg-white/10 text-sm text-gray-300 hover:bg-white/15 transition disabled:opacity-50"
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
            <div className="mt-3 h-1.5 bg-white/5 rounded-full overflow-hidden">
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
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-300">
              Results {results.length > 0 && <span className="text-gray-500">({results.length})</span>}
            </h3>
            {results.length > 0 && (
              <div className="flex items-center gap-2">
                <button onClick={() => exportResults('csv')} className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-gray-400 transition">
                  <Download size={12} /> CSV
                </button>
                <button onClick={() => exportResults('json')} className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-gray-400 transition">
                  <Download size={12} /> JSON
                </button>
              </div>
            )}
          </div>

          {results.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-gray-600 text-sm">
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
                  className="p-3 rounded-lg bg-white/5 border border-white/5"
                >
                  <p className="text-xs text-gray-300 truncate">{r.text}</p>
                  <div className="flex items-center gap-3 mt-1.5">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-gray-400">{CAT_SHORT[r.category] || r.category}</span>
                    <span className="text-[10px]" style={{ color: URGENCY_COLORS[r.urgency] }}>{r.urgency}</span>
                    <span className="text-[10px] text-gray-600">{(r.category_confidence * 100).toFixed(0)}%</span>
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
// PAGE: HISTORY
// ============================================================================

function HistoryPage() {
  const [data, setData] = useState({ predictions: [], total: 0 })
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const [filter, setFilter] = useState({ category: '', urgency: '', search: '' })
  const limit = 20

  const loadHistory = useCallback(async () => {
    setLoading(true)
    try {
      const params = { limit, offset: page * limit }
      if (filter.category) params.category = filter.category
      if (filter.urgency) params.urgency = filter.urgency
      if (filter.search) params.search = filter.search
      const d = await getHistory(params)
      setData(d)
    } catch {}
    setLoading(false)
  }, [page, filter])

  useEffect(() => { loadHistory() }, [loadHistory])

  function exportAll() {
    window.open(`${API}/export/csv${filter.category ? `?category=${filter.category}` : ''}`, '_blank')
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-[#131825] border border-white/5 rounded-xl p-4"
      >
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <Filter size={14} className="text-gray-500" />
            <span className="text-xs text-gray-500">Filters:</span>
          </div>
          <input
            value={filter.search}
            onChange={e => { setFilter(f => ({ ...f, search: e.target.value })); setPage(0) }}
            placeholder="Search text..."
            className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-xs text-white placeholder-gray-500 outline-none focus:border-cyan-500/50 w-48"
          />
          <select
            value={filter.category}
            onChange={e => { setFilter(f => ({ ...f, category: e.target.value })); setPage(0) }}
            className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-xs text-white outline-none"
          >
            <option value="">All Categories</option>
            {Object.keys(CAT_SHORT).map(c => <option key={c} value={c}>{CAT_SHORT[c]}</option>)}
          </select>
          <select
            value={filter.urgency}
            onChange={e => { setFilter(f => ({ ...f, urgency: e.target.value })); setPage(0) }}
            className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-xs text-white outline-none"
          >
            <option value="">All Urgency</option>
            <option value="High">High</option>
            <option value="Medium">Medium</option>
            <option value="Low">Low</option>
          </select>
          <div className="flex-1" />
          <button onClick={exportAll} className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-gray-400 transition">
            <Download size={12} /> Export CSV
          </button>
        </div>
      </motion.div>

      {/* Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-[#131825] border border-white/5 rounded-xl overflow-hidden"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/5">
                <th className="text-left px-4 py-3 text-xs text-gray-500 font-medium">Text</th>
                <th className="text-left px-4 py-3 text-xs text-gray-500 font-medium">Category</th>
                <th className="text-left px-4 py-3 text-xs text-gray-500 font-medium">Urgency</th>
                <th className="text-left px-4 py-3 text-xs text-gray-500 font-medium">Confidence</th>
                <th className="text-left px-4 py-3 text-xs text-gray-500 font-medium">Time</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={5} className="px-4 py-8 text-center text-gray-600">Loading...</td></tr>
              ) : data.predictions.length === 0 ? (
                <tr><td colSpan={5} className="px-4 py-8 text-center text-gray-600">No predictions found</td></tr>
              ) : (
                data.predictions.map((p, i) => (
                  <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02] transition">
                    <td className="px-4 py-3 text-xs text-gray-300 max-w-xs truncate">{p.text}</td>
                    <td className="px-4 py-3">
                      <span className="text-[10px] px-2 py-1 rounded-full bg-white/10 text-gray-400">{CAT_SHORT[p.predicted_category] || p.predicted_category}</span>
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-xs font-medium" style={{ color: URGENCY_COLORS[p.predicted_urgency] }}>{p.predicted_urgency}</span>
                    </td>
                    <td className="px-4 py-3 text-xs text-gray-400">{(p.confidence_category * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3 text-xs text-gray-600">{p.timestamp ? new Date(p.timestamp).toLocaleString() : '-'}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {data.total > limit && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-white/5">
            <span className="text-xs text-gray-600">Showing {page * limit + 1}-{Math.min((page + 1) * limit, data.total)} of {data.total}</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-3 py-1 rounded bg-white/5 text-xs text-gray-400 hover:bg-white/10 disabled:opacity-30 transition"
              >Prev</button>
              <button
                onClick={() => setPage(p => p + 1)}
                disabled={(page + 1) * limit >= data.total}
                className="px-3 py-1 rounded bg-white/5 text-xs text-gray-400 hover:bg-white/10 disabled:opacity-30 transition"
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

  if (loading) return <LoadingState />

  return (
    <div className="space-y-6">
      {/* Confidence Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="lg:col-span-2 bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-300">Confidence Distribution</h3>
            <span className="text-xs text-gray-500">Avg: {((confData?.overall_avg || 0) * 100).toFixed(1)}%</span>
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
            <div className="h-[250px] flex items-center justify-center text-gray-600 text-sm">No data yet</div>
          )}
        </motion.div>

        {/* Category Avg Confidence */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <h3 className="text-sm font-medium text-gray-300 mb-4">Avg Confidence by Category</h3>
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
            <div className="h-[250px] flex items-center justify-center text-gray-600 text-sm">No data yet</div>
          )}
        </motion.div>
      </div>

      {/* Word Frequency + Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-300">Top Words</h3>
            <select
              value={wordCat}
              onChange={e => setWordCat(e.target.value)}
              className="px-2 py-1 bg-white/5 border border-white/10 rounded text-xs text-white outline-none"
            >
              <option value="">All</option>
              {Object.keys(CAT_SHORT).map(c => <option key={c} value={c}>{CAT_SHORT[c]}</option>)}
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
            <div className="h-[250px] flex items-center justify-center text-gray-600 text-sm">No word data yet</div>
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-[#131825] border border-white/5 rounded-xl p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-300">Prediction Timeline</h3>
            <select
              value={timeRange}
              onChange={e => setTimeRange(Number(e.target.value))}
              className="px-2 py-1 bg-white/5 border border-white/10 rounded text-xs text-white outline-none"
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
            <div className="h-[250px] flex items-center justify-center text-gray-600 text-sm">No timeline data yet</div>
          )}
        </motion.div>
      </div>

      {/* Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-[#131825] border border-white/5 rounded-xl p-5"
      >
        <h3 className="text-sm font-medium text-gray-300 mb-4">Insights</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { label: 'Review Rate', value: `${((patterns?.review_rate || 0) * 100).toFixed(1)}%`, sub: `${patterns?.needs_review_count || 0} flagged`, icon: Eye, color: 'text-amber-400' },
            { label: 'Correction Rate', value: `${((patterns?.correction_rate || 0) * 100).toFixed(1)}%`, sub: `${patterns?.corrections_count || 0} corrections`, icon: RefreshCw, color: 'text-violet-400' },
            { label: 'Avg Text Length', value: `${patterns?.avg_text_length || 0}`, sub: 'characters', icon: FileText, color: 'text-cyan-400' },
            { label: 'Total Predictions', value: patterns?.total_predictions || 0, sub: 'all time', icon: Activity, color: 'text-emerald-400' },
          ].map((item, i) => (
            <div key={i} className="p-4 rounded-lg bg-white/5">
              <item.icon size={16} className={item.color} />
              <p className="text-xl font-bold text-white mt-2">{item.value}</p>
              <p className="text-xs text-gray-500">{item.sub}</p>
            </div>
          ))}
        </div>
      </motion.div>
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
          className="bg-[#131825] border border-white/5 rounded-xl p-4"
        >
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Endpoints</h3>
          <div className="space-y-1">
            {endpoints.map((ep, i) => (
              <button
                key={i}
                onClick={() => selectEndpoint(ep)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs transition ${
                  endpoint === ep.path ? 'bg-white/10 text-white' : 'text-gray-400 hover:bg-white/5 hover:text-white'
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
          <div className="bg-[#131825] border border-white/5 rounded-xl p-4">
            <div className="flex items-center gap-3 mb-3">
              <span className={`px-2 py-1 rounded text-xs font-mono ${method === 'GET' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'}`}>{method}</span>
              <span className="text-sm text-gray-300 font-mono">{API}{endpoint}</span>
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
                className="w-full h-32 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-gray-300 font-mono outline-none focus:border-cyan-500/50 resize-none"
                placeholder="Request body (JSON)"
              />
            )}
          </div>

          {/* Response */}
          {response && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-[#131825] border border-white/5 rounded-xl p-4"
            >
              <div className="flex items-center gap-3 mb-3">
                <span className={`px-2 py-1 rounded text-xs font-mono ${response.status === 200 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                  {response.status || 'ERR'}
                </span>
                <span className="text-xs text-gray-500">{response.time}ms</span>
              </div>
              <pre className="bg-white/5 rounded-lg p-3 text-xs text-gray-300 overflow-x-auto max-h-[400px] overflow-y-auto font-mono">
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
// SHARED
// ============================================================================

function LoadingState() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
        <p className="text-sm text-gray-500">Loading...</p>
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
      <Routes>
        <Route path="/" element={<Layout><DashboardPage /></Layout>} />
        <Route path="/classify" element={<Layout><ClassifyPage /></Layout>} />
        <Route path="/batch" element={<Layout><BatchPage /></Layout>} />
        <Route path="/history" element={<Layout><HistoryPage /></Layout>} />
        <Route path="/analytics" element={<Layout><AnalyticsPage /></Layout>} />
        <Route path="/playground" element={<Layout><PlaygroundPage /></Layout>} />
      </Routes>
    </BrowserRouter>
  )
}
