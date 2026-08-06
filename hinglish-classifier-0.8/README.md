# HinglishAI

Classifies Hinglish e-commerce complaints into 9 categories and 3 urgency levels. Built for Indian support teams drowning in "mera order nahi aaya" messages.

## Quick Start

```
git clone https://github.com/savai15/hinglish-classifier.git
cd hinglish-classifier
start.bat
```

`start.bat` handles everything — checks Python/Node, installs all dependencies (first time takes ~2 min), kills old processes, launches both servers, opens browser automatically. Press any key to stop.

**Requires:** Python 3.10+, Node.js 18+

**Manual start:**
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
python -m uvicorn api.main:app --reload --port 8000
cd frontend && npm run dev
```

## AI Integration (Optional)

The AI Assistant page uses **Groq** (free, fast) to generate resolution steps and draft responses. To enable:

1. Get a free API key at [console.groq.com](https://console.groq.com) (sign up → API Keys → Create)
2. Set it as an environment variable before starting the server:

**Windows (temporary — current terminal only):**
```cmd
set GROQ_API_KEY=gsk_your_key_here
```

**Windows (permanent — survives restart):**
```cmd
setx GROQ_API_KEY "gsk_your_key_here"
```
Then restart your terminal for it to take effect.

**Linux/Mac:**
```bash
export GROQ_API_KEY=gsk_your_key_here
```

**Or in start.bat:** Open `start.bat` and add this line before the server launch:
```cmd
set GROQ_API_KEY=gsk_your_key_here
```

Without the key, the AI Assistant page shows an error. All other features (classify, batch, analytics, feedback, retrain) work without it.

## What's Inside

### Dashboard
- Real-time stats cards (total predictions, accuracy, needs review)
- Category distribution donut chart
- Urgency breakdown bar chart
- 24-hour prediction timeline
- Model performance bars (Category F1=0.9969, Urgency F1=0.9996)
- Retrain history timeline
- Quick action buttons

### Classifier
- Type any Hinglish complaint and get instant classification
- Confidence bars for category and urgency
- Probability breakdown across all 9 categories
- 9 clickable quick samples
- **Feedback panel** — mark predictions as correct/incorrect, submit corrections
- **Retrain banner** — triggers model retrain when 20+ corrections collected
- Recent predictions sidebar
- Ctrl+Enter shortcut to classify

### Batch Processing
- Upload CSV file for bulk classification
- Or paste complaints (one per line)
- Progress bar during processing
- Export results as CSV or JSON
- Review confidence flags per result

### History
- Full searchable prediction log
- Filter by category, urgency, or text search
- **Correction status column** (green check / red X)
- **"Corrected only" filter**
- Paginated table with timestamps
- One-click CSV export

### Analytics
- Confidence distribution histogram
- Average confidence by category (horizontal bars)
- Top word frequency (filterable by category)
- Prediction timeline (6h / 24h / 3d / 7d)
- Insights panel (review rate, correction rate, avg text length)
- **Smart Suggestions** — actionable recommendations based on prediction patterns

### AI Assistant
- Paste a complaint → get resolution steps or draft a customer response
- Powered by Groq (llama-3.3-70b-versatile)
- Side-by-side results with copy-to-clipboard
- Optional category/urgency hints

### Review Queue
- Low-confidence predictions flagged for manual review
- Quick feedback (correct/incorrect) on each item
- Refresh to pull latest

### Model Comparison
- Load and compare individual sklearn models side-by-side
- View category and urgency predictions per model
- Confidence breakdowns with consensus detection
- Compare TF-IDF+SVM, TF-IDF+LR, and Ensemble models

### Settings
- **Tenant Branding** — customize company name and accent color
- Dark/light mode toggle
- Keyboard shortcuts overview

### API Playground
- Test all endpoints interactively
- GET/POST with editable JSON body
- Response status + timing display

### Search
- Ctrl+K global search across all predictions
- Fuzzy matching on complaint text
- Results with category/urgency badges

## Features

### Dark/Light Mode
- Toggle between themes via the Sun/Moon button in the top bar
- Preference persisted in localStorage
- All components adapt with Tailwind `dark:` prefix

### Keyboard Shortcuts
- Press `?` to open the shortcuts modal
- `G` then `D` — Dashboard
- `G` then `C` — Classify
- `G` then `B` — Batch
- `G` then `H` — History
- `G` then `A` — Analytics
- `G` then `I` — AI Assistant
- `G` then `R` — Review Queue
- `G` then `P` — Compare
- `Ctrl+Enter` — Classify from text input
- `Ctrl+K` — Global search

### PDF Report Export
- Export page generates a full HTML report
- Stats, distributions, model performance, recent predictions
- Open in new tab → Ctrl+P to save as PDF

### Onboarding Wizard
- 4-step walkthrough on first visit: Welcome → Classify → Batch → AI
- Skip or navigate through steps
- Persistence in localStorage (only shows once)

### Error Boundaries
- Each page wrapped in ErrorBoundary
- Crashes show a fallback UI instead of white screen
- Reset button to return to Dashboard

### Loading Skeletons
- Skeleton placeholders on Dashboard, History, Analytics, Review Queue
- Replace spinners for a professional feel

## Production Features

### Rate Limiting
- All endpoints rate-limited via slowapi
- `/predict`: 30 requests/min
- `/predict/batch`: 10 requests/min
- `/ai/*`: 10 requests/min
- `/feedback`: 60 requests/min
- `/retrain`: 5 requests/hour
- `/health`, `/retrain/status`: exempted

### Input Validation
- Pydantic models with `Field(min_length=1, max_length=5000)` on text
- `Literal` types for category and urgency parameters
- Max batch size: 100 complaints

### Structured Logging
- Python `logging` module with `INFO` level
- Request IDs (`X-Request-ID` header, 8-char UUID)
- Request timing middleware (logs response time)
- All `print()` statements replaced with `logger.info()`

### Health Check
- `GET /health` checks:
  - DB connectivity (SQLite write/read test)
  - Model load status (sklearn_preprocessor)
  - Groq API key presence
- Returns `status: "ok"` or `"degraded"` with details

### Database Performance
- 7 indexes on `predictions` table:
  - `idx_category`, `idx_urgency`, `idx_timestamp`
  - `idx_confidence`, `idx_text`, `idx_correction`, `idx_id`

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Classify a single complaint |
| POST | `/predict/batch` | Classify multiple complaints |
| GET | `/predict/compare` | Compare predictions across models |
| GET | `/history` | Prediction history (paginated, filterable) |
| GET | `/search?q=` | Search predictions by text |
| GET | `/stats` | Summary statistics |
| GET | `/analytics/timeline` | Predictions over time |
| GET | `/analytics/word-frequency` | Top words per category |
| GET | `/analytics/confidence` | Confidence distribution |
| GET | `/analytics/patterns` | Insights (review rate, avg length) |
| GET | `/analytics/suggestions` | Smart suggestions based on patterns |
| GET | `/export/csv` | Download predictions as CSV |
| GET | `/export/report` | Full HTML report for PDF export |
| POST | `/feedback` | Submit correction |
| POST | `/retrain` | Trigger model retrain |
| GET | `/retrain/status` | Retrain readiness (corrections vs threshold) |
| GET | `/retrain/history` | Retrain log with accuracy changes |
| GET | `/low-confidence` | Predictions needing review |
| POST | `/ai/resolve` | AI resolution steps (requires `GROQ_API_KEY`) |
| POST | `/ai/draft-response` | AI draft response (requires `GROQ_API_KEY`) |
| GET | `/categories` | List categories + colors |
| GET | `/health` | Health check with model/db/api status |

## Tech Stack

- **Frontend:** React 19, Tailwind CSS 3, Recharts, Framer Motion, React Router, PapaParse
- **Backend:** FastAPI, scikit-learn, SQLite, Groq (AI), slowapi (rate limiting)
- **Models:** TF-IDF + SVM Category F1=0.9969, Combined Urgency F1=0.9996
- **Dataset:** 30K synthetic Hinglish complaints, 9 categories, realistic typos
- **GPU Ready:** MuRIL fine-tuning for future improvement

## Project Structure

```
hinglish-classifier/
├── api/main.py              FastAPI server (24 endpoints)
├── src/
│   ├── preprocessor.py      Hinglish text normalization + urgency cues
│   ├── models.py            sklearn pipelines + ensemble
│   ├── data_loader.py       Dataset loading + CV splits
│   ├── augment.py           30K complaint generator
│   ├── evaluation.py        Cross-validation + visualizations
│   ├── error_analysis.py    Confused pairs + per-class errors
│   ├── active_learning.py   SQLite storage + retrain manager (7 indexes)
│   └── muril_trainer.py     MuRIL fine-tuning (GPU ready)
├── frontend/
│   ├── src/App.jsx          React UI (10 pages, dark/light mode, onboarding)
│   ├── src/index.css        Tailwind base + theme
│   ├── tailwind.config.js   Color palette + animations
│   ├── postcss.config.cjs   PostCSS config
│   └── vite.config.js       Vite config with API proxy
├── models/                  Trained .pkl files (individual + ensemble)
├── data/raw/                30K dataset
├── main.py                  Training pipeline
├── start.bat                One-click launcher
└── requirements.txt         Python dependencies
```

## License

Educational project.
