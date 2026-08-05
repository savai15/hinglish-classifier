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

## What's Inside

### Dashboard
- Real-time stats cards (total predictions, accuracy, needs review)
- Category distribution donut chart
- Urgency breakdown bar chart
- 24-hour prediction timeline
- Model performance bars (99.7% category, 99.96% urgency)
- Quick action buttons

### Classifier
- Type any Hinglish complaint and get instant classification
- Confidence bars for category and urgency
- Probability breakdown across all 9 categories
- 9 clickable quick samples
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
- Paginated table with timestamps
- One-click CSV export

### Analytics
- Confidence distribution histogram
- Average confidence by category (horizontal bars)
- Top word frequency (filterable by category)
- Prediction timeline (6h / 24h / 3d / 7d)
- Insights panel (review rate, correction rate, avg text length)

### API Playground
- Test all 10 endpoints interactively
- GET/POST with editable JSON body
- Response status + timing display
- Endpoint list with color-coded methods

### Search
- Ctrl+K global search across all predictions
- Fuzzy matching on complaint text
- Results with category/urgency badges

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Classify a single complaint |
| POST | `/predict/batch` | Classify multiple complaints |
| GET | `/history` | Prediction history (paginated, filterable) |
| GET | `/search?q=` | Search predictions by text |
| GET | `/stats` | Summary statistics |
| GET | `/analytics/timeline` | Predictions over time |
| GET | `/analytics/word-frequency` | Top words per category |
| GET | `/analytics/confidence` | Confidence distribution |
| GET | `/analytics/patterns` | Insights (review rate, avg length) |
| GET | `/export/csv` | Download predictions as CSV |
| POST | `/feedback` | Submit correction |
| POST | `/retrain` | Trigger model retrain |
| GET | `/categories` | List categories + colors |
| GET | `/health` | Health check |

## Tech Stack

- **Frontend:** React 19, Tailwind CSS 3, Recharts, Framer Motion, React Router
- **Backend:** FastAPI, scikit-learn, SQLite
- **Models:** TF-IDF + SVM (99.7% F1), Combined Ensemble (99.96% F1)
- **Dataset:** 30K synthetic Hinglish complaints, 9 categories, realistic typos

## Project Structure

```
hinglish-classifier/
├── api/main.py              FastAPI server (14 endpoints)
├── src/
│   ├── preprocessor.py      Hinglish text normalization + urgency cues
│   ├── models.py            sklearn pipelines + ensemble
│   ├── data_loader.py       Dataset loading + CV splits
│   ├── augment.py           30K complaint generator
│   ├── evaluation.py        Cross-validation + visualizations
│   ├── error_analysis.py    Confused pairs + per-class errors
│   ├── active_learning.py   SQLite storage + retrain manager
│   └── muril_trainer.py     MuRIL fine-tuning (GPU ready)
├── frontend/src/App.jsx     React UI (all 6 pages)
├── frontend/src/index.css   Dark theme base
├── frontend/tailwind.config.js  Color palette + animations
├── models/                  Trained .pkl files
├── data/raw/                30K dataset
├── main.py                  Training pipeline
├── start.bat                One-click launcher
└── requirements.txt         Python dependencies
```

## License

Educational project.
