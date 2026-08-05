# Hinglish Complaint Classifier

Routes Hinglish e-commerce complaints into 9 categories and flags urgency — built for Indian support teams drowning in "mera order nahi aaya" messages.

**Stack:** React + Tailwind (dark UI) | FastAPI backend | scikit-learn + MuRIL | SQLite feedback loop

## Quick Start

```
git clone https://github.com/savai15/test.git
cd test
start.bat
```

`start.bat` handles everything — checks Python/Node, installs dependencies, launches both servers, opens browser. Press any key to stop.

**Requires:** Python 3.10+, Node.js 18+

## Results

| Task | Model | F1 Score |
|------|-------|----------|
| Category (9 classes) | TF-IDF + SVM | **99.7%** |
| Urgency (3 classes) | Combined ensemble | **99.96%** |

Trained on 30K synthetic Hinglish complaints with realistic typos, code-switching, and slang.

### Categories
Account Technical, Customer Service, Delivery Issue, Order Status, Payment/Invoice, Pricing/Discount, Product Quality, Returns/Refunds, Wrong/Damaged Product

### Urgency Levels
High (threats, escalation), Medium (firm complaints), Low (general inquiries)

## Architecture

```
Browser (React :5173)
    |
    | POST /predict
    v
FastAPI (:8000)
    |
    |---> Preprocessor (Hinglish normalization + urgency cues)
    |---> sklearn models (TF-IDF + SVM, ensemble)
    |---> SQLite (stores predictions for active learning)
    |
    v
Response: category, urgency, confidence scores
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Classify a complaint |
| POST | `/predict/batch` | Classify multiple complaints |
| GET | `/history` | Recent prediction history |
| GET | `/stats` | Category/urgency distribution |
| POST | `/feedback` | Submit correction (triggers retrain) |
| GET | `/categories` | List all categories |
| GET | `/health` | Server health check |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "mera order 3 din se nahi aaya, delivery boy phone nahi utha raha"}'
```

### Example Response
```json
{
  "category": "Delivery_Issue",
  "category_confidence": 0.9949,
  "urgency": "Low",
  "urgency_confidence": 0.9139,
  "source": "sklearn",
  "needs_review": false
}
```

## Using the Python API Directly

```python
from src.preprocessor import HinglishPreprocessor
from src.models import load_model

preprocessor = HinglishPreprocessor.load("models/preprocessor.pkl")
cat_model = load_model("models/category_ensemble.pkl")
urg_model = load_model("models/urgency_ensemble.pkl")

text = "Consumer court jaunga agar refund nahi mila!"
cleaned = preprocessor.preprocess(text)

print(cat_model.predict([cleaned])[0])  # Returns_Refunds
print(urg_model.predict([cleaned])[0])  # High
```

## Training

### Retrain sklearn models (CPU, ~30 sec)
```bash
python main.py
```
Loads 30K dataset, runs 5-fold CV, tunes hyperparameters, saves models to `models/`.

### Fine-tune MuRIL on GPU (~15 min)
```bash
# First, uncomment torch/transformers in requirements.txt and re-run start.bat
python -m src.muril_trainer
```
Fine-tunes `google/muril-base-cased` (Indian language model) on your dataset. Needs ~2GB VRAM.

## Project Structure

```
project/
├── api/
│   └── main.py              FastAPI server (9 endpoints)
├── src/
│   ├── preprocessor.py      Hinglish text normalization + urgency cues
│   ├── models.py            sklearn pipelines + ensemble
│   ├── data_loader.py       Dataset loading + CV splits
│   ├── augment.py           30K complaint generator (9 categories)
│   ├── evaluation.py        Cross-validation + visualizations
│   ├── error_analysis.py    Confused pairs + per-class errors
│   ├── active_learning.py   SQLite storage + retrain manager
│   └── muril_trainer.py     MuRIL fine-tuning pipeline
├── frontend/
│   ├── src/App.jsx          React UI (single file, all components)
│   ├── src/index.css        Dark theme + aurora gradient
│   ├── tailwind.config.js   Midnight Aurora color palette
│   └── package.json         React + Framer Motion + Axios
├── models/                  Trained .pkl model files
├── data/raw/                30K Hinglish complaint dataset
├── reports/                 Confusion matrices + error analysis
├── main.py                  End-to-end training pipeline
├── start.bat                One-click launcher (auto-installs deps)
├── run.py                   Python launcher (backup)
└── requirements.txt         Python dependencies
```

## Preprocessing

The preprocessor handles Hinglish-specific patterns that break standard NLP:

1. **Urgency cue detection** — extracts CAPS, threats ("consumer court"), escalation ("manager"), high amounts before cleaning
2. **Spelling normalization** — `nahi/nai/nahee` → `nahin`, `urgenttt` → `urgent`
3. **Token injection** — `URGENTCAPS`, `THREAT`, `TIMEPRESSURE`, `HIGHAMOUNT` tokens appended
4. **Code-mixing handling** — Hindi stopwords removed alongside English, short word filtering

## License

Educational project.
