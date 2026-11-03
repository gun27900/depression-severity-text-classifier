# Depression Severity Text Classifier v3

Full-stack portfolio version of an explainable NLP text-classification project.

## Architecture

- **Frontend:** Next.js 16 + React + TypeScript
- **Backend:** FastAPI + Python
- **ML:** TF-IDF (Unigram + Bigram) + Balanced Logistic Regression
- **Explainability:** local feature contribution from `TF-IDF value × Logistic Regression coefficient`
- **Deployment target:** Vercel Services

```text
Browser
  ↓
Next.js frontend
  ↓  POST /api/predict
FastAPI backend
  ↓
scikit-learn Pipeline
  ↓
class scores + clarity + feature contribution
```

## Local development

### Recommended: Vercel Services

Install the latest Vercel CLI:

```bash
npm install -g vercel@latest
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

Then run both services from the project root:

```bash
vercel dev -L
```

Open the URL printed by Vercel, normally `http://localhost:3000`.

### Test the Python backend directly

From `backend/`:

```bash
python -m pip install -e .
fastapi dev main.py
```

Then test:

```text
GET  http://localhost:8000/api/health
POST http://localhost:8000/api/predict
```

## API

### `POST /api/predict`

Request:

```json
{
  "text": "For the past few weeks, I have lost interest in things I used to enjoy and it is becoming difficult to focus on my work."
}
```

Response includes:

- four class scores
- top class
- top-two margin
- prediction clarity
- supporting features
- score-reducing features

## Evaluation snapshot

- Clean records: **3,534**
- 5-Fold CV Macro F1: **44.24%**
- Holdout Accuracy: **70.58%**
- Holdout Macro F1: **45.60%**

## Important boundary

This application is an educational / portfolio prototype. Model scores are not medical probabilities and the system is not a diagnostic or emergency-risk tool.

## Deploy to Vercel

This repository uses `vercel.json` Services configuration. Import the repository into Vercel or run:

```bash
vercel link
vercel deploy --prod
```

Vercel builds `frontend/` as Next.js and `backend/` as FastAPI, then routes `/api/*` to the backend and all other paths to the frontend.
