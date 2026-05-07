# SkillSevak — Understand Talent Beyond Keywords

> An AI-powered hiring platform that matches candidates to jobs using a custom-trained Artificial Neural Network — not keyword filters.

---

## The Problem

Traditional job platforms rank resumes by keyword overlap. A candidate who has built production ML pipelines gets filtered out because they wrote "PyTorch" instead of "TensorFlow". A recruiter drowns in 200 applications, most irrelevant.

**SkillSevak fixes this.** It understands *what a person can do*, not just what words appear on their resume.

---

## Live Demo

| Role | URL |
|------|-----|
| Candidate | `/candidate` — upload resume, get ranked job matches |
| Recruiter | `/recruiter` — post jobs, view ranked applicants |

Sign in with Google OAuth or create an account.

---

## How It Works

```
Resume (PDF/DOCX)
      │
      ▼
┌─────────────────────┐
│   Resume Parser     │  pdfplumber / python-docx
│   + Skill Extractor │  spaCy NLP (zero hardcoded skills)
└─────────┬───────────┘
          │  raw text + skills
          ▼
┌─────────────────────┐
│  Feature Extractor  │  5 numerical features (0–1)
│                     │  • Semantic similarity  (sentence-transformers)
│                     │  • Skill overlap score
│                     │  • Experience match
│                     │  • Education match
│                     │  • Profile completeness
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  ANN Match Scorer   │  Custom PyTorch model
│  5 → 128 → 64 →     │  Trained on 11,959 samples
│  32 → 1  (0–100)    │  RMSE: 7.19 | Accuracy@10%: 83.5%
└─────────┬───────────┘
          │
          ▼
   Match Score + Ranked Results
```

---

## Key Features

### For Job Seekers
- Upload resume (PDF or DOCX) — parsed and understood instantly
- See a ranked list of jobs with a **0–100 match score** per role
- Understand *why* you match: skill gaps and strengths surfaced
- Passive talent pool — get discovered by recruiters even without applying

### For Recruiters
- Post jobs in plain English — no rigid keyword forms
- Get applicants ranked by AI match score, not application time
- One-click view of candidate skills vs. job requirements
- Seed demo candidates to test the platform

### Under the Hood
- **Zero hardcoded skills** — spaCy extracts skills dynamically from any resume
- **Semantic understanding** — `all-MiniLM-L6-v2` embeddings capture meaning, not just words
- **Custom ANN** — trained from scratch, not a wrapped API
- **Async processing** — Celery + Redis handles heavy ML jobs in the background
- **Score range is calibrated**: Poor ≈ 4, Average ≈ 41, Strong ≈ 85, Excellent ≈ 93

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Django 5.2 (Python) |
| ML Model | PyTorch — custom `MatchPredictor` ANN |
| NLP | spaCy `en_core_web_sm` + `sentence-transformers` |
| Resume Parsing | pdfplumber, python-docx |
| Async Tasks | Celery + Redis |
| Auth | Google OAuth (django-allauth) |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Containerization | Docker + docker-compose |
| Frontend | Django templates + Tailwind CSS |

---

## ANN Model Details

The heart of SkillSevak is a custom neural network trained entirely from scratch.

**Architecture** (`ann/ml/model.py`):
```
Input (5 features)
  → Linear(5 → 128) + ReLU + LayerNorm + Dropout(0.15)
  → Linear(128 → 64) + ReLU + LayerNorm + Dropout(0.10)
  → Linear(64 → 32)  + ReLU + LayerNorm
  → Linear(32 → 1)   + Sigmoid
Output × 100  →  Match Score (0–100)
```

**Training data** (`ann/ml/train.py`):
- 10,000 synthetic samples with calibrated score distributions
- 1,959 real-world samples from HuggingFace `cnamuangtoun/resume-job-description-fit`
- Loss: `MSELoss` | Optimizer: `Adam(lr=1e-3, weight_decay=1e-4)` | Scheduler: `CosineAnnealingLR`

**Performance**:
| Metric | Value |
|--------|-------|
| RMSE | 7.19 |
| MAE | 5.58 |
| Accuracy within ±10 pts | **83.5%** |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/rishiraj-21/N26_14-SkillSevak.git
cd N26_14-SkillSevak

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set SECRET_KEY, USE_SQLITE=True for local dev
```

### 3. Run migrations & seed data

```bash
python manage.py migrate
python manage.py populate_jobs          # seed sample job listings
python manage.py seed_demo_candidates   # seed demo candidates (optional)
```

### 4. Train the model (optional — pretrained weights not in repo)

```bash
python manage.py train_model
```

### 5. Start the server

```bash
python manage.py runserver
```

Visit `http://localhost:8000`

### Docker (optional)

```bash
docker-compose up --build
```

---

## Project Structure

```
neuralnetwork/
├── ann/
│   ├── ml/
│   │   ├── model.py          # ANN architecture (PyTorch)
│   │   ├── train.py          # Training loop + data generation
│   │   ├── text_features.py  # 5-feature extractor from raw text
│   │   └── inference.py      # Score prediction at runtime
│   ├── services/
│   │   ├── matching_engine.py   # Core match orchestrator
│   │   ├── skill_extractor.py   # spaCy-based dynamic skill extraction
│   │   ├── resume_parser.py     # PDF/DOCX parsing
│   │   └── embedding_service.py # sentence-transformers wrapper
│   ├── management/commands/
│   │   ├── train_model.py        # CLI: train ANN
│   │   ├── load_external_data.py # CLI: download HuggingFace dataset
│   │   ├── populate_jobs.py      # CLI: seed job listings
│   │   └── seed_demo_candidates.py
│   ├── templates/            # Django HTML templates (Tailwind)
│   ├── models.py             # Django ORM models
│   ├── views.py              # Candidate + Recruiter views
│   └── tasks.py              # Celery async tasks
├── neuralnetwork/
│   ├── settings.py
│   └── celery.py
├── requirements.txt
├── docker-compose.yml
└── .env.example
```

---

## What Makes This Different

| Feature | Keyword-based ATS | SkillSevak |
|---------|-------------------|------------|
| Skill detection | Exact match only | NLP extraction, understands synonyms |
| Scoring | Binary pass/fail | Continuous 0–100 ANN score |
| Model | None / rule-based | Custom trained neural network |
| Resume understanding | Keyword scan | Semantic embedding similarity |
| Transparency | Black box filter | Per-feature breakdown |
| Async processing | No | Celery + Redis background jobs |

---


---

## Hackathon 

Built for **N26_14** — a fully functional AI hiring platform with a custom-trained neural network, end-to-end resume understanding, and dual-role UX (candidate + recruiter), developed from scratch.


---

*SkillSevak — Understand Talent Beyond Keywords*
