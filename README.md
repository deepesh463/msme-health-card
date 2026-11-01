# Cars24 · MSME Financial Health Card

An explainable credit-assessment system for **new-to-credit (NTC)** and **new-to-bank (NTB)** small businesses. It turns consent-based alternate data into a Financial Health Card and a credit decision:
- **Data sources:** GST, UPI, Account Aggregator bank data, EPFO payroll, invoices.
- **Card:** a 0–100 score, 6 pillar scores, risk flags, reasons and improvement actions.
- **Decision:** approve / decline, limit, tenure, rate and conditions.

> **Live demo:** [Dashboard](https://msme-health-dashboard.onrender.com/) · [API docs](https://msme-health-api-yx6e.onrender.com/docs) · [Slides (PDF)](docs/MSME_Health_Card_deck.pdf)
> Free hosting: if a link has been idle, the first load takes about a minute while it wakes up.

---

## Why it matters

42% of the 50,000 businesses in the dataset are NTC: no loan, no bureau history. A bureau-only lender rejects all of them. On alternate data, **about 73% of NTC businesses are not high-risk**, nearly the same as existing borrowers (78%). This system finds them, explains why, and sizes a loan they can afford.

**Cars24 angle:** Cars24's business customers include used-car dealers and fleet operators. As an OCEN loan service provider, Cars24 could use the Health Card to offer dealer inventory finance and commercial-vehicle loans to businesses that banks currently can't assess.

## What's inside

| Deliverable (from the brief) | Where |
|---|---|
| 1. Solution architecture, ingestion pipeline, AI/ML workflow, integration, tech stack | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| 2. Financial Health Card: score, 6 pillars, risk indicators, insights | Dashboard → *MSME Health Card*; `GET /api/v1/msmes/{id}/health-card` |
| 3. AI/ML model: health prediction, credit eligibility, anomaly detection, SHAP | [`src/msme_health/`](src/msme_health), [`notebooks/03_modelling.ipynb`](notebooks/03_modelling.ipynb) |
| 4. Dashboard: score, revenue, cash flow, GST, UPI, payroll, risk, recommendation, explainability | [`dashboard/`](dashboard) (5 pages) |
| 5. REST APIs: ingestion, scoring, card, recommendation, dashboard | [`api/main.py`](api/main.py), OpenAPI at `/docs` |
| 6. Integration strategy: ULI, OCEN, AA, GSTN, banking APIs | [`docs/ARCHITECTURE.md §4`](docs/ARCHITECTURE.md#4-integration-strategy) |
| Research: EDA, features | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb), [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb), [`docs/FEATURE_DICTIONARY.md`](docs/FEATURE_DICTIONARY.md) |

## Results (held-out test set, 7,500 businesses)

| Model | Default-prob. R² | Risk-band accuracy | Eligibility accuracy | High-risk AUC | Gini* |
|---|---|---|---|---|---|
| Ridge baseline (scorecard-style) | 0.918 | 83.2% | 92.4% | 0.979 | 0.468 |
| LightGBM + monotonic constraints | 0.912 | 82.9% | 92.4% | 0.977 | 0.466 |
| *Ceiling: the supplied PD itself* | 1.000 | 100% | 100% | 1.000 | 0.483 |

\* The data has no real defaults. Gini/KS are measured on defaults **simulated** from the supplied PD, so 0.483 is the best any model can reach.

**Key findings**
- The supplied scores are near-linear formulas plus random noise. Even an "oracle" model given the true pillar scores reaches the same R² (0.876) on the health score. **We are at the noise ceiling.** The simple baseline matches LightGBM.
- **No Low↔High mistakes:** every band error is one step (Low↔Medium or Medium↔High).
- **Works for NTC:** accuracy for NTC businesses (R² 0.912, AUC 0.979) equals accuracy for existing borrowers.
- **Prudent limits:** an affordability check (EMI ≤ 50% of monthly surplus) lowers the limit for about 12% of approved businesses, cutting total exposure by about 7% versus the supplied policy.
- **Fraud checks:** cross-source rules catch 100% of planted circular-transaction, inflated-GST and hidden-debt cases, with 0.65% false alarms.

## How the score works (in one minute)

1. **Data in:** 33 business inputs from 6 sources (GSTN, UPI, AA bank data, bureau, EPFO, invoices).
2. **Features:** 11 analyst ratios. Examples:
   - cash surplus margin;
   - months of expenses covered by the bank balance;
   - EMI burden;
   - cross-source gaps (GST vs bank);
   - a **lean-month stress test** from seasonality.
3. **Models:**
   - default probability (log-odds), the overall score and each pillar;
   - LightGBM with monotonic constraints, so filing GST late can never *improve* a score.
4. **Decision:**
   - risk band from PD (< 6% Low, 6–15% Medium, > 15% High);
   - limit = lower of 16%/8% of turnover and what the cash surplus can repay;
   - seasonal stress, data flags and thin files add conditions or send the case to review.
5. **Explain:**
   - SHAP waterfall in score points;
   - reason codes for a decline;
   - peer benchmarks within industry × location;
   - "what would improve this score" actions, re-scored by the model.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements-dev.txt
pip install -e .
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

API (Swagger UI at http://localhost:8000/docs):

```bash
uvicorn api.main:app --reload
```

Tests:

```bash
pytest -q
```

Retrain from scratch (≈ 20 s), then re-score the portfolio:

```bash
python -m msme_health.train
python -m msme_health.engine
```

Rebuild the executed notebooks:

```bash
python tools/build_notebooks.py
```

## API at a glance

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/consents` | Create an (AA-style) consent artefact |
| POST | `/api/v1/msmes/{id}/fetch` | Pull all sources through connectors (mocked) |
| POST | `/api/v1/msmes/{id}/sources/{source}` | Push one source's data (gstn, upi, account_aggregator, epfo…) |
| POST | `/api/v1/msmes` | Ingest a full consolidated record |
| POST | `/api/v1/score` | **Real-time**, stateless: data in, full Health Card out |
| POST | `/api/v1/msmes/{id}/score` | Score a stored business; saves the card |
| GET | `/api/v1/msmes/{id}/health-card` | Full card: scores, pillars, SHAP, risks, insights, peers |
| GET | `/api/v1/msmes/{id}/credit-recommendation` | Decision, limit, tenure, rate, EMI, conditions, reason codes |
| GET | `/api/v1/dashboard/portfolio` | Portfolio KPIs (filter by segment / industry) |
| GET | `/api/v1/dashboard/msmes` | Paginated, filterable list of scored businesses |

## Deploy on Render

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, then select the repo. [`render.yaml`](render.yaml) creates two free web services: the dashboard and the API.
3. Paste the two URLs into the "Live demo" line above.

The dashboard uses about 350 MB of memory, comfortably within Render's free tier (512 MB). Explanations use LightGBM's built-in TreeSHAP, so the `shap` library is only needed for training and notebooks. Free services sleep when idle, so the first request takes about a minute.

## Repository layout

```
api/                 FastAPI app
dashboard/           Streamlit app (app.py + pages/)
src/msme_health/     config · features · train · engine · recommend · anomaly · peers · connectors · store · schemas
artifacts/           trained models, metrics.json, scored portfolio (parquet)
notebooks/           01_eda · 02_feature_engineering · 03_modelling (executed); _src/ holds their sources
docs/                ARCHITECTURE.md · FEATURE_DICTIONARY.md
tests/               API flow + behavioural tests
data/raw/            supplied dataset (50k synthetic MSMEs)
```

## Limitations and next steps

- **No observed defaults.** The models learn the supplied scoring, not real repayment. With real outcomes:
  - train a classifier on 90+ days-past-due within 12 months;
  - validate it out-of-time;
  - run Ridge as champion and LightGBM as challenger.
- **Snapshot data.** There is no monthly history, so "trends" are shown as peer comparisons and a seasonal lean / average / peak profile. With AA and GST time series, add rolling features: 3/6/12-month trends, bounce counts, balance dips.
- **Mocked connectors.** Real AA, GSTN and EPFO clients plug in behind the same interface.
- **Drift and monitoring** (PSI, score stability, approval rates by segment) are designed but not built.
