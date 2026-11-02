# Solution Architecture

**In one line:** with the business owner's permission, we collect their bank, GST, payroll and UPI data, turn it into a health score and a loan offer, explain the result, and serve it through an API and a dashboard.

## 1. End-to-end view

The system has five parts. Read the diagram left to right.

```mermaid
flowchart LR
    subgraph Sources["1. Data sources"]
        BANK["Bank statements"]
        GST["GST returns"]
        UPI["UPI payments"]
        PF["Payroll (EPFO)"]
        INV["Invoices"]
        CB["Credit bureau<br/>(empty for new borrowers)"]
    end
    subgraph Ingest["2. Data collection"]
        CON["Consent<br/>(owner's permission)"]
        FETCH["One connector<br/>per source"]
        CHK["Checks"]
    end
    subgraph Core["3. Scoring engine"]
        FE["Features<br/>(ratios)"]
        ML["Models<br/>risk · score · 6 pillars"]
        AN["Fraud checks"]
        WHY["Explanations<br/>(SHAP)"]
        POL["Credit policy<br/>(loan offer)"]
    end
    subgraph Serve["4. Delivery"]
        API["REST API"]
        DB[("Database")]
        DASH["Dashboard"]
    end
    USERS["5. Users<br/>Cars24 app · lenders · credit officers"]

    Sources --> FETCH
    CON --> FETCH --> CHK --> DB
    DB --> FE --> ML --> WHY
    FE --> AN
    ML --> POL
    AN --> POL
    WHY --> API
    POL --> API
    API --> DASH
    API --> USERS
```

| Part | What it does | Where in the code |
|---|---|---|
| 1. Data sources | Where the business's financial data lives | Simulated by the 50k dataset |
| 2. Data collection | Ask permission, fetch each source, check the values | `api/main.py`, `connectors.py`, `schemas.py` |
| 3. Scoring engine | Build features, predict risk, check for fraud, explain, make the loan offer | `src/msme_health/` |
| 4. Delivery | Save results and serve them to apps and people | `api/main.py`, `store.py`, `dashboard/app.py` |
| 5. Users | The Cars24 app, partner lenders, credit officers | Through the API and dashboard |

**Key design choice:** training, the API and the dashboard all use the **same scoring code**. A feature is calculated the same way when the model is trained and when a live business is scored, so results never differ between the two.

## 2. Data collection (ingestion)

**Four steps: permission → fetch → check → store.**

```mermaid
sequenceDiagram
    participant B as Business owner
    participant APP as Cars24 app
    participant API as Our API
    participant S as Data sources (bank, GST, EPFO, UPI)
    B->>APP: Applies for a loan
    APP->>API: Ask for consent
    API->>B: "Can we read your bank and GST data for 12 months?"
    B-->>API: Approves
    API->>S: Fetch data (with the consent)
    S-->>API: Data
    Note over API: Check values → merge → save → log
    APP->>API: Score this business
    API-->>APP: Health Card + loan offer (~0.5 second)
```

| Step | What happens | In the demo | In production |
|---|---|---|---|
| 1. Permission | The owner approves which data, for what purpose and for how long. No permission, no data. | `POST /consents` creates a consent record | The owner approves in an RBI-regulated consent app (Account Aggregator) |
| 2. Fetch | Pull data from each source | `POST /msmes/{id}/fetch`: one connector per source, reading from the dataset | The same connectors, connected to the real sources |
| 3. Check | Reject impossible values (e.g. a percentage over 100) | Automatic checks on every field | Plus cross-checks between sources |
| 4. Store | Save the record and log every step | SQLite database + audit log | A production database (Postgres) |

A source can also **send its data on its own** (`POST /msmes/{id}/sources/{source}`), for example when new GST returns are filed.

**The API enforces consent:** fetching without a valid consent, or with another business's consent, is refused.

## 3. AI/ML workflow

**Seven steps: explore → build features → split → train → test → explain → score everyone.**

```mermaid
flowchart LR
    D["50k businesses"] --> EDA["1. Explore<br/>(notebook 01)"] --> FEAT["2. Features<br/>(notebook 02)"]
    FEAT --> SPLIT["3. Split<br/>70% train · 15% tune · 15% test"]
    SPLIT --> M["4. Train<br/>simple model vs LightGBM"]
    M --> EVAL["5. Test<br/>on unseen data"]
    EVAL --> SHAP["6. Explain<br/>(SHAP)"]
    SHAP --> BATCH["7. Score all 50k"]
```

| Choice | What we did | Why, in plain words |
|---|---|---|
| What we predict | Probability of default (PD), the 0–100 health score, the 6 pillar scores | PD is what lenders decide on; the score and pillars are easy for people to read |
| Main model | LightGBM (many small decision trees) | Strong on table data, handles blanks, fast |
| Common-sense rules | "Monotonic constraints" on 14 features | Better behaviour can never give a worse score (e.g. filing GST on time can never raise risk) |
| Comparison model | Ridge (a simple straight-line model, like a bank scorecard) | Proves the complex model is worth it. Here both perform about the same. |
| Explanations | SHAP: how many points each factor added or removed | Every decision can be explained; decline reasons come from it |
| Fraud checks | 8 rules + Isolation Forest (flags the most unusual 2%) | Rules catch known tricks; the forest catches unusual patterns |
| Loan decision | A separate policy: risk band → eligible? → limit → conditions | The risk team can change rules without retraining the model |

**In production:** retrain regularly. Watch whether incoming data starts to look different from the training data, and only replace the model when a new one beats it on real repayment data.

## 4. Integration strategy

**Two questions: how does data come IN, and how does the result go OUT?**

### Data in

| Source | What it is | What we use from it |
|---|---|---|
| **Bank statements** (through an **Account Aggregator**) | RBI's consent system: the owner taps "approve" in an app and their bank shares statements | Money in and out, balance, overdraft use, EMIs |
| **GST portal** | Monthly tax returns, shared with the owner's OTP | Sales, purchases, whether GST was filed on time |
| **EPFO** | Monthly provident-fund filings by employers | Employees, salary total, salary paid on time |
| **UPI provider** | The business's payment app data | Daily transactions, payment size, UPI income |
| **GST e-invoices** | Invoices the business raises and receives | How fast customers pay, dependence on top customers, paying suppliers on time |
| **Credit bureau** (e.g. CIBIL) | Loan history | EMI record. **Optional**: empty for new-to-credit businesses, and the model works without it |
| **Business registration** (Udyam) | The MSME's official registration | Industry, business type, age, location |

**In the code:** every source sits behind **one common "connector" design** (`connectors.py`). The demo uses mock connectors that read from the dataset. Going live means replacing each mock with a real connector. **The model and the API do not change.** Adding a new source later is just one more connector.

### Result out

| Who | How they use it |
|---|---|
| **Cars24 app** | Calls the API when a dealer or fleet owner applies; shows the offer in seconds |
| **Partner lenders** | Cars24 can send the Health Card along with the loan application, so several lenders can make offers |
| **Credit officers** | Use the dashboard to review a business, its reasons and fraud flags |

Everything goes through the **REST API**, so any system in any language can use it.

## 5. Scaling up

### Measured (laptop, this repo)

| What | Time / size |
|---|---|
| Full card (8 models + SHAP + what-if), first view | 0.45 s (95% under 0.52 s) |
| Same card viewed again (saved) | 7 ms |
| Score only, no explanations | ~0.1 s (PD model alone: 22 ms) |
| Portfolio dashboard numbers | 13 ms |
| Re-score all 50,000 businesses | 5.7 s |
| Memory per API copy | ~350 MB |
| One saved card | ~9 KB |

### How we scale (high level)

1. **Load balancer + autoscaling.** Run many copies of the API behind a load balancer, which sends each request to a free copy. Autoscaling adds copies when traffic rises and removes them when it drops. This works because the API keeps no memory between requests.
2. **Cache.** Save each computed card in a fast store (e.g. Redis). A repeat request returns in 7 ms instead of 0.45 s. When new data arrives for a business, clear its saved card.
3. **Decision first, explanations later.** Return the score and loan offer in about 0.1 s. Compute SHAP explanations in the background, since they take about 80% of the time.
4. **Queue for slow or bursty work.** Explanations, re-scoring on new GST or bank data, and fetching from data sources go through a queue (e.g. Kafka or SQS). During a spike, jobs wait safely instead of crashing the system.
5. **Rate limits per partner.** An API gateway checks API keys and limits how many requests each partner can send, so one faulty caller cannot slow down everyone.
6. **Features prepared in advance.** Fetch bank and GST data when consent is given and store the ready features. Real-time scoring then only reads features and runs the model.
7. **Score in small batches.** Scoring together costs about 0.1 ms per business (50k in 5.7 s), against about 0.1 s one at a time. Grouping requests for a few milliseconds multiplies what each copy can handle.
8. **One shared Postgres.** It replaces SQLite, so every copy sees the same consents, cards and audit log. This is the first thing to change.
9. **Nightly batch** for re-scoring the whole portfolio, and **versioned models** that can be rolled back.
10. **Monitoring.** Track response time, errors, requests per second and queue length, and alert when they go past limits.

## 6. Security and compliance

- **Consent first:** no data is fetched without the owner's permission, and data is used only for the stated purpose (India's data-protection law, DPDP Act 2023).
- **Explainable decisions:** every decline has reasons, as RBI's fair-lending rules expect.
- **Audit log:** every consent, fetch, score and decision is recorded.
- **Fairness:** accuracy is checked separately for new-to-credit, new-to-bank and existing borrowers. No personal attributes (gender, religion and so on) are used.
- **Production:** encrypt data, mask personal details, delete data when consent expires.

## 7. Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data | pandas, NumPy |
| ML | LightGBM, scikit-learn (Ridge, Isolation Forest), SHAP |
| API | FastAPI (auto-generated docs at `/docs`) |
| Dashboard | Streamlit + Plotly |
| Storage | SQLite (demo) → Postgres (production) |
| Notebooks | Jupyter |
| Testing | pytest (19 tests) |
| Deployment | Render, Docker |
