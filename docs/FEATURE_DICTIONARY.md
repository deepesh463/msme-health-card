# Feature Dictionary

What every column means, where it would come from in real life, and how it relates to risk.
The last section lists **new feature ideas** for us to discuss.

Ranges are from the 50,000-row dataset (min · median · max).
"Risk direction" says what a higher value usually means for default risk: ↑ riskier, ↓ safer, ~ unclear.

---

## 1. Business profile

| Column | Meaning | Real-world source | Range | Risk direction |
|---|---|---|---|---|
| `MSME_ID` | Unique business ID | Lender's system | — | — |
| `Customer_Segment` | **NTC**: never had a loan (no bureau record). **NTB**: new to this bank, may have loans elsewhere. **Existing-to-Credit**: has a current loan | Bureau + bank CRM | 42% / 30% / 28% | NTC slightly riskier |
| `Business_Type` | Legal form: Proprietorship, Partnership, LLP, Private Limited | Udyam / MCA21 | 48% proprietorships | Pvt Ltd usually safer (more governance) |
| `Industry` | Sector (10 values) | Udyam NIC code | Retail largest (18%) | Varies by sector cycle |
| `Location_Category` | Metro, Tier 1–3, Rural | Address / PIN code | Tier 2 largest (30%) | ~ |
| `Years_in_Operation` | Business age (vintage) | GST registration date / Udyam | 0.2 · 4.3 · 29.4 | ↓ older = safer |
| `Employee_Count` | Number of employees | EPFO | 1 · 3 · 102 | ↓ slightly |
| `Annual_Turnover_INR` | Declared yearly revenue | GST returns / ITR | ₹3 L · ₹49 L · ₹25 Cr | Sets loan size, not risk |

## 2. GST (from GSTN)

| Column | Meaning | How it's computed in practice | Range | Risk |
|---|---|---|---|---|
| `Monthly_GST_Sales_INR` | Average monthly taxable sales | GSTR-1 / GSTR-3B outward supplies | ₹20 K · ₹4 L · ₹2.3 Cr | Size |
| `Monthly_GST_Purchases_INR` | Average monthly taxable purchases | GSTR-2B inward supplies (input tax credit) | ₹7 K · ₹2.5 L · ₹1.5 Cr | Size |
| `GST_Filing_Timeliness_Pct` | % of returns filed by the due date | Filing date vs due date over last 12–24 returns | 50 · 84 · 100 | ↓ **(strongest single driver)** |
| `GST_Return_Frequency` | Monthly or Quarterly (QRMP scheme for turnover < ₹5 Cr) | GSTN profile | 82% monthly | ~ quarterly = less visibility |
| `Seasonality_Index` | How much sales swing across the year (0 = flat, 1 = extreme) | e.g. (peak month − lean month) / (2 × average) from monthly GSTR-1 | 0 · 0.34 · 1 | ↑ for cash planning |
| `Revenue_Growth_Rate_Pct` | Year-on-year sales growth | Last 12 months vs previous 12 months | −55 · 9 · 76 | ↓ |

## 3. UPI (merchant transactions)

| Column | Meaning | Real-world source | Range | Risk |
|---|---|---|---|---|
| `Monthly_UPI_Inflow_INR` | UPI money received per month | PSP / bank merchant data, or UPI lines in AA statements | ₹1.3 K · ₹1.3 L · ₹1.35 Cr | Size |
| `Monthly_UPI_Outflow_INR` | UPI money paid per month | Same | — | ~ |
| `UPI_Avg_Ticket_Size_INR` | Average receipt size | Inflow ÷ number of receipts | ₹100 · ₹491 · ₹61 K | Small tickets = retail, many customers |
| `UPI_Daily_Txn_Count` | Average receipts per day | Count ÷ days | 0.1 · 9.4 · 82 | ↓ more customers = steadier |

> **Found in EDA:** Inflow ≈ ticket × daily count × 26 working days for every row. The three columns carry the same information.

## 4. Bank statements (Account Aggregator)

| Column | Meaning | How it's computed in practice | Range | Risk |
|---|---|---|---|---|
| `Monthly_Bank_Credits_INR` | Average money coming into the account per month | Sum of credits ÷ months (excluding own-account transfers) | ₹19 K · ₹4.1 L · ₹2.7 Cr | Size |
| `Monthly_Bank_Debits_INR` | Average money going out per month | Sum of debits ÷ months | — | Size |
| `Average_Bank_Balance_INR` | Average daily (or end-of-day) balance | Mean of daily balances | ₹5 K · ₹1.95 L · ₹1.7 Cr | ↓ buffer |
| `Cashflow_Stability_Index` | How steady monthly net cash flow is (0–1, higher = steadier) | e.g. 1 − coefficient of variation of monthly net flow | 0.27 · 0.68 · 0.99 | ↓ **(2nd strongest)** |
| `Transaction_Volatility_Index` | How much transaction values jump around (0–1) | e.g. std ÷ mean of daily credits | 0.03 · 0.27 · 0.72 | ↑ |
| `Overdraft_Usage_Ratio` | Average share of the overdraft/CC limit in use | Avg utilisation ÷ sanctioned limit | 0 · 0.24 · 0.6 | ↑ **(3rd strongest)** |

## 5. Credit (bureau + bank)

| Column | Meaning | Source | Range | Risk |
|---|---|---|---|---|
| `Has_Existing_Loan` | Currently has a loan | Bureau / AA | 35% yes | ~ |
| `Monthly_Loan_EMI_INR` | Total existing EMIs per month | Bureau / EMI debits in AA | 0 when no loan | ↑ via EMI burden |
| `EMI_On_Time_Rate_Pct` | % of EMIs paid on time. **Blank when there is no loan** (by design, not an error) | Bureau DPD history | 69 · 89 · 100 | ↓ |
| `Credit_History_Months` | Length of credit record. Always 0 for NTC | Bureau | 0 · 0 · 180 | ↓ |

## 6. Payroll (EPFO)

| Column | Meaning | Source | Range | Risk |
|---|---|---|---|---|
| `Monthly_Payroll_INR` | Monthly wage bill | EPFO ECR filings | ₹6.7 K · ₹57 K · ₹26 L | Size (~₹22 K per employee) |
| `Salary_Consistency_Pct` | % of months salaries/PF were paid on schedule | ECR filing dates | 53 · 85 · 100 | ↓ |
| `Employee_Attrition_Rate_Pct` | % of staff leaving per year | EPFO joins/exits | 1 · 16 · 45 | ↑ |

## 7. Invoices and receivables

| Column | Meaning | Source | Range | Risk |
|---|---|---|---|---|
| `Avg_Invoice_Payment_Delay_Days` | How late customers pay, on average | E-invoices (IRP) matched to bank receipts; TReDS | 0 · 12 · 97 days | ↑ |
| `Customer_Concentration_Ratio` | Share of sales from the top customers | E-invoice / GSTR-1 buyer GSTINs | 0.05 · 0.40 · 0.95 | ↑ dependence risk |
| `Vendor_Payment_Timeliness_Pct` | % of supplier bills paid on time | Purchase invoices matched to bank payments | 50 · 82 · 100 | ↓ |

## 8. Answer columns (supplied by the company; never used as model inputs)

| Column | What we found |
|---|---|
| 6 pillar scores | Each is an almost exact formula of the raw columns (a model recovers them with R² ≈ 0.999) |
| `Financial_Health_Score` | ≈ weighted sum of the pillars (Compliance 28%, Cash Flow 24%, Payment 23%, Revenue Consistency 11%, Stability 9%, Growth 8%) plus noise (R² 0.83) |
| `Probability_of_Default` | ≈ logistic function of the health score plus noise |
| `Credit_Risk_Category` | Exact cut on PD: < 6% Low, 6–15% Medium, > 15% High |
| `Credit_Eligible` | "No" exactly when risk is High (3 exceptions) |
| `Recommended_Credit_Limit_INR` | ≈ 16% of turnover (Low), ≈ 8% (Medium), 0 (High), rounded to ₹1,000 |

---

## 9. Derived features already built (`src/msme_health/features.py`)

| Feature | Formula | Why it matters |
|---|---|---|
| `GST_Bank_Gap` | (bank credits − GST sales) ÷ GST sales | Do the GST and bank sources agree? Big gaps mean unreported sales or circular money |
| `GST_Turnover_Gap` | GST sales × 12 ÷ declared turnover − 1 | Is the declared turnover consistent with GST? |
| `UPI_Share_of_Credits` | UPI inflow ÷ bank credits | How digital the business is; > 1 is impossible |
| `Net_Surplus_Ratio` | (credits − debits) ÷ credits | Monthly cash margin: the money available to repay |
| `Balance_Cover_Months` | average balance ÷ monthly debits | How many months the business survives with no income |
| `EMI_Burden` | existing EMI ÷ credits | FOIR-style debt load |
| `Gross_Margin_GST` | (GST sales − purchases) ÷ sales | Profitability proxy |
| `Payroll_to_Sales` | payroll ÷ GST sales | Cost structure; very high = ghost payroll or losses |
| `Revenue_per_Employee` | turnover ÷ employees | Productivity |
| `Lean_Month_Surplus_Ratio` | Lean month: sales-linked flows × (1 − seasonality), fixed costs (payroll + EMI) unchanged | **Seasonal stress test:** can the business pay its bills in its worst month? |
| `Thin_File` | credit history = 0 | Flags NTC so the model can treat it differently |

---

## 10. New feature ideas (for our discussion)

**Your input needed.** Mark each one keep, drop or modify.

| # | Idea | Formula | Hypothesis |
|---|---|---|---|
| 1 | Digital footprint score | UPI share × log(daily txns) | Businesses with many small digital receipts are steadier |
| 2 | Debt service coverage (DSCR) | (credits − debits + EMI) ÷ EMI | Classic bank metric; only for those with loans |
| 3 | Working-capital cycle | invoice delay − (days to pay vendors, from vendor timeliness) | Long cycles need more working capital |
| 4 | Compliance composite | mean of GST timeliness, salary consistency, vendor timeliness | One "discipline" signal; less noisy |
| 5 | Payroll per employee vs industry median | (payroll ÷ employees) ÷ industry median | Very low = possible under-reporting; very high = ghost staff |
| 6 | Growth quality | growth × cash-flow stability | Growth funded by steady cash vs growth that burns cash |
| 7 | Peer-relative features | each key metric as a percentile within industry × location | Removes industry effects, e.g. seasonality is normal for some sectors |
| 8 | Overdraft × seasonality interaction | overdraft × seasonality | Seasonal businesses leaning on OD are more fragile |
| 9 | Scale (log turnover) | log(turnover) | Size effects without money columns dominating |
| 10 | Vintage buckets | <1 yr, 1–3, 3–5, 5+ | Policy-friendly; banks often require 2–3 years of vintage |
| 11 | Credit-limit-to-surplus | proposed limit ÷ (12 × monthly surplus) | Leverage after the new loan |
| 12 | NTC interactions | NTC × each behaviour feature | Lets the model weigh alternate data more when the bureau is empty |

**Note:** the supplied answers look like near-linear formulas of the raw columns. New features will help most for **real** default data, where interactions and ratios usually add lift. On this synthetic data, gains will be small. That's worth saying in the interview.
