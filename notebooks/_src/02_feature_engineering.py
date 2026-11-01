# %% [markdown]
# # 02 · Feature Engineering
#
# The EDA showed three things:
# - The raw money columns mostly measure **size**.
# - The behavioural columns carry the **risk** signal.
# - The data sources can be **cross-checked** against each other.
#
# This notebook builds analyst-style features on top of the raw data and tests whether they carry signal.
#
# Full definitions: `docs/FEATURE_DICTIONARY.md`. Code: `src/msme_health/features.py`. The same function runs in training, in the API and in the dashboard, so there is no training/serving skew.

# %%
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

from msme_health.config import DATA_PATH, NUMERIC_RAW
from msme_health.features import DERIVED, LABELS, add_derived

sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams["figure.dpi"] = 110
raw = pd.read_csv(DATA_PATH)
df = add_derived(raw)
df["logit_pd"] = np.log(df.Probability_of_Default / (1 - df.Probability_of_Default))
df[DERIVED].describe(percentiles=[0.01, 0.5, 0.99]).T.round(3)

# %% [markdown]
# ## 1. The derived features
#
# | Family | Features | Credit question it answers |
# |---|---|---|
# | Cross-source consistency | GST-bank gap, GST-turnover gap, UPI share of credits | Do the sources tell the same story? |
# | Cash-flow capacity | Net surplus ratio, balance cover months, EMI burden | Is there money left to repay a loan? |
# | Business economics | Gross margin, payroll-to-sales, revenue per employee | Is the business model healthy? |
# | Seasonality stress | Lean-month surplus ratio | Can it pay in its *worst* month? |
# | Credit file | Thin-file flag | Is the bureau empty (NTC)? |
#
# ### The lean-month stress test, step by step
#
# 1. Seasonality index *s* says how far sales drop below average in the lean month (sales × (1 − s)).
# 2. Costs that move with sales (purchases, other variable costs) fall by the same share.
# 3. **Fixed costs do not fall:** payroll and existing EMIs stay the same.
# 4. Lean-month surplus = (credits − variable costs) × (1 − s) − payroll − EMI.

# %%
ex = df.loc[0, ["Monthly_Bank_Credits_INR", "Monthly_Bank_Debits_INR", "Monthly_Payroll_INR", "Monthly_Loan_EMI_INR", "Seasonality_Index", "Lean_Month_Surplus_INR"]]
credits, debits, payroll, emi, s, lean = ex.values
variable = debits - payroll - emi
print(f"Worked example: {raw.MSME_ID[0]}")
print(f"  Average month: credits ₹{credits:,.0f} − debits ₹{debits:,.0f} = surplus ₹{credits - debits:,.0f}")
print(f"  Variable costs = debits − payroll − EMI = ₹{variable:,.0f}")
print(f"  Lean month (s = {s:.3f}): (₹{credits:,.0f} − ₹{variable:,.0f}) × {1 - s:.3f} − ₹{payroll:,.0f} − ₹{emi:,.0f} = ₹{lean:,.0f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
neg = df.assign(lean_deficit=df.Lean_Month_Surplus_INR < 0).groupby("Industry").lean_deficit.mean().sort_values()
axes[0].barh(neg.index, neg.values, color="#6366F1")
axes[0].set_title("Share of businesses with a lean-month cash deficit")
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
order = ["Low", "Medium", "High"]
sns.boxplot(data=df, x="Credit_Risk_Category", y="Lean_Month_Surplus_Ratio", order=order, ax=axes[1],
            palette=["#10B981", "#F59E0B", "#EF4444"], fliersize=0)
axes[1].set_ylim(-0.5, 0.5)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("Lean-month surplus ratio by supplied risk band")
plt.tight_layout()
plt.show()
print(f"Businesses that run a cash deficit in their lean month: {(df.Lean_Month_Surplus_INR < 0).mean():.1%}")

# %% [markdown]
# **Takeaway:** about a third of businesses cannot cover fixed costs in their lean month. That is too many to decline outright, so we use the stress test as a **loan-structuring signal** (seasonal or step-down EMI) rather than a hard cut-off.
#
# ## 2. How much signal does each feature carry?
#
# Mutual information (MI) measures *any* relationship (not only linear) between a feature and the log-odds of default. We compare raw and derived features on the same scale.

# %%
cols = NUMERIC_RAW + [c for c in DERIVED if c != "Thin_File"]
X = df[cols].fillna(-1)
smp = X.sample(15000, random_state=0).index
mi = pd.Series(mutual_info_regression(X.loc[smp], df.loc[smp, "logit_pd"], random_state=0), index=cols).sort_values()
colors = ["#F59E0B" if c in DERIVED else "#6366F1" for c in mi.index]
plt.figure(figsize=(8, 10))
plt.barh([LABELS.get(c, c) for c in mi.index], mi.values, color=colors)
plt.title("Mutual information with log-odds of default\n(orange = derived feature, purple = raw)")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Takeaway:**
# - Behavioural raw features (GST timeliness, cash-flow stability, overdraft, salary consistency) dominate.
# - Among the derived features, **net surplus ratio** and **lean-month surplus** carry real signal. They summarise cash capacity in one number.
# - Cross-source gaps carry little *risk* signal. That's expected: they exist to catch **fraud**, not to rank honest businesses.
#
# ## 3. Derived features vs default risk

# %%
show = ["Net_Surplus_Ratio", "Lean_Month_Surplus_Ratio", "Balance_Cover_Months", "EMI_Burden", "Gross_Margin_GST", "Payroll_to_Sales"]
fig, axes = plt.subplots(2, 3, figsize=(16, 7))
for ax, col in zip(axes.flat, show):
    d = df[[col, "Probability_of_Default"]].dropna()
    if col == "EMI_Burden":
        d = d[d[col] > 0]
    b = pd.qcut(d[col], 20, duplicates="drop")
    g = d.groupby(b, observed=True).agg(x=(col, "median"), y=("Probability_of_Default", "mean"))
    ax.plot(g.x, g.y, marker="o", color="#4F46E5")
    ax.set_title(LABELS.get(col, col) + (" (borrowers only)" if col == "EMI_Burden" else ""))
    ax.set_ylabel("mean PD")
plt.suptitle("Mean supplied PD across 20 bins of each derived feature", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Takeaway:** surplus ratio and lean-month surplus show clean monotonic relationships with PD, so we give them **monotonic constraints** (more surplus → never riskier). Gross margin and payroll share are nearly flat. We keep them for explanations and anomaly checks.
#
# ## 4. Redundancy check

# %%
c = df[DERIVED + ["Cashflow_Stability_Index", "Overdraft_Usage_Ratio"]].corr(method="spearman")
plt.figure(figsize=(10, 8))
sns.heatmap(c, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot_kws={"size": 7},
            xticklabels=[LABELS.get(i, i) for i in c.index], yticklabels=[LABELS.get(i, i) for i in c.index])
plt.title("Spearman correlation among derived features")
plt.show()

# %% [markdown]
# **Takeaway:** net surplus and lean-month surplus are strongly related (by construction) but answer different questions: the average month vs the worst month. Nothing else is redundant. Tree models handle correlated features well. For the linear baseline, Ridge regularisation absorbs it.
#
# ## 5. Peer-relative view
#
# The same overdraft usage can be normal in one industry and alarming in another. The dashboard ranks each business within its **industry × location** peer group (50 groups, about 1,000 businesses each).

# %%
g = df.groupby(["Industry", "Location_Category"]).size()
print(f"Peer groups: {len(g)} · size min {g.min()} · median {int(g.median())} · max {g.max()}")
piv = df.pivot_table(index="Industry", columns="Location_Category", values="Overdraft_Usage_Ratio", aggfunc="median")
plt.figure(figsize=(8, 5))
sns.heatmap(piv[["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"]], annot=True, fmt=".3f", cmap="Purples")
plt.title("Median overdraft usage by peer group")
plt.show()

# %% [markdown]
# **Takeaway:** peer medians are very similar in this synthetic data. In real data they differ a lot. The peer benchmark is built so it adapts automatically, and it falls back to industry-only peers when a group has fewer than 200 businesses.
#
# ## 6. Feature ideas still to test
#
# Listed in `docs/FEATURE_DICTIONARY.md` §10:
# - DSCR
# - Working-capital cycle
# - Compliance composite
# - Payroll per employee vs industry
# - Growth quality
# - Peer-percentile features
# - NTC interactions
#
# We'll test them in `03_modelling` with the same ablation method used there.
