# Summary Report — Mall Shopper Segmentation

## Business Problem & Dataset

A large Indian shopping mall chain (Phoenix Marketcity / Nexus Malls style operator) wanted to
understand the spending behaviour of its visitors in order to redesign store layouts, plan targeted
promotional events, and personalise loyalty rewards. We used the **Mall Customer Segmentation**
dataset from Kaggle: 200 customers with `CustomerID`, `Gender`, `Age`, `AnnualIncome` (k\$), and
`SpendingScore` (1–100). The dataset is small and clean by design — the analytical challenge lies
entirely in algorithm selection, hyperparameter tuning, and business interpretation, not data cleaning.

## Features Used: 2D vs Multi-Dimensional Clustering

We ran two parallel clustering experiments. **Experiment A (2D)** clustered on
`[AnnualIncome, SpendingScore]` only — the classic, highly visual two-feature setup that directly
answers the core business question ("who spends how much relative to what they earn"). **Experiment
B (multi-dim)** added `Age` and `Gender_enc` (label-encoded gender) to test whether demographic
context sharpens or dilutes segmentation. All features were standardised via `StandardScaler` prior
to clustering, since every algorithm used (K-Means, Ward-linkage Agglomerative, DBSCAN) relies on
Euclidean distance, and `AnnualIncome`'s much larger numeric range (15–137) would otherwise dominate
`SpendingScore` (1–100) purely due to units.

## Algorithm Performance & Business Alignment

Three algorithms were trained and compared on the 2D feature set using Silhouette Score,
Davies-Bouldin Index, and Calinski-Harabasz Index:

| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | % Noise |
|---|---|---|---|---|---|
| K-Means (k=5) | 5 | 0.5547 | 0.5722 | 248.65 | 0.0% |
| Agglomerative (Ward) | 5 | 0.5538 | 0.5779 | 244.41 | 0.0% |
| DBSCAN (eps=0.4, min_samples=10) | 4 | 0.5972 | 0.4734 | 263.61 | 25.5% |

K-Means and Ward-linkage Agglomerative produced **near-identical, strong scores** across all three
metrics and achieved **98.0% cluster agreement** (computed via Hungarian-algorithm label alignment,
since raw cluster IDs are arbitrary across algorithms). DBSCAN posted a nominally higher silhouette
score, but this is computed only over the ~74.5% of points it didn't discard as noise — a meaningful
trade-off, since a quarter of real customers would be left unclassified in a production system.
**K-Means was selected as the production model**: its metrics tie with Agglomerative, it is the most
stable across random seeds (mean ± std silhouette of 0.5547 ± negligible variation across 5 seeds),
and — critically — it natively supports `.predict()` on new, unseen shoppers, which Agglomerative
Clustering does not support out-of-the-box. This directly aligns with business intuition: the
Elbow Method, Silhouette scan, and a purely visual inspection of the income-vs-spending scatter plot
(Step 1.3 of the notebook) all independently converged on **k=5**, giving high confidence that five
genuine, business-relevant shopper archetypes exist in this population.

## Shopper Segment Descriptions

1. **Big Spenders (Premium Target)** — High income and high spending score. The mall's premium
   customer base; ideal for luxury brand placement and VIP loyalty tiers.
2. **Careful Spenders (High Income, Low Spend)** — Affluent but spend conservatively; a segment
   better served by exclusive experiences than blunt discounting.
3. **Young Aspirers (Low Income, High Spend)** — Lower income but high spending score; classic
   aspirational shoppers well-suited to EMI/BNPL-driven fast-fashion promotions.
4. **Budget Shoppers (Value Seekers)** — Low income and low spending; the mall's value-conscious,
   deal-seeking segment, best reached with entrance-adjacent value retail and coupon incentives.
5. **Average/Balanced Shoppers** — The largest, moderate-income/moderate-spend segment representing
   the mainstream visitor; responsive to broad, general-interest seasonal campaigns.

DBSCAN's noise points (25.5% of shoppers at the tuned hyperparameters) were found to sit at
**moderate income and moderate spending** — the dense, ambiguous zone between segments — rather than
being genuine statistical outliers (consistent with the near-total absence of boxplot outliers found
in EDA). These are best treated as a general, unsegmented audience rather than forced into a
specific persona.

## Next Steps

To move from a one-time segmentation to a continuously useful production system, we recommend:
(1) enriching the feature set with **purchase-category-level transaction data** to segment not just
by *how much* customers spend but *on what*; (2) incorporating **footfall-sensor or Wi-Fi-analytics
visit-frequency data**, currently entirely absent from this snapshot dataset; (3) building a
**real-time persona-tagging API** around the persisted `mall_scaler.pkl` / `mall_segmentation_model.pkl`
pipeline (already implemented and validated in Step 7 of the notebook via the `classify_shopper()`
function) so that loyalty-app behavioural signals can update a shopper's segment continuously rather
than relying on a static, periodically-retrained snapshot model.
