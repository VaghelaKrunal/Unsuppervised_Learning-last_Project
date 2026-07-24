# 🛍️ Mall Shopper Segmentation — Unsupervised Learning

**Production-grade customer segmentation pipeline for retail analytics**, built for a large Indian
shopping mall chain (Phoenix Marketcity / Nexus Malls style operating context) to discover natural
shopper personas and translate them into store-layout, promotion, and loyalty strategies.

---

## 📌 Overview

| | |
|---|---|
| **Business Problem** | Understand mall visitor spending behaviour to redesign store layouts, plan promotions, and personalise loyalty rewards. |
| **Dataset** | [Mall Customer Segmentation (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) — 200 customers, 5 raw columns. |
| **Approach** | Comparative unsupervised clustering: **K-Means**, **Agglomerative Hierarchical Clustering**, and **DBSCAN**. |
| **Selected Model** | K-Means (k=5), on standardised `AnnualIncome` / `SpendingScore`. |
| **Output** | 5 actionable retail personas + a reloadable production inference pipeline. |

### Key Results (from full execution of the notebook)
- **Optimal k = 5** — confirmed both visually (income vs spending scatter plot) and statistically
  (Elbow Method + Silhouette Score peak).
- **K-Means vs Agglomerative agreement: 98.0%** (Hungarian-algorithm-aligned label matching).
- **DBSCAN** (best grid params: `eps=0.4, min_samples=10`) found 4 dense clusters with ~25.5% of
  shoppers flagged as "noise" — boundary shoppers between segments, not true outliers.
- Final metrics comparison, stability check, and business recommendation are documented in-notebook
  (Steps 6.1–6.3).

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data handling | pandas, numpy |
| Clustering | scikit-learn (`KMeans`, `AgglomerativeClustering`, `DBSCAN`, `PCA`, `TSNE`) |
| Hierarchical analysis | scipy (`linkage`, `dendrogram`, `linear_sum_assignment`) |
| Visualisation | matplotlib, seaborn (custom corporate dark theme) |
| Model persistence | joblib |
| Notebook environment | Jupyter / nbformat |

---

## 🏗️ Architecture

```
mall-shopper-segmentation-unsupervised-learning/
│
├── data/
│   └── Mall_Customers.csv          # Raw Kaggle dataset
│
├── notebooks/
│   └── MallShopperSegmentation_UnsupervisedLearning.ipynb   # Fully executed, end-to-end analysis
│
├── src/
│   ├── config.py                   # Centralised paths, seeds, hyperparameters, plot theme
│   └── utils.py                    # Reusable, exception-hardened helper functions
│
├── models/
│   ├── mall_scaler.pkl             # Fitted StandardScaler (2D: Income + Spending)
│   ├── mall_segmentation_model.pkl # Best model: KMeans(k=5)
│   └── persona_map.pkl             # {cluster_id: persona_name} mapping
│
├── outputs/
│   └── *.png                       # All 18 generated plots (histograms, elbow, dendrogram, etc.)
│
├── requirements.txt
├── README.md
└── summary_report.md
```

### Design Principles

- **Separation of concerns**: all hyperparameters, file paths, and magic numbers live in `config.py`;
  all reusable logic (data validation, metric computation, persona derivation, inference) lives in
  `utils.py`. The notebook itself is an *orchestration layer* that calls into these modules — this
  mirrors how a real production ML pipeline is structured (config / lib / pipeline-entrypoint).
- **Reproducibility**: a single `RANDOM_STATE = 42` constant is propagated into every stochastic
  algorithm (K-Means, PCA, t-SNE). Re-running the notebook produces bit-identical cluster assignments.
- **Fail loudly, fail early**: data loading, scaling, and model persistence are all wrapped in
  explicit validation and exception handling (see `utils.load_and_validate_data`,
  `utils.safe_cluster_metrics`, `utils.classify_shopper`) rather than allowing silent corruption to
  propagate downstream.
- **Label-permutation-safe evaluation**: cross-algorithm agreement (K-Means vs Agglomerative) is
  computed via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), since raw cluster
  label integers are arbitrary and not directly comparable across algorithms.
- **Dynamic persona derivation**: retail persona names are derived at runtime from cluster centroid
  rank (`utils.build_persona_map`), not hardcoded to a specific cluster ID — this makes the mapping
  robust to model retraining, where sklearn's arbitrary label ordering can change between runs.

---

## ▶️ How to Run

### 1. Clone and set up environment
```bash
git clone https://github.com/<your-org>/mall-shopper-segmentation-unsupervised-learning.git
cd mall-shopper-segmentation-unsupervised-learning
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Place the dataset
Ensure `Mall_Customers.csv` is present at `data/Mall_Customers.csv` (already included in this repo).

### 3. Run the notebook
```bash
jupyter notebook notebooks/MallShopperSegmentation_UnsupervisedLearning.ipynb
```
Or execute headlessly end-to-end:
```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/MallShopperSegmentation_UnsupervisedLearning.ipynb
```

### 4. Use the trained pipeline for inference
```python
import sys
sys.path.insert(0, "src")
import config, utils

scaler = utils.load_artifact(config.SCALER_2D_PATH)
model = utils.load_artifact(config.BEST_MODEL_PATH)
persona_map = utils.load_artifact(config.PERSONA_MAP_PATH)

result = utils.classify_shopper(
    age=25, annual_income=40, spending_score=80, gender="Female",
    scaler=scaler, model=model, persona_map=persona_map,
)
print(result)
# {'input': {...}, 'cluster_label': 1, 'persona': 'Young Aspirers (Low Income, High Spend)'}
```

---

## 🧑‍🤝‍🧑 Shopper Personas

| Persona | Profile | Retail Strategy |
|---|---|---|
| **Big Spenders (Premium Target)** | High income, high spending | Premium lounge, luxury pop-ups near food court |
| **Careful Spenders (High Income, Low Spend)** | High income, low spending | Exclusive member-preview events, non-discount perks |
| **Young Aspirers (Low Income, High Spend)** | Low income, high spending | BNPL/EMI partnerships, app flash sales |
| **Budget Shoppers (Value Seekers)** | Low income, low spending | Value stores near entrances, app coupons |
| **Average/Balanced Shoppers** | Moderate income & spending | Broad seasonal/family promotions |

---

## 📹 Video Walkthrough

_Paste your recorded video URL here (Google Drive "Anyone with link" or YouTube Unlisted)._

`[VIDEO_URL_PLACEHOLDER]`

---

## 📄 License & Attribution

Dataset: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (Kaggle, public benchmark dataset).
This project was built as an end-to-end unsupervised learning exercise / production reference implementation.
