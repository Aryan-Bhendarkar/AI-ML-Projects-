# Patent Success Prediction Model

> Goal: predict whether a USPTO patent becomes “high_impact” using only safe, non-leaky features.

---

## 1. Project Overview

- **Problem statement**: classify patents into high vs. low impact based on metadata (title, abstract, claims, assignee info, etc.).
- **Motivation**: practise an end-to-end ML workflow—clean raw CSVs, remove data leaks, build baseline models, and track learning.
- **Key takeaways**: basics matter most—data types, duplicates, leakage checks, and simple baselines before trying advanced models.

---

## 2. Dataset Summary

| Item | Notes |
|------|-------|
| Source | `content/patent_sample_*.csv` (2 files merged) |
| Target | `high_impact` (0/1) |
| Raw shape | 200000 rows × 16 columns |
| After cleaning | 99495 rows × 13 columns (duplicates + constant columns removed) |
| Class balance | 0: 80.7%, 1: 19.2% |

### Cleaning steps
1. Standardised `grant_date` → datetime, other numeric columns → integers.
2. Imputed missing values with medians (numerics) or sensible text defaults.
3. Removed ~50% duplicate rows.
4. Dropped nearly-constant columns (`patent_country`, `cpc_section`, `cpc_subsection`).
5. Identified and excluded leakage columns (`citation_decile`, `forward_citations`, `backward_citations`).

---

## 3. Feature Engineering & Preprocessing

- **Numeric**: `num_claims`, `assignee_type`, `grant_year` → scaled.
- **Categorical**: remaining object columns (e.g., `assignee_name`, `cpc_subsection`) → one-hot.
- **Text**: `title`, `abstract` → TF‑IDF (1–2 grams, max 2000/5000 features).
- **Tools**: `ColumnTransformer`, `Pipeline` keep train/test separation clean.

---

## 4. Modeling Timeline

| Stage | Model & Settings | Why we tried it | F1 (weighted) |
|-------|-----------------|-----------------|---------------|
| Baseline 1 | `LogisticRegression` on leaky numerics | Quick sanity check → revealed leakage | 1.0 (misleading) |
| Diagnosis | Correlation + groupby checks | Found `citation_decile` leak | — |
| Baseline 2 | `LogisticRegression` (`class_weight="balanced"`) on safe numerics | Honest baseline | 0.57 |
| Full pipeline | `LogisticRegression` on numeric + categorical + TF-IDF text | Demonstrate preprocessing pipeline | 0.84 |
| Stretch goal | `CatBoostClassifier` (`auto_class_weights="Balanced"`) | Handles mixed data easily, higher F1 | **0.85(target)** |

> Note: CatBoost selected as the “simple but strong” learner once basics were solid.

---
## 5. Observations

- Leakage detection was critical: `citation_decile` mapped uniquely to the target.  
- Class imbalance handled with stratified sampling and model-based class weights.  
- Text features delivered the biggest jump beyond numeric baselines.  
- CatBoost offered robustness with minimal feature engineering once leakage was resolved.  
- Documentation of each iteration simplified traceability and compliance with project guidelines.

---

## 6. Learning Reflections

1. Always **profile raw data**—duplicates and dtypes were bigger issues than modeling.
2. **Leakage hunting** is essential; perfect correlations are suspicious.
3. Building a **modular pipeline** kept the workflow repeatable.
4. Advanced models (CatBoost) only shine **after** fundamentals are in place.
5. Documentation (like this README) helps articulate decisions for future us.

---

## 7. Reproducibility Steps

1. Create environment: `pip install -r requirements.txt` (includes `catboost`, `scikit-learn`, `pandas`).
2. Place raw CSVs under `content/`.
3. Run notebook `PatentSuccessPrediction.ipynb` sequentially (cleans data + trains models).

---
---

## 9. References

- Scikit-learn docs: <https://scikit-learn.org/stable/>
- CatBoost docs: <https://catboost.ai/en/docs/>
- USPTO patent dataset downloads:
	- [Part 1 (Google Drive)](https://drive.google.com/file/d/1hB6nm4tVKwhQWPy1wk8mnpiuEfgMjU6W/view?usp=sharing)
	- [Part 2 (Google Drive)](https://drive.google.com/file/d/1OreKjHk0rmRIbZ_QbRUDrH9xuRUuUC0L/view?usp=sharing)

---

