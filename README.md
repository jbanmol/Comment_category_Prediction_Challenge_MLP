# Comment Moderation NLP Ensemble

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-NLP-F7931E?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-02569B)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6600)
![CatBoost](https://img.shields.io/badge/CatBoost-Ensemble-FFCC00)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-Embeddings-F59E0B)

NLP classification pipeline for an imbalanced comment-moderation challenge. The solution combines structured feature engineering, TF-IDF/SVD representations, sentence-transformer embeddings, gradient boosting models, and per-class threshold optimization to improve macro F1.

## What This Demonstrates

- Feature engineering from text, metadata, engagement, identity signals, and platform flags
- Dual text representation: sparse TF-IDF/SVD plus dense sentence embeddings
- Multi-model ensemble using LightGBM, XGBoost, CatBoost, and logistic regression
- Class-imbalance handling for rare labels
- Per-class threshold tuning for macro-F1 optimization
- Kaggle-style experimental workflow with reproducible notebooks/scripts

## Problem

The task is to predict one of four moderation categories for user-generated comments. The main difficulty is class imbalance: one minority class is rare and semantically close to a more common hostile/inflammatory class.

## Approach

```text
Raw comments and metadata
  -> exploratory analysis
  -> engineered structured features
  -> TF-IDF + TruncatedSVD
  -> sentence-transformer embeddings
  -> gradient boosting and linear text models
  -> weighted ensemble
  -> per-class threshold optimization
```

## Results Snapshot

| Model / Stage | OOF Macro F1 |
|---|---:|
| Logistic regression text baseline | ~0.55 |
| Tree models with structured + text features | ~0.64-0.65 |
| Weighted ensemble | ~0.66+ |
| Ensemble + threshold optimization | ~0.67+ |

## Repository Structure

| Path | Purpose |
|---|---|
| `23f1001015-notebook-v4.ipynb` | Main solution notebook |
| `v4_solution.py` | Python script version of the solution |
| `PROBLEM_STATEMENT.md` | Competition description and EDA findings |
| `README.md` | Portfolio overview |

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, sentence-transformers, TF-IDF, TruncatedSVD.

## Key Lesson

The biggest gains came from combining domain-specific EDA with representation diversity: structured signals solved some classes, while semantic embeddings helped distinguish the ambiguous minority class.
