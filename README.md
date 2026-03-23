# 🏷️ Comment Category Prediction Challenge

> **Kaggle Competition** — Predicting how an online platform categorizes user-generated comments into 4 distinct labels using textual, metadata, and system-generated features.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-GPU-02569B?logo=microsoft&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-FF6600?logo=xgboost&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-GPU-FFCC00?logo=catboost&logoColor=black)
![HuggingFace](https://img.shields.io/badge/🤗_Sentence_Transformers-384d-FFD21E)
![TF-IDF](https://img.shields.io/badge/TF--IDF-SVD_300d-6DB33F)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)

---

## 📋 Problem Statement

An online discussion platform processes and categorizes user-generated comments. Each record includes:
- **Comment text** — raw content
- **Engagement signals** — upvotes, downvotes
- **System-generated features** — internal platform signals (`if_1`, `if_2`)
- **Identity detections** — race, religion, gender, disability mentions
- **Emoticon indicators** — 3 separate emoticon group flags

**Goal:** Predict the final category label (4 classes) assigned to each comment.  
**Metric:** Macro F1-Score (treats all classes equally — critical for minority class performance).

### Label Distribution & Key Insights

| Label | Meaning | % of Data | `if_2` Median | Key Signal |
|:---:|---|:---:|:---:|---|
| **0** | Normal comment | 57.7% | 4 | `if_2==4`, `if_1==0` (79%) |
| **1** | Hate speech (racial/gender bias) | 8.0% | 10 | `if_1 > 0` (83%), high race mentions |
| **2** | Hostile / inflammatory | 31.5% | 10 | `if_2==10`, mixed identity |
| **3** | Borderline political | 2.8% | 10 | Hardest class — easily confused with Label 2 |

> **Core challenge:** Label 3 represents only **2.8%** of training data and is semantically close to Label 2, dragging macro F1 down significantly.

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAW INPUT DATA                             │
│           comment · metadata · engagement · system signals      │
└──────────────┬──────────────────────────────────┬───────────────┘
               │                                  │
    ┌──────────▼──────────┐            ┌──────────▼──────────┐
    │   TEXT PIPELINE      │            │  STRUCTURED PIPELINE │
    │                      │            │                      │
    │  TF-IDF (word+char)  │            │  90+ engineered      │
    │       ↓              │            │  features from EDA   │
    │  TruncatedSVD(300)   │            │  (if_1/if_2 flags,   │
    │       +              │            │   identity signals,  │
    │  Sentence-Transformer│            │   engagement stats,  │
    │  (all-MiniLM, 384d)  │            │   temporal, etc.)    │
    └──────────┬───────────┘            └──────────┬───────────┘
               │                                   │
               └─────────────┬─────────────────────┘
                             │
               ┌─────────────▼─────────────────┐
               │    COMBINED FEATURE MATRIX     │
               │  (structured + SVD + ST = 774) │
               └─────────────┬─────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
    │ LightGBM  │     │  XGBoost  │     │  CatBoost │
    │   (GPU)   │     │   (GPU)   │     │   (GPU)   │
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                  │
          └────────┬────────┴──────────────────┘
                   │            ┌───────────────┐
                   │            │ Logistic Reg  │
                   │            │ (on text only)│
                   │            └───────┬───────┘
                   │                    │
          ┌────────▼────────────────────▼───────┐
          │     WEIGHTED ENSEMBLE BLEND          │
          │  (optimized 4-model weight search)   │
          └────────────────┬────────────────────┘
                           │
          ┌────────────────▼────────────────────┐
          │   PER-CLASS THRESHOLD OPTIMIZATION   │
          │   (boost minority label 3 recall)    │
          └────────────────┬────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ SUBMISSION  │
                    └─────────────┘
```

---

## 🔬 Approach Breakdown

### 1. Exploratory Data Analysis

- Discovered that `if_2 == 4` is a near-perfect indicator for Label 0
- `if_1 > 0` appears in 83% of Label 1 (hate speech) vs only 21% in Label 0
- Race mentions (`black`, `white`) are 5× more frequent in Label 1 vs Label 0
- Labels 2 and 3 share similar structured feature distributions — text content is the key differentiator

### 2. Feature Engineering (90+ features)

| Group | Features | Purpose |
|---|---|---|
| **`if_1` / `if_2` Flags** | Exact value flags, log transforms, interactions (`product`, `ratio`, `diff`) | Primary class discriminators |
| **Identity Signals** | Per-value binary flags for race, religion, gender + NaN indicators + cross-feature combos | Hate speech detection (Label 1) |
| **Engagement** | Log votes, vote ratio, controversy score, vote diff | Behavioral signals |
| **Text Statistics** | Word/char counts, caps ratio, punctuation density, URL/hashtag/mention counts | Content style indicators |
| **Temporal** | Hour, day-of-week, month with cyclical sin/cos encoding | Posting pattern signals |
| **Post-Level Aggregations** | Per-`post_id` comment count, mean `if_2`, mean upvotes | Thread-level context |
| **Emoticons** | Sum, presence flags, diversity (distinct nonzero groups) | Expression signals |

### 3. Dual Text Representation

| Method | Dimensions | Captures |
|---|---|---|
| **TF-IDF** (word 1-2gram + char 2-4gram) → **TruncatedSVD(300)** | 300 | Lexical patterns, slang, misspellings |
| **Sentence-Transformers** (`all-MiniLM-L6-v2`) | 384 | Semantic meaning, contextual similarity |

> The transformer embeddings are critical for separating Label 2 (hostile) from Label 3 (borderline political) — two classes that share similar vocabulary but differ in semantic intent.

### 4. Multi-Model Ensemble

All models trained with **5-fold Stratified K-Fold** cross-validation:

| Model | Key Hyperparameters | Role |
|---|---|---|
| **LightGBM** | 4000 trees, lr=0.03, 255 leaves, GPU | Primary structured learner |
| **XGBoost** | 3000 trees, lr=0.03, depth=8, GPU | Diversity via different tree algorithm |
| **CatBoost** | 3000 iters, lr=0.05, depth=8, GPU | Handles categoricals natively |
| **Logistic Regression** | C=3.0, balanced weights, multinomial | Text-specialist with linear boundaries |

### 5. Post-Processing

- **Ensemble weight optimization** — grid search over 4-model blend weights on OOF predictions
- **Per-class threshold tuning** — adjusts decision boundaries to maximize macro F1, specifically boosting Label 3 recall

---

## 📁 Repository Structure

```
.
├── 23f1001015-notebook-v4.ipynb   # Main Kaggle notebook (v4 — current best)
├── 23f1001015-notebook-t12026.ipynb  # Previous v3 baseline notebook
├── v4_solution.py                  # v4 source as Python script
├── PROBLEM_STATEMENT.md            # Competition description + EDA findings
└── README.md                       # This file
```

---

## 🚀 How to Reproduce

### On Kaggle (Recommended)

1. Fork/upload `23f1001015-notebook-v4.ipynb` to [Kaggle](https://www.kaggle.com)
2. Add the competition dataset as input
3. In Settings → enable **GPU accelerator** and **Internet access**
4. Click **Run All** (~20-30 min)
5. Download and submit `submission.csv`

### Locally

```bash
git clone https://github.com/jbanmol/Comment_category_Prediction_Challenge_MLP.git
cd Comment_category_Prediction_Challenge_MLP

# Install dependencies
pip install numpy pandas scikit-learn lightgbm xgboost catboost sentence-transformers matplotlib scipy

# Place train.csv, test.csv, Sample.csv in the appropriate path
# Update file paths in the notebook, then run
jupyter notebook 23f1001015-notebook-v4.ipynb
```

---

## 📊 Results

| Model | OOF Macro F1 |
|---|---|
| Logistic Regression (text only) | ~0.55 |
| LightGBM (structured + text) | ~0.65 |
| XGBoost (structured + text) | ~0.64 |
| CatBoost (structured + text) | ~0.64 |
| **Weighted Ensemble** | **~0.66+** |
| **+ Threshold Optimization** | **~0.67+** |

> **Leaderboard baseline:** 0.65656 macro F1

---

## 💡 Key Takeaways

1. **Domain-specific EDA matters** — discovering `if_2==4 → Label 0` was more impactful than any model choice
2. **Class imbalance kills macro F1** — Label 3 at 2.8% needed explicit threshold tuning, not just `class_weight='balanced'`
3. **Semantic embeddings complement TF-IDF** — transformer representations capture meaning that n-gram statistics miss
4. **Ensemble diversity > individual model quality** — combining different tree algorithms (LightGBM/XGBoost/CatBoost) gave more robust predictions

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.12 |
| **ML Frameworks** | LightGBM, XGBoost, CatBoost, scikit-learn |
| **NLP** | TF-IDF, TruncatedSVD, Sentence-Transformers (HuggingFace) |
| **Data** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib |
| **Compute** | Kaggle GPU (NVIDIA Tesla T4) |
| **Platform** | Kaggle Notebooks, Jupyter |

---

<p align="center">
  <i>Built as part of the IIT Madras BS Degree — Machine Learning Practice course</i>
</p>
