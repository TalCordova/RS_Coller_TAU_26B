# Recommender Systems — TA Session Notebooks
### Tel-Aviv University · Coller School of Management · Semester B 2026

This repository contains Google Colab notebooks used in the TA sessions of the Recommender Systems course. The notebooks build on each other sequentially — each one picks up where the previous left off.

The slides for this session can be found [here](https://docs.google.com/presentation/d/1jjdi-QIrcRPyKd0FAdInjVFw3YSNbQXXgjGV6qGDi60/edit?slide=id.p1#slide=id.p1).

---

## Notebooks

### [Notebook 1 — Data Preparation](notebook1_data_handling.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TalCordova/RS_Coller_TAU_26B/blob/master/notebook1_data_handling.ipynb)

Before building any recommender, you need to understand your data and make a series of decisions that will affect everything downstream.

**Topics covered:**
- Loading and exploring the MovieLens Small dataset (~100k ratings, 610 users, 9,742 movies)
- Exploratory data analysis: rating distributions, the long tail, user activity (Lorenz/Pareto)
- The utility matrix and sparsity (~98.3%)
- Filtering low-activity users (< 5 interactions)
- Three train/test split strategies and their trade-offs:
  - Random split
  - Global temporal split
  - **User temporal split** (used in subsequent notebooks)

---

### [Notebook 2 — Ranking & Evaluation](notebook2_ranking_evaluation.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TalCordova/RS_Coller_TAU_26B/blob/master/notebook2_ranking_evaluation.ipynb)

Production systems show users a ranked list, not a predicted score. This notebook covers how to evaluate and train for ranking directly.

**Topics covered:**
- Pointwise vs. pairwise ranking framing
- Evaluation protocol: 1 positive item + 99 sampled negatives per user
- Ranking metrics:
  - **HR@K** — did the positive land in the top K?
  - **NDCG@K** — how high up in the top K did it land?
  - **MRR** — reciprocal rank across the full list
- **Pointwise model**: Matrix Factorization trained with MSE/SGD
- **Pairwise model**: Bayesian Personalized Ranking (BPR) via `implicit`
- Head-to-head comparison — each model wins on what it was trained to optimize

---

### [Notebook 3 — Collaborative Filtering](notebook3_collaborative_filtering.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TalCordova/RS_Coller_TAU_26B/blob/master/notebook3_collaborative_filtering.ipynb)

Memory-based collaborative filtering: where does the signal come from, and how do we use it?

**Topics covered:**
- **Item-item CF** — cosine similarity between item vectors; *"Because you watched X..."*
- **User-user CF** — cosine similarity between user vectors; *"Users like you also watched..."*
- kNN lookup using `sklearn.neighbors.NearestNeighbors`
- Rating prediction formulas for both approaches
- 🏋️ **Exercises**: implement `predict_rating_item_item` and `predict_rating_user_user`
- Real-world challenges:
  - **Cold start** — new users and new items have no interaction history
  - **Popularity bias** — the long tail rarely gets recommended
  - **Beyond accuracy** — diversity, serendipity, fairness, catalog coverage

---

### [Notebook 4 — Content-Based Filtering](notebook4_content_based.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TalCordova/RS_Coller_TAU_26B/blob/master/notebook4_content_based.ipynb)

Same retrieval machinery as item-item CF — *"Because you watched X..."* — but the similarity signal now comes from **item features** instead of the utility matrix.

**Topics covered:**
- Loading movie metadata (genres) from *The Movies Dataset* (TMDB, ~45k movies)
- Minimal preprocessing: parsing genre strings → multi-hot encoding
- Building a genre-based similarity index with `NearestNeighbors` (cosine)
- *"Because you watched X..."* recommendations driven purely by content, with no interaction data
- 🏋️ **Exercise**: critically examine similarity-1.0 ties and discuss richer item features (overview text, cast/crew, embeddings)
- Where content-based fits relative to CF: solves **item** cold start, but not **user** cold start

---

## Additional Reference Notebook

### [Recommender Systems in Python 101](Recommender_Systems_in_Python_101.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TalCordova/RS_Coller_TAU_26B/blob/master/Recommender_Systems_in_Python_101.ipynb)

> **Note:** This notebook is not covered in the TA sessions and was not written for this course. It is a copy of the public Kaggle kernel **["Recommender Systems in Python 101"](https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101)** by **[Gabriel Moreira](https://www.kaggle.com/gspmoreira)** (@gspmoreira), included here as a reference.

A broader end-to-end walkthrough of the main recommender system families, using the [CI&T Deskdrop dataset](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop) (~73k user interactions on ~3k internal company articles) — created and published by the same author.

**What it covers:**
- **Popularity model** — a strong non-personalized baseline
- **Content-Based Filtering** — TF-IDF article profiles + weighted user profiles, scored via cosine similarity
- **Collaborative Filtering** — SVD matrix factorization (scipy `svds`)
- **Hybrid model** — weighted ensemble of CB and CF scores
- Evaluation using **Recall@5** and **Recall@10** (1 positive + 100 sampled negatives per user)

---

## Setup

The notebooks are designed to run in **Google Colab** — no local installation needed. Open any notebook via the Colab badge above, or clone the repo and open locally.

**Dependencies** (installed automatically in each notebook):
```
numpy  pandas  scikit-learn  matplotlib  seaborn  implicit
```

---

## Structure

```
notebook1_data_handling.ipynb            # EDA, filtering, train/test split
notebook2_ranking_evaluation.ipynb       # Ranking metrics, MF vs BPR
notebook3_collaborative_filtering.ipynb  # Item-item & user-user CF
notebook4_content_based.ipynb            # Content-based filtering via genres
Recommender_Systems_in_Python_101.ipynb  # Reference: CB, CF, Hybrid (not covered in sessions)
```
