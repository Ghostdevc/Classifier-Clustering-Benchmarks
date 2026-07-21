# Classification & Clustering: A Comparative Study

> A hands-on comparison of supervised and unsupervised machine learning methods across tabular and text data, with an emphasis on model evaluation beyond accuracy — including calibration, clustering validity, and probabilistic interpretation.

---

## Overview

This repository explores how different families of machine learning models behave on the same data. Rather than searching for a single "best" model, the goal is to understand **why** each method works the way it does, and what each metric actually tells us.

Two datasets and two task types are covered:

| Dataset | Task |
|---|---|
| Wisconsin Breast Cancer | Binary classification + Clustering |
| AG News | 4-class text classification + Text clustering |

---

## Repository Structure

```
├── classifier_clustering_compare.ipynb   # Main experiments: classification & clustering
├── report.ipynb                          # Theoretical questions and written answers
├── toy_distributions_scripts.ipynb       # Distribution visualizations and proofs
└── README.md
```

---

## Datasets

### Wisconsin Breast Cancer Dataset
- **Source:** UCI Machine Learning Repository
- **Task:** Binary classification — Malignant (1) vs. Benign (0)
- **Size:** 569 samples, 30 features
- **Feature type:** Integer-valued (1–10 scale), making Categorical NB a natural fit
- **Why this dataset:** Clean, well-studied benchmark; small enough for interpretable comparisons

### AG News Dataset
- **Source:** [AG's corpus of news articles](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
- **Task:** 4-class text classification — World, Sports, Business, Sci/Tech
- **Size:** ~120,000 articles
- **Feature representation:** TF-IDF vectors (max_features=1000)
- **Why this dataset:** Tests how models handle high-dimensional, sparse representations

---

## Methods

### Classification (Breast Cancer)

| Model | Key Parameters | Paradigm |
|---|---|---|
| Bayesian Network | Structure learning: Hill Climbing + AIC | Generative |
| Categorical Naive Bayes | alpha=1 (Laplace smoothing) | Generative |
| Decision Tree (CART) | criterion='gini' | Discriminative |
| SVM — Linear | probability=True, random_state=42 | Discriminative |
| SVM — RBF | probability=True, random_state=42 | Discriminative |

### Clustering (Breast Cancer)

| Model | Key Parameters |
|---|---|
| K-Means | n_clusters=2, n_init=10 |
| GMM | n_components=2 |
| Spectral Clustering | affinity='nearest_neighbors' |
| Agglomerative | default linkage |

### Classification (AG News)

| Model | Key Parameters |
|---|---|
| Decision Tree | max_depth=10, criterion='gini' |
| Multinomial Naive Bayes | default |
| Linear SVC | dual=False |

### Clustering (AG News)

| Model | Key Parameters |
|---|---|
| K-Means | n_clusters=4 |
| Agglomerative | n_clusters=4, linkage='ward' |

---

## Results

### Breast Cancer — Classification

| Model | Accuracy |
|---|---|
| SVM Linear | ~0.960 |
| Bayesian Network | ~0.956 |
| Categorical NB | ~0.940 |

In addition to accuracy, models are evaluated with **Confusion Matrix**, **ROC-AUC**, **Calibration Curves**, and **ECE (Expected Calibration Error)** — since in a medical context, a model's confidence values matter as much as its predictions.

### Breast Cancer — Clustering

| Model | Silhouette | ARI |
|---|---|---|
| K-Means | 0.336 | 0.736 |
| GMM | 0.335 | 0.706 |
| Spectral Clustering | 0.334 | **0.830** |
| Agglomerative | 0.331 | 0.798 |

Silhouette scores are nearly identical across all methods, but ARI reveals meaningful differences. Spectral Clustering achieves the highest ARI without having the highest Silhouette — this discrepancy is intentional and discussed in the analysis.

### AG News — Classification

| Model | Accuracy |
|---|---|
| Linear SVC | Best |
| Multinomial NB | 0.8113 |
| Decision Tree | 0.4725 |

The ~34-point gap between Decision Tree and Multinomial NB on TF-IDF vectors is one of the more instructive findings in this project.

### AG News — Clustering

| Model | Silhouette | ARI |
|---|---|---|
| K-Means | ≈ 0.013 | ≈ 0.063 |
| Agglomerative | ≈ 0.013 | ≈ 0.039 |

Text clustering scores are low despite strong classification performance on the same data. This gap between supervised and unsupervised performance on text is analyzed in detail.

---

## Key Findings

**On the Silhouette vs. ARI discrepancy:**
Spectral Clustering scores the lowest Silhouette but the highest ARI among clustering methods. Silhouette measures compactness and separation in the original feature space. ARI measures agreement with ground-truth labels. When the true class boundaries are not globular — which Spectral Clustering does not assume — these two metrics can diverge substantially.

**On why text clustering fails where text classification succeeds:**
Multinomial NB reaches 81% accuracy on AG News with the same TF-IDF representation that yields ARI ≈ 0.04 in clustering. The difference is that classification uses labels to learn discriminative boundaries, while clustering must discover structure geometrically. High-dimensional sparse TF-IDF vectors make Euclidean distance — used by K-Means — unreliable. Cosine similarity or dimensionality reduction (LSA/SVD) would be more appropriate.

**On probability=True in SVM:**
SVMs do not produce calibrated probabilities natively. Enabling `probability=True` applies Platt Scaling (a sigmoid fit over the decision function scores via 5-fold CV). This is necessary for computing ROC-AUC and calibration curves, but comes at a training time cost.

**On Categorical vs. Gaussian NB:**
Wisconsin features are integers on a 1–10 scale — genuinely discrete. Using Gaussian NB would impose a continuous distribution assumption on discrete data, introducing a model mismatch. Categorical NB is the correct choice, not just a preference.

**On AIC over BIC for Bayesian Network structure learning:**
BIC penalizes model complexity proportionally to log(n), which tends to produce overly sparse graphs on medium-sized datasets. AIC applies a lighter penalty, allowing the Hill Climbing algorithm to find richer dependency structures without overfitting.

---

## Evaluation Metrics

### Classification
- **Accuracy, Precision, Recall** — standard per-class breakdown
- **ROC-AUC** — threshold-independent discriminative power
- **Calibration Curve (Reliability Diagram)** — visual check: does confidence match observed frequency?
- **ECE (Expected Calibration Error)** — scalar summary of calibration quality

### Clustering
- **Silhouette Score** — internal quality (no labels needed); measures cohesion and separation
- **ARI (Adjusted Rand Index)** — external quality; agreement with ground-truth labels, corrected for chance

---

## Theoretical Background

`report.ipynb` covers the following questions in written form:

1. Impurity axioms — Gini, Entropy, and Misclassification Error compared
2. Bayes-consistency and overfitting — why unbounded trees are inconsistent
3. Tree pruning vs. rule pruning
4. Naive Bayes — the conditional independence assumption
5. MAP derivation for Naive Bayes
6. K-Means monotonicity — why SSE decreases at every step
7. DBSCAN — core, border, and noise points

---

## Requirements

```
Python 3.9+
scikit-learn
bnlearn
numpy
pandas
matplotlib
scipy
```

Install dependencies:

```bash
pip install scikit-learn bnlearn numpy pandas matplotlib scipy
```

---

## Usage

Clone the repository and run notebooks in order:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
jupyter notebook
```

Open `classifier_clustering_compare.ipynb` to reproduce all experiments.  
Open `report.ipynb` for theoretical discussions.  
Open `toy_distributions_scripts.ipynb` for distribution visualizations.

---

## Concepts Covered

This project touches the following topics, each examined through both implementation and interpretation:

- Decision Trees: Gini impurity, Entropy, axis-aligned splits, Bayes-consistency
- Naive Bayes: Conditional independence, MAP estimation, Laplace smoothing, Categorical vs. Gaussian vs. Multinomial variants
- Bayesian Networks: Structure learning, Hill Climbing, AIC vs. BIC scoring
- SVM: Maximum margin, Linear vs. RBF kernel, Platt Scaling
- K-Means: Lloyd's algorithm, SSE monotonicity, local minima, K selection
- GMM: Mixture models, soft assignment, EM algorithm
- Spectral Clustering: Graph Laplacian, non-convex cluster shapes
- Agglomerative Clustering: Linkage strategies
- DBSCAN: Core, border, noise points; density-based boundaries
- Evaluation: Confusion matrix, ROC-AUC, calibration curves, ECE, Silhouette, ARI
- Text representations: TF-IDF, sparse vectors, curse of dimensionality

---

## License

MIT
