# Data reproduction

Reproduce using the same seed, preprocessing, split, and vocabulary as the modelers.

1. Download `Reviews.csv` from the Amazon Fine Food Reviews Kaggle page and put it in `data/` at the repo root (`data/Reviews.csv`).
2. Install `pip install pandas scikit-learn scipy matplotlib joblib`.
3. Run the notebooks in this folder in order: `load_and_label.ipynb`, then `bow_feature.ipynb` and `tf-idf_feature.ipynb`.
   All generated files go to `notebooks/outputs/`.
   The original `make_sample.py` / `preprocess.py` / `feature_engineering.py` scripts produce identical outputs, but they expect `Reviews.csv` next to them.

## Decisions

- Sample original 568,454-row dataset first: `sample(n=50000, random_state=42)` (Group Alpha).
- Drop missing `Text`/`Score` and exact duplicates on (`UserId`, `ProductId`, `Text`); do not deduplicate by user ID alone.
- Drop 3-star (neutral/mixed) reviews.
- `Sentiment=1` for 4-5 stars; `Sentiment=0` for 1-2 stars.
- HTML-decode and strip HTML tags from review text before vectorizing.
- Stratified 80/20 train/test split with `random_state=42`.
- Train-only `fit_transform`, test-only `transform` prevents train/test feature leakage.
- Both vectorizers use English stop words and `max_features=10000`.

## Outputs

- `outputs/train.csv`, `outputs/test.csv`: same row ordering as features; labels in `Sentiment` column.
- `outputs/X_train_bow.npz`, `X_test_bow.npz`: sparse count features.
- `outputs/X_train_tfidf.npz`, `X_test_tfidf.npz`: sparse TF-IDF features.
- `outputs/bow_vectorizer.joblib`, `tfidf_vectorizer.joblib`: fitted transformations for later inference.
- `outputs/word_frequency.png`: exploratory chart of frequent terms, not top predictive features.

Example ML engineer loading:

```python
import pandas as pd
from scipy.sparse import load_npz
train = pd.read_csv('outputs/train.csv')
test = pd.read_csv('outputs/test.csv')
X_train = load_npz('outputs/X_train_bow.npz')
X_test = load_npz('outputs/X_test_bow.npz')
y_train = train['Sentiment']
y_test = test['Sentiment']
```

## What to commit

Commit the notebooks, this README, and optionally `outputs/word_frequency.png` to the public GitHub repository. Do not commit `Reviews.csv`, the sampled/cleaned CSVs, the split CSVs, the `.npz` matrices, or `.joblib` artifacts; these are generated locally (`.gitignore` already excludes everything in `notebooks/outputs/` except the chart). Share generated data privately if a teammate cannot run the scripts.

## To complete Milestone 2

Classical ML Engineer: use these exact splits and produce model metrics, confusion matrix, and top 10 positive and negative informative features for EACH representation based on a documented method. Word frequency does not meet that last requirement. Analyst/Writer: integrate this data section into the milestone PDF; project lead handles timeline and progress logs.
