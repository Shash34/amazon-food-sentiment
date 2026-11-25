# Amazon Fine Food Reviews — Sentiment Analysis

**CSE 3521: Survey of Artificial Intelligence I — Autumn 2026 Group Project**

## Description

This project builds and deploys a sentiment analysis system for the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset. We use machine learning to classify reviews as positive or negative and to surface what reviewers are talking about (e.g., taste, packaging, price), helping retailers and food providers understand product satisfaction and spot areas of concern.

We work from a 50,000-review sample drawn with our group seed (**seed = 42**).

### Planned approach

- **Features:** Bag of Words and TF-IDF
- **Classical models:** Naive Bayes, Logistic Regression
- **Deep model:** Feedforward Neural Network
- **App:** Streamlit web app with live predictions from multiple models

## Team Members

- Kevin Turner
- Maximilian Wang
- Kunaal Gudavalli
- Sumanth Gopal
- Adam Hagarty
- Shashank Kesavamatham

## Data

Download `Reviews.csv` from Kaggle and sample 50,000 rows with `random_state = 42`. Data files are not committed due to size.

## Reproducing the pipeline

1. Create a `data/` folder in the repository root if it doesn't already exist. Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place it in `data/Reviews.csv`.
2. `pip install -r requirements.txt`
3. Run the scripts from inside `notebooks/`:


```bash
cd notebooks
python make_sample.py
python preprocess.py
python feature_engineering.py
python train_naive_bayes.py
```

See [notebooks/README_DATA_ENGINEERING.md](notebooks/README_DATA_ENGINEERING.md) for preprocessing decisions and output files.
