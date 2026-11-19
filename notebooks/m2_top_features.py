import numpy as np
import joblib
from pathlib import Path

root = Path(__file__).resolve().parent.parent

models = [
    ('Bag of Words', 'nb_bow_model.joblib', 'bow_vectorizer.joblib'),
    ('TF-IDF', 'nb_tfidf_model.joblib', 'tfidf_vectorizer.joblib')
]

for name, model_file, vectorizer_file in models:
    model = joblib.load(root / 'models' / model_file)
    vectorizer = joblib.load(root / 'notebooks' / 'outputs' / vectorizer_file)

    terms = vectorizer.get_feature_names_out()

    # Compare the learned word probabilities for each sentiment
    pos_index = list(model.classes_).index(1)
    neg_index = list(model.classes_).index(0)

    scores = model.feature_log_prob_[pos_index] - model.feature_log_prob_[neg_index]


    from scipy.sparse import load_npz

    feature_file = 'X_train_bow.npz' if name == 'Bag of Words' else 'X_train_tfidf.npz'
    X_train = load_npz(root / 'notebooks' / 'outputs' / feature_file)

    doc_freq = np.asarray((X_train > 0).sum(axis=0)).ravel()
    eligible = np.where(doc_freq >= 30)[0]

    top_positive = eligible[np.argsort(scores[eligible])[-10:][::-1]]
    top_negative = eligible[np.argsort(scores[eligible])[:10]]


    print(f'\n===== {name} =====')
    print(f'{"Rank":<6} {"Positive":<22} {"Negative":<22}')

    for i in range(10):
        print(f'{i+1:<6} {terms[top_positive[i]]:<22} {terms[top_negative[i]]:<22}')
