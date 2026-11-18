from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib
import os

os.makedirs('../models', exist_ok=True)

# Features (sparse matrices)
X_train_bow = load_npz('outputs/X_train_bow.npz')
X_test_bow = load_npz('outputs/X_test_bow.npz')

X_train_tfidf = load_npz('outputs/X_train_tfidf.npz')
X_test_tfidf = load_npz('outputs/X_test_tfidf.npz')

# Labels (Sentiment column from the CSVs)
y_train = pd.read_csv('outputs/train.csv')['Sentiment']
y_test = pd.read_csv('outputs/test.csv')['Sentiment']

nb_bow_model = MultinomialNB()
nb_bow_model.fit(X_train_bow, y_train)

nb_tfidf_model = MultinomialNB()
nb_tfidf_model.fit(X_train_tfidf, y_train)

y_pred_bow = nb_bow_model.predict(X_test_bow)
y_pred_tfidf = nb_tfidf_model.predict(X_test_tfidf)

joblib.dump(nb_bow_model, '../models/nb_bow_model.joblib')
joblib.dump(nb_tfidf_model, '../models/nb_tfidf_model.joblib')

print('Saved models to models/nb_bow_model.joblib and models/nb_tfidf_model.joblib')

# Function to print out metrics
def print_metrics(name, y_pred):
    print(f'\n===== {name} =====')
    print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], digits=4))
    print('Confusion matrix (rows = true, cols = predicted; order = Negative, Positive):')
    print(confusion_matrix(y_test, y_pred))

print_metrics('Naive Bayes + BoW', y_pred_bow)
print_metrics('Naive Bayes + TF-IDF', y_pred_tfidf)

# M2 Question 3: Extract misclassified reviews

# Get reviews that Naive Bayes predicted incorrectly
test_df = pd.read_csv('outputs/test.csv')
incorrect = y_test.to_numpy() != y_pred_bow

errors = test_df.loc[incorrect, ['Text', 'Sentiment']].copy()
errors['Predicted'] = y_pred_bow[incorrect]

# Print the number of errors and 10 examples for M2
print('\nTotal misclassified:', len(errors))
print(errors.head(10).to_string(index=False))

# Save all incorrect predictions for further analysis
errors.to_csv('outputs/naive_bayes_errors.csv', index=False)
