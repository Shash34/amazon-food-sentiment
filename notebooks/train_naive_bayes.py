from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Features (sparse matrices)
X_train = load_npz('outputs/X_train_bow.npz')
X_test = load_npz('outputs/X_test_bow.npz')

# Labels (Sentiment column from the CSVs)
y_train = pd.read_csv('outputs/train.csv')['Sentiment']
y_test = pd.read_csv('outputs/test.csv')['Sentiment']

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
print(confusion_matrix(y_test, y_pred))


# M2 Question 3: Extract misclassified reviews

# Get reviews that Naive Bayes predicted incorrectly
test_df = pd.read_csv('outputs/test.csv')
incorrect = y_test.to_numpy() != y_pred

errors = test_df.loc[incorrect, ['Text', 'Sentiment']].copy()
errors['Predicted'] = y_pred[incorrect]

# Print the number of errors and 10 examples for M2
print('\nTotal misclassified:', len(errors))
print(errors.head(10).to_string(index=False))

# Save all incorrect predictions for further analysis
errors.to_csv('outputs/naive_bayes_errors.csv', index=False)
