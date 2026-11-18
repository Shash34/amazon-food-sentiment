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
