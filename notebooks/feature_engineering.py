import os
import re
import html
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import save_npz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Inputs: cleaned_reviews.csv produced by preprocess.py
# All modelers must reuse these exact train/test splits.
df = pd.read_csv('../data/cleaned_reviews.csv')
os.makedirs('outputs', exist_ok=True)

def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    return text.strip()

df['Text'] = df['Text'].apply(clean_text)
df = df[df['Text'].str.len() > 0].copy()
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['Sentiment']
)
train_df.to_csv('outputs/train.csv', index=False)
test_df.to_csv('outputs/test.csv', index=False)

bow = CountVectorizer(stop_words='english', max_features=10000)
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_bow = bow.fit_transform(train_df['Text'])
X_test_bow = bow.transform(test_df['Text'])
X_train_tfidf = tfidf.fit_transform(train_df['Text'])
X_test_tfidf = tfidf.transform(test_df['Text'])

for filename, mat in [('X_train_bow.npz', X_train_bow), ('X_test_bow.npz', X_test_bow),
                      ('X_train_tfidf.npz', X_train_tfidf), ('X_test_tfidf.npz', X_test_tfidf)]:
    save_npz(os.path.join('outputs', filename), mat)
joblib.dump(bow, 'outputs/bow_vectorizer.joblib')
joblib.dump(tfidf, 'outputs/tfidf_vectorizer.joblib')

counts = np.asarray(X_train_bow.sum(axis=0)).ravel()
terms = bow.get_feature_names_out()
indices = np.argsort(counts)[-10:][::-1]
print('Training reviews:', len(train_df))
print('Testing reviews:', len(test_df))
print('BoW vocabulary size:', len(bow.vocabulary_))
print('TF-IDF vocabulary size:', len(tfidf.vocabulary_))
print('\nTop 10 most frequent words:')
for i in indices:
    print(terms[i], int(counts[i]))

plt.figure(figsize=(9, 5))
plt.bar(terms[indices], counts[indices])
plt.title('Top 10 Most Frequent Words in Training Reviews')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/word_frequency.png', dpi=150)
plt.close()
print('\nFeature engineering complete!')
print('Files saved in the outputs folder.')
