import pandas as pd

# Use the exact Group Alpha sample BEFORE filtering.
df = pd.read_csv('group_sample_seed42.csv')
print('Original sample:', len(df))
df = df.dropna(subset=['Text', 'Score'])
df = df.drop_duplicates(subset=['UserId', 'ProductId', 'Text'])
df = df[df['Score'] != 3].copy()
df['Sentiment'] = (df['Score'] >= 4).astype(int)
df.to_csv('cleaned_reviews.csv', index=False)
print('Cleaned reviews:', len(df))
print('Positive:', (df['Sentiment'] == 1).sum())
print('Negative:', (df['Sentiment'] == 0).sum())
