import pandas as pd

df = pd.read_csv('Reviews.csv')
sample = df.sample(n=50000, random_state=42)
sample.to_csv('group_sample_seed42.csv', index=False)
print('Saved group_sample_seed42.csv:', len(sample), 'reviews')
