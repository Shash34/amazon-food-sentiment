import pandas as pd

#Using the random seed 42 and samples 50000, then writes sample to a new CSV
df = pd.read_csv('Reviews.csv')
sample = df.sample(n=50000, random_state=42)
sample.to_csv('group_sample_seed42.csv', index=False)
print('Saved group_sample_seed42.csv:', len(sample), 'reviews')
