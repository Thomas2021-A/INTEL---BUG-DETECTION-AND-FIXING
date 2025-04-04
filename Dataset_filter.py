import pandas as pd
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv("final_fixed_bug_dataset.csv")

# Check class distribution
print("Original class distribution:")
print(df['Project'].value_counts())

# Find the minimum class count
min_count = df['Project'].value_counts().min()

# Balance the dataset
balanced_df = pd.concat([
    resample(df[df['Project'] == project], replace=True, n_samples=min_count, random_state=42)
    for project in df['Project'].unique()
])

# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset
balanced_df.to_csv("balanced_bug_dataset.csv", index=False)

print("Balanced dataset saved as balanced_bug_dataset.csv")
