import pandas as pd
from sklearn.utils import shuffle

# Load Kaggle files
fake_df = pd.read_csv("../data/raw/Fake.csv")
true_df = pd.read_csv("../data/raw/True.csv")

# Combine title + text (better features)
fake_df["text"] = fake_df["title"] + " " + fake_df["text"]
true_df["text"] = true_df["title"] + " " + true_df["text"]

# Keep only needed column
fake_df = fake_df[["text"]].copy()
true_df = true_df[["text"]].copy()

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

# Combine and shuffle
data = pd.concat([fake_df, true_df], ignore_index=True)
data = shuffle(data, random_state=42)

# Remove empty rows
data = data.dropna(subset=["text"]).reset_index(drop=True)

# Save new dataset version
data.to_csv("../data/raw/dataset_v2.csv", index=False)

print("Combined dataset created successfully.")
print(data.head())
print(data["label"].value_counts())