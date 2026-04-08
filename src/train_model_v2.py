import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

print("Loading dataset...")

# Load dataset
data = pd.read_csv("../data/raw/dataset_v2.csv")

X = data["text"]
y = data["label"]

print("Vectorising text...")

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

print("Training model...")

# Train classifier
model = LogisticRegression()
model.fit(X_vectorized, y)

print("Saving model...")

# Save trained model
pickle.dump(model, open("../models/model_v2.pkl", "wb"))
pickle.dump(vectorizer, open("../models/vectorizer_v2.pkl", "wb"))

print("Training complete!")