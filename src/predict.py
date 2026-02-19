import pickle
import time

# Load saved model
model = pickle.load(open("../models/model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))

print("----------------------------------")
print(" Fake News Detection Tool")
print("----------------------------------")

text = input("\nEnter text to analyse:\n> ")

print("\nScanning...")
time.sleep(2)

# Convert text
text_vectorized = vectorizer.transform([text])

# Predict
prediction = model.predict(text_vectorized)[0]

if prediction == 0:
    print("\nResult: FAKE NEWS ❌")
else:
    print("\nResult: REAL NEWS ✅")
