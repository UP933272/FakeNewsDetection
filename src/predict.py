import pickle
import time

# Load saved model
model = pickle.load(open("../models/model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))

print("----------------------------------")
print(" Fake News Detection Tool")
print("----------------------------------")
print("\n")
print("Welcome to the fake news detection tool")

text = input("Please enter text from a social media post to scan it for fake news:\n")



print("\nScanning...")
time.sleep(2)

# Convert text
text_vectorized = vectorizer.transform([text])

# Predict
prediction = model.predict(text_vectorized)[0]

if prediction == 0:
    print("\nResult: This is fake news ❌")
else:
    print("\nResult: This is not fake news ✅")
