import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from tqdm import tqdm

# Load dataset
file_path = os.path.join(os.path.dirname(__file__), "passwords.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the file path.")

print("Loading dataset...")
df = pd.read_csv(file_path).dropna()
print("Dataset loaded successfully.")

# Use a smaller subset of data for initial testing
df = df.sample(n=10000, random_state=42)

# Convert passwords to numerical features
print("Converting passwords to numerical features...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
X = vectorizer.fit_transform(df["password"])
y = df["strength"]
print("Conversion complete.")

# Split data
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete.")

# Train model with progress bar
print("Training model...")
clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("Model training complete.")

# Test model
print("Testing model...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model & vectorizer
print("Saving model and vectorizer...")
joblib.dump(clf, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved successfully.")
