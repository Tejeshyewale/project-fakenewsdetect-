import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load datasets
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Check the structure of the datasets
print("Fake DataFrame Head:\n", fake_df.head())
print("Real DataFrame Head:\n", real_df.head())

# Add label: 0 for fake, 1 for real
fake_df["label"] = 0
real_df["label"] = 1

# Combine datasets and shuffle
data = pd.concat([fake_df, real_df], ignore_index=True)
data = data.sample(frac=1, random_state=41).reset_index(drop=True)

# Check for null values
print("Null Values:\n", data.isnull().sum())

# Drop rows with missing text
data = data.dropna(subset=['text'])

# Check the data
print("Data Info:\n", data.info())
print("Label Distribution:\n", data['label'].value_counts())

# Use the 'text' column as input features
X = data['text']
y = data['label']

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Check the shape of the vectorized data
print(f"Shape of X_train_tfidf: {X_train_tfidf.shape}")
print(f"Shape of X_test_tfidf: {X_test_tfidf.shape}")

# Train a Logistic Regression model
model = LogisticRegression(max_iter=500)  # Increased iterations
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Output accuracy and classification report
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Return accuracy and classification report
accuracy, report
