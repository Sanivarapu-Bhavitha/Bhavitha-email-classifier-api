import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    df = pd.read_csv("data/combined_emails_with_natural_pii.csv")
    X = df["email"]
    y = df["type"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, "email_classifier.pkl")
    print("✅ Logistic Regression model trained and saved.")
