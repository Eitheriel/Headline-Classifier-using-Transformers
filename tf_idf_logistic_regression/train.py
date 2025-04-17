import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump
from pathlib import Path
from util import preprocess

# Paths
DATA_PATH = Path("../data/ag_news/train.csv")
MODEL_OUT_PATH = Path("../models/tf_idf_logistic_regression")
MODEL_OUT_PATH.mkdir(parents=True, exist_ok=True)

# Load training data
df = pd.read_csv(DATA_PATH)

# Rename columns and shift labels from 1-4 to 0-3
df = df.rename(columns={"Class Index": "label", "Title": "title", "Description": "description"})
df["label"] = df["label"] - 1

# Combine title and description
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

# Clean the text
df["text_clean"] = df["text"].apply(preprocess)

# Initialize TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["text_clean"])
y = df["label"]

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the model and vectorizer
dump(model, MODEL_OUT_PATH / "model.joblib")
dump(vectorizer, MODEL_OUT_PATH / "vectorizer.joblib")

print("Model and vectorizer were saved to:", MODEL_OUT_PATH)