import pandas as pd
from joblib import load
from util import preprocess
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Paths
DATA_PATH = Path("../data/ag_news/test.csv")
MODEL_PATH = Path("../models/tf_idf_logistic_regression")

# Load test data
df = pd.read_csv(DATA_PATH)

# Rename columns and shift labels from 1-4 to 0-3
df = df.rename(columns={"Class Index": "label", "Title": "title", "Description": "description"})
df["label"] = df["label"] - 1

# Combine title and description into a single text column
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

# Clean the text
df["text_clean"] = df["text"].apply(preprocess)

# Load model and vectorizer
model = load(MODEL_PATH / "model.joblib")
vectorizer = load(MODEL_PATH / "vectorizer.joblib")

# Transform the text
X_test = vectorizer.transform(df["text_clean"])
y_test = df["label"]

# Predict
y_pred = model.predict(X_test)

# Evaluation report
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["World", "Sports", "Business", "Sci/Tech"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["World", "Sports", "Business", "Sci/Tech"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()