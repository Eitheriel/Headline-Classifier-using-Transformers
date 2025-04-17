from joblib import load
from util import preprocess
from pathlib import Path

# Load model and vectorizer
MODEL_PATH = Path("../models/tf_idf_logistic_regression")
model = load(MODEL_PATH / "model.joblib")
vectorizer = load(MODEL_PATH / "vectorizer.joblib")

# Label mapping (AG News: 0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech)
label_names = ["World", "Sports", "Business", "Sci/Tech"]

def classify(title, description=""):
    text = preprocess(title + " " + description)
    vector = vectorizer.transform([text])
    # Predict
    prediction = model.predict(vector)
    return label_names[prediction[0]]

# Example inputs
examples = [
    ("NASA discovers new exoplanet", "Astronomers at NASA announced the discovery of a potentially habitable planet."),
    ("Stock market crashes amid economic concerns", ""),
    ("Real Madrid wins Champions League final", ""),
    ("Apple unveils latest iPhone model", "The new phone features improved battery life and camera."),
]

# Run inference
for title, description in examples:
    category = classify(title, description)
    print(f"{title} → {category}")