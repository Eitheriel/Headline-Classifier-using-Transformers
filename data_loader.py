import pandas as pd

def load_ag_news(filepath):
    # Load CSV
    df = pd.read_csv(filepath)

    # Rename and clean
    df = df.rename(columns={"Class Index": "label", "Title": "title", "Description": "description"})
    df["label"] = df["label"] - 1  # Convert labels 1–4 to 0–3

    # Combine title and description
    df["text"] = df["title"].fillna('') + " " + df["description"].fillna('')

    # Label dictionary
    label_dict = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    return df["text"].tolist(), df["label"].tolist(), label_dict
