import re

def preprocess(text):
    text = str(text).lower()

    # Remove content in parentheses (e.g. (AP), (Reuters))
    text = re.sub(r"\([^)]*\)", "", text)

    # Remove agency prefixes
    text = re.sub(r"^[a-z0-9\.\-]+(\s+)?\-\s+", "", text)

    # Remove all characters except letters and spaces
    text = re.sub(r"[^a-zá-ž\s]", "", text)

    # Replace multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()
