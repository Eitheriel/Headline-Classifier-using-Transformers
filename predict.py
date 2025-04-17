import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle


def predict(text, model_path="models/bert_model.pt", label_dict_path="models/label_dict.pkl"):
    # Load label dictionary
    with open(label_dict_path, "rb") as f:
        label_dict = pickle.load(f)
    id2label = label_dict  # e.g., {0: 'World', 1: 'Sports', ...}

    # Load tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_dict))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()

    # Preprocess input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    return id2label[predicted_class_id]

# Example usage
text = "NASA confirms discovery of Earth-like exoplanet"
predicted = predict(text)
print("Predicted category:", predicted)