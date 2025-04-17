from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import NewsDataset
from data_loader import load_ag_news
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pickle

def evaluate(model, val_dataset, label_dict):
    model.eval()
    model.cuda()
    dataloader = DataLoader(val_dataset, batch_size=16)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Decode labels
    id2label = label_dict  # {0: "World", ...}
    labels = list(id2label.keys())
    label_names = [id2label[i] for i in labels]

    print("Accuracy Score:", accuracy_score(all_labels, all_preds))
    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=label_names,
        zero_division=0
    ))
    return all_labels, all_preds

def plot_confusion_matrix(true_labels, pred_labels, label_dict):
    label_names = list(label_dict.values())
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Load validation data
val_texts, val_labels, label_dict = load_ag_news("data/ag_news/test.csv")

# Tokenizer and dataset
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
val_dataset = NewsDataset(val_texts, val_labels, tokenizer)

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_dict))
model.load_state_dict(torch.load("models/bert_model.pt"))
model.cuda()

# Evaluate and visualize
true_labels, pred_labels = evaluate(model, val_dataset, label_dict)
plot_confusion_matrix(true_labels, pred_labels, label_dict)
