from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_loader import load_ag_news
from dataset import NewsDataset
from train import train_model
import torch
import os
import pickle

def main():
    model_name = "bert-base-uncased"

    # Load and prepare data
    train_texts, train_labels, label_dict = load_ag_news("data/ag_news/train.csv")
    val_texts, val_labels, _ = load_ag_news("data/ag_news/test.csv")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create datasets
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer)

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_dict))

    # Train the model
    trained_model = train_model(model, train_dataset, val_dataset, epochs=3, batch_size=8)

    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(trained_model.state_dict(), "models/bert_model.pt")

    # Save label dictionary
    with open("models/label_dict.pkl", "wb") as f:
        pickle.dump(label_dict, f)

    print("Model was trained and saved.")

if __name__ == "__main__":
    main()