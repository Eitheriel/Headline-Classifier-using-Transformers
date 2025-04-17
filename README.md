# News Headline Classifier using BERT
This project uses a pre-trained English BERT model (bert-base-uncased) to classify news headlines into four categories from the AG News dataset.

### Endpoints
main.py - starts training of the model and saves it after completion  
evaluate.py - evaluates the trained model - prints accuracy, classification report, and confusion matrix  
predict.py - script for predicting the category of a given news article

### Other Files
data_loader.py - loads, cleans and prepares data from CSV files  
dataset.py - defines the NewsDataset class used as input to PyTorch  
train.py - contains the train_model() function for model training

### Data and Model
The project expects the AG News dataset to be located in the data/ag_news/ folder as two CSV files: `train.csv` and `test.csv`.

The pre-trained model is automatically downloaded from the Hugging Face model hub during the first run. 
After that, it is cached locally in ~/.cache/huggingface/.  
After training, the model is saved into the models/ directory as bert_model.pt, along with the label dictionary saved as label_dict.pkl.

### Dataset Source
The dataset used in this project is a CSV version of the AG News corpus, downloaded from Kaggle:  
🔗 [AG News - News Topic Classification Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

The dataset contains news article headlines and descriptions categorized into four classes:
- World
- Sports
- Business
- Sci/Tech

The original dataset was introduced by Xiang Zhang et al. (2015), but this version has been made available as a preprocessed CSV for ease of use.

### Baseline

A simple baseline using TF-IDF and logistic regression is available in a separate folder.  
[See the baseline README](tf_idf_logistic_regression/README.md)

### Results
[Confusion Matrix](images/confusion_matrix_transformer.png) 

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| World       | 0.95      | 0.96   | 0.96     | 1900    |
| Sports      | 0.99      | 0.98   | 0.99     | 1900    |
| Business    | 0.91      | 0.92   | 0.92     | 1900    |
| Sci/Tech    | 0.93      | 0.92   | 0.92     | 1900    |
| **Accuracy**|           |        | **0.95** | **7600**|
| Macro avg   | 0.95      | 0.95   | 0.95     | 7600    |
| Weighted avg| 0.95      | 0.95   | 0.95     | 7600    |

### Interpretation
The model achieved an overall accuracy of 95%, which is considered very high for multi-class text classification.

The lowest performance was observed in the **Business** category (F1-score: 0.92), similar to the TF-IDF baseline. 
This suggests that the issue may not lie in understanding semantics or context, but rather in the vocabulary overlap 
between *Business* and *Sci/Tech* categories. This interpretation is supported by the confusion matrix, 
which shows increased misclassification between these two classes. 
It's also possible that some mislabels for these two classes exist in the dataset.

### Dependencies
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121  
pip install transformers datasets pandas scikit-learn matplotlib seaborn tqdm pynvml

### Motivation
This project was created as a personal exercise to get hands-on experience with large language models (LLMs) 
and commonly used NLP tools, as well as to gain practical experience with libraries like PyTorch, scikit-learn, and others.