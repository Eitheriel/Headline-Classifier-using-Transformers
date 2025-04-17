# News Headline Classifier using TF-IDF

This project implements a traditional text classification pipeline using TF-IDF vectorization for text representation 
and logistic regression as the classifier. The model is trained to categorize news headlines from the AG News dataset into four predefined topics.

### Endpoints
train.py - trains the model and saves both the classifier and the TF-IDF vectorizer  
evaluate.py - evaluates the trained model, prints accuracy, classification report, and confusion matrix  
predict.py - performs category prediction for custom input text

### Other Files
utils.py - contains the preprocess() function used for basic text cleaning

### Preprocessing

Unlike transformer-based models that use `AutoTokenizer` to handle lowercasing, special characters, and tokenization internally, 
the TF-IDF approach requires manual text preprocessing before vectorization. The script uses a custom preprocess() function (see utils.py) 
that converts text to lowercase and removes excess whitespace, special characters, source tags, 
and content in parentheses (e.g., “(AP)”, “(Reuters)”).

### Data and Model
This baseline uses the same AG News dataset as the transformer-based model (see [README](../README.md#data-and-model)).  
After training, the model and vectorizer are saved as `model.joblib` and `vectorizer.joblib` in models/tf_idf_logistic_regression.

### Results
[Confusion Matrix](../images/confusion_matrix_tf_idf.png)  

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| World       | 0.92      | 0.89   | 0.91     | 1900    |
| Sports      | 0.95      | 0.97   | 0.96     | 1900    |
| Business    | 0.87      | 0.87   | 0.87     | 1900    |
| Sci/Tech    | 0.88      | 0.88   | 0.88     | 1900    |
| **Accuracy**|           |        | **0.90** | **7600**|
| Macro avg   | 0.90      | 0.90   | 0.90     | 7600    |
| Weighted avg| 0.90      | 0.90   | 0.90     | 7600    |


### Interpretation

The model achieved an overall accuracy of 90% on the AG News test set, with strong performance across all categories.

Best performance was seen in the **Sports category**, likely due to consistent terminology in sports news.  

The lowest performance was seen in the **Business** category, with an F1-score of 0.87. 
This may be caused by higher topical diversity, or more abstract/ambiguous headlines. 
Additional confusion was observed between the **Business** and **Sci/Tech** categories 
in the confusion matrix. This likely stems from overlapping vocabulary and thematic similarity.

The results indicate good generalization and robustness for real-world news classification.

### Dependencies
pip install pandas scikit-learn joblib matplotlib