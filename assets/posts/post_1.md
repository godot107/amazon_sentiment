# Sentiment Analysis with TF-IDF and Logistic Regression

This notebook demonstrates a baseline approach to sentiment analysis using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and Logistic Regression. While simpler than modern deep learning approaches, this method provides a solid foundation for understanding text classification.

## 📚 Concepts

### TF-IDF Vectorization
TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents (corpus). It balances term frequency with word uniqueness, ensuring that frequently occurring but unimportant words do not dominate text representations.

#### Components of TF-IDF:
1. **Term Frequency (TF)**: Measures how often a term appears in a document.
   - Formula: \( TF(t,d) = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in document } d} \)
2. **Inverse Document Frequency (IDF)**: Measures how important or unique a term is across all documents.
   - Formula: \( IDF(t) = \log\left(\frac{N}{1 + DF(t)}\right) \)
   - Where:
     - \( N \) = total number of documents
     - \( DF(t) \) = number of documents containing term \( t \)

#### Why Use TF-IDF?
- **Downweights common words** (e.g., "the", "is") while highlighting informative terms.
- **Emphasizes distinctive words**, making text classification more effective.
- **Handles varying document lengths** by normalizing term frequencies.

### Logistic Regression
A linear classification model that predicts probabilities using the sigmoid function:
\[ P(Y=1|X) = \frac{1}{1 + e^{- (\beta_0 + \beta_1 X_1 + ... + \beta_n X_n )}} \]

#### Key Advantages:
- **Handles high-dimensional, sparse data** well (like TF-IDF features).
- **Interpretable**: Provides feature importance via learned weights.
- **Efficient and scalable**, making it a great baseline model.

## 🎯 Implementation Details

### 1. Data Preprocessing
- Text cleaning (removing punctuation, lowercasing, etc.).
- Handling missing values.
- Converting ratings to sentiment labels (negative, neutral, positive).

### 2. Feature Engineering
- **TF-IDF vectorization** to convert text into numerical features.
- **Stop word removal** to eliminate common words that add little value.
- **Feature limitation**: Restricting to the top 10,000 most frequent terms to reduce sparsity.

### 3. Model Training
- **Train-test split** with stratification to maintain class distribution.
- **Class weight balancing** to handle imbalanced datasets.
- **Cross-validation** for robust performance evaluation.

## ⚠️ Limitations and Pitfalls

### 1. Bag-of-Words Limitations
- **Loses word order information** (e.g., "not happy" and "happy" are treated similarly).
- **Cannot capture context** beyond individual words.
- **Struggles with negations** (e.g., "not good" vs. "good").

### 2. Vocabulary Issues
- **Out-of-vocabulary (OOV) words** in test data may not be well represented.
- **Sparse feature matrix** leads to memory inefficiency.
- **Fixed vocabulary**: Adding new words requires retraining.

### 3. Class Imbalance
- **Biased predictions** toward the majority class.
- **Need for proper evaluation metrics** beyond accuracy.
- **Importance of stratified sampling** to maintain label proportions.

## 🚀 Potential Improvements

### 1. Text Preprocessing
- More sophisticated text cleaning (handling contractions, special characters, etc.).
- **Lemmatization** instead of stemming for better root-word representation.
- **Explicit negation handling** (e.g., replacing "not happy" with "not_happy").
- **Emoji and emoticon processing** for sentiment-rich symbols.

### 2. Feature Engineering
- **N-gram features** (bigrams, trigrams) to capture phrase-level information.
- **Custom stop words** tailored to the domain.
- **Part-of-Speech (POS) tagging** to identify important words.
- **Named Entity Recognition (NER)** for identifying entities.

### 3. Model Enhancements
- **Ensemble methods** (e.g., Random Forest, XGBoost) for better performance.
- **Feature selection** to remove noisy terms.
- **Hyperparameter tuning** using GridSearchCV.
- **More sophisticated cross-validation strategies** (e.g., stratified k-fold).

### 4. Advanced Techniques
- **Word embeddings** (Word2Vec, GloVe) for dense representations.
- **Deep learning models** (BERT, RoBERTa) for context-aware sentiment analysis.
- **Transfer learning** to leverage pre-trained language models.

## 📊 Evaluation Metrics

### Key Metrics:
- **Accuracy**: Overall correctness of the model.
- **F1-score**: Balances precision and recall, especially for imbalanced datasets.
- **Precision & Recall**: Measures of positive prediction quality and coverage.
- **Confusion Matrix**: Provides insights into misclassification.
- **ROC-AUC Curve**: Evaluates classification threshold performance.

## 🔍 Use Cases

This approach is particularly useful for:
1. **Baseline model development** before exploring deep learning.
2. **Quick prototyping** with minimal computational cost.
3. **Small to medium-sized datasets** where deep learning isn't necessary.
4. **Interpretability-focused applications**, such as legal or financial text classification.
5. **Limited computational resources**, where traditional ML is preferable.

## 📝 Next Steps

To further improve sentiment analysis:
1. Experiment with advanced feature engineering techniques.
2. Implement more robust text preprocessing strategies.
3. Optimize hyperparameters and cross-validation methods.
4. Compare performance against transformer-based models.
5. Analyze misclassified examples to refine the approach.

## 📚 References

1. [Scikit-learn TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
2. *Introduction to Information Retrieval* by Manning, Raghavan, and Schütze
3. *Pattern Recognition and Machine Learning* by Christopher Bishop
