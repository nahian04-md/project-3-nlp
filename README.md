# 🔍 TruthLens: Fake News Detection with NLP

A comprehensive Natural Language Processing project that implements and compares multiple approaches for detecting fake news, from classical machine learning models to state-of-the-art transformer-based architectures using Hugging Face.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models & Approaches](#models--approaches)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## Overview

TruthLens tackles the critical challenge of automated fake news detection using two complementary approaches. The project demonstrates the evolution from traditional ML techniques to modern deep learning solutions, providing insights into the trade-offs between interpretability, performance, and computational requirements.

The project achieves **98.6% accuracy** with the fine-tuned BERT model and **93.76% accuracy** with the optimized Tuned Logistic Regression model.

## Project Structure

```
TruthLens/
├── fake_news_classification_with_visualizations.ipynb  # Classical ML approach
├── trails_with_pretrained.ipynb                        # Transformer-based approach
├── dataset/
│   ├── training_data.csv
│   └── testing_data.csv
├── results/
│   └── testing_data_predictions.csv
└── README.md
```

## Notebooks

### 1. Classical Machine Learning Approach
**File:** `fake_news_classification_with_visualizations.ipynb`

This notebook implements traditional NLP techniques with extensive experimentation:

**Pipeline:**
- Text preprocessing (lowercasing, URL removal, tokenization, stop word removal)
- Feature extraction using TF-IDF vectorization with n-grams
- Model comparison and hyperparameter tuning
- Comprehensive visualizations and analysis

**Models Evaluated:**
- XGBoost
- Logistic Regression
- Random Forest Classifier
- Linear Support Vector Machine (SVM)
- Multinomial Naive Bayes

**Best Model:** Linear (SVM) model with TF-IDF (1,2)-grams
- **Accuracy:** 93.7 %
- **F1-Score:** 93.5 %
- **Features:** 15,000 max features

### 2. Transformer-Based Approach

**Pipeline:**
- Data cleaning and duplicate removal
- Pre-trained model evaluation
- Fine-tuning with custom dataset
- Performance evaluation and visualization

**Best Model:** Logistic Regression model with TF-IDF (1,2)-grams
- **Accuracy:** 93.9 %
- **F1-Score:** 93.8 %
- **Features:** 15,000 max features

**Models Used:**
- Pre-trained: `jy46604790/Fake-News-Bert-Detect` (RoBERTa-based)
- Fine-tuned BERT for sequence classification

**Best Model:** Fine-tuned BERT
- **Accuracy:** 98.6%
- **F1-Score:** 98.6%
- **Precision:** 98.6%
- **Recall:** 98.6%


### Prerequisites

```bash
# Python 3.13.8+
python --version
```

### Dependencies

```bash
# Core dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# For transformer models
pip install torch transformers datasets tqdm

# Optional: For GPU acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu121
```


## Usage

### Running the Classical ML and Transformer-Based Approach Notebook

```python
# Open Jupyter Notebook
Natural_Language.ipynb

Pretrained_Model_Fake_News.ipynb

```

### Making Predictions


## Dataset

- **Training samples:** 34,152
- **Test samples:** 9,984
- **Classes:** Binary (0 = Real News, 1 = Fake News)
- **Format:** Tab-separated values (TSV)
- **Class distribution:** Relatively balanced (~51.5% Real, ~48.5% Fake)

## Models & Approaches

### Classical ML Approach

**Text Preprocessing:**
- Lowercasing
- URL and special character removal
- Tokenization using NLTK
- Stop word removal

**Feature Engineering:**
- TF-IDF Vectorization
- Unigrams and bigrams (1,2)-grams
- Maximum 15,000 features
- Document frequency filtering (min_df=2, max_df=0.95)

**Hyperparameter Tuning:**
- Grid Search with Cross-Validation
- Optimized regularization parameters
- Evaluated multiple kernels and classifiers

### Transformer Approach

**Model Architecture:**
- Base: `jy46604790/Fake-News-Bert-Detect`
- Architecture: RoBERTa for Sequence Classification


## Future Improvements

1. **Ensemble Methods** - Combine classical ML and transformer predictions
2. **Multi-lingual Support** - Extend to non-English news detection
3. **Real-time API** - Deploy as REST API service
4. **Active Learning** - Implement continuous learning pipeline
5. **Cross-domain Testing** - Evaluate model on different news sources


## Acknowledgments

- Hugging Face for the Transformers library
- The creators of the `jy46604790/Fake-News-Bert-Detect` model
- Ironhack for the educational support


