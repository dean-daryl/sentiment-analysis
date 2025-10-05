# Sentiment Analysis

**Team**: Group 10
**Course**: Machine Learning Techniques - I    

A text classification system comparing traditional machine learning and deep learning approaches for sentiment analysis on the IMDB Movie Reviews dataset.

## Project Overview

This project implements and compares four machine learning models for binary sentiment classification:

- **Traditional ML Models**: Logistic Regression, Support Vector Machine (SVM)
- **Deep Learning Models**: Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU)
- **Feature Engineering**: TF-IDF vectorization for ML models, sequential tokenization for DL models
- **Dataset**: IMDB Movie Reviews (50,000 reviews)

## Key Results

**Best Model Performance:**
- **SVM (C=0.1)**: 88.90% accuracy, 89.11% F1-score
- **Logistic Regression (C=1.0)**: 88.79% accuracy, 88.99% F1-score  
- **LSTM (RMSprop)**: 88.32% accuracy, 88.41% F1-score
- **GRU (RMSprop)**: 88.02% accuracy, 88.57% F1-score

## Project Structure

```
sentiment-analysis/
├── report/
│   └── report.md                           
├── results/
│   ├── gru_experiment_results.csv          
│   ├── lstm_experiment_results.csv          
│   ├── lr_experiment_results.csv           
│   └── SVM Experiment Results - Sentiment Analysis.csv
├── visualization/
│   ├── gru_cm.png                         
│   ├── gru_training_curves.png            
│   ├── lstm_cm.png                         
│   ├── lstm_training_curves.png            
│   ├── lr_cm.png                          
│   ├── svm_cm.png                          
│   ├── review_length_distribution.png     
│   ├── review_length_by_sentiment.png      
│   ├── sentiment_distribution.png          
│   ├── wordclouds_side_by_side.png         
│   └── SVM Hyperparameter Plot from Sentiment Analysis.png
├── README.md                               
├── requirements.txt                        
└── sentiment_analysis.ipynb               
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/dean-daryl/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Run the Analysis

```bash
jupyter notebook sentiment_analysis.ipynb
```

## Dataset Information

### IMDB Movie Reviews Dataset
- **Source**: [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 movie reviews (25,000 positive, 25,000 negative)
- **Classes**: Binary sentiment classification (positive/negative)
- **After preprocessing**: 49,582 unique reviews (418 duplicates removed)
- **Balance**: Perfectly balanced dataset ensuring unbiased model training
- **Format**: CSV with 'review' text and 'sentiment' labels

### Data Quality
- **Missing Values**: None detected in dataset
- **Duplicates**: 418 removed to prevent data leakage
- **Average Review Length**: ~1,300 characters
- **Text Preprocessing**: HTML tag removal, lowercasing, tokenization, stopword removal

## Experimental Results

### Experiment 1: Logistic Regression Hyperparameter Tuning
| C Value | Accuracy | F1-Score | Model Complexity |
|---------|----------|----------|------------------|
| 0.1     | 0.8732   | 0.8768   | High Regularization |
| **1.0** | **0.8879** | **0.8899** | **Optimal** |
| 10.0    | 0.8830   | 0.8845   | Low Regularization |

### Experiment 2: SVM Hyperparameter Analysis  
| C Value | Accuracy | F1-Score | Training Time |
|---------|----------|----------|---------------|
| **0.1** | **0.8890** | **0.8911** | Fast |
| 1.0     | 0.8814   | 0.8829   | Medium |
| 10.0    | 0.8687   | 0.8701   | Slow |

### Experiment 3: LSTM Architecture Variations
| Epochs | Batch Size | Optimizer | Accuracy | F1-Score |
|--------|------------|-----------|----------|----------|
| 3      | 32         | Adam      | 0.8780   | 0.8780   |
| **3**  | **32**     | **RMSprop** | **0.8832** | **0.8841** |
| 5      | 64         | Adam      | 0.8611   | 0.8655   |

### Experiment 4: GRU Comprehensive Analysis
| Epochs | Batch Size | Optimizer | Accuracy | Precision | Recall | F1-Score | Cross-Entropy |
|--------|------------|-----------|----------|-----------|--------|----------|---------------|
| **3**  | **32**     | **RMSprop** | **0.8802** | **0.8500** | **0.9245** | **0.8857** | **0.2916** |
| 5      | 64         | RMSprop   | 0.8809   | 0.8646    | 0.9044 | 0.8840   | 0.3389   |
| 3      | 32         | Adam      | 0.8770   | 0.8934    | 0.8572 | 0.8749   | 0.3064   |



## Model Performance Comparison

| Model | Best Accuracy | Best F1-Score | Training Time | Complexity | Memory Usage |
|-------|---------------|---------------|---------------|------------|--------------|
| **SVM** | **88.90%** | **89.11%** | Fast | Low | Low |
| **Logistic Regression** | **88.79%** | **88.99%** | Very Fast | Very Low | Very Low |
| **LSTM** | **88.32%** | **88.41%** | Slow | High | High |
| **GRU** | **88.02%** | **88.57%** | Medium | Medium | Medium |

### Key Findings
1. **Traditional ML models outperformed deep learning** on this dataset
2. **SVM with strong regularization (C=0.1)** achieved the best overall performance
3. **Deep learning models** showed competitive results but with higher computational cost
4. **All models achieved >88% accuracy**, indicating the effectiveness of the preprocessing pipeline

## Team Contributions

**Team**: Group 10

### Individual Contributions
- **Dean Daryl Murenzi**: SVM model, Dataset acquisition, EDA implementation, hyperparameter tuning
- **Nelly Iyabikoze**: GRU ML model, Performance evaluation, visualizations creation, confusion matrix analysis  
- **Excel Asaph**: Logistic Regression and LSTM model, Dataset acquisition, preprocessing pipeline

## Report

The full academic report is available at [`report/report.pdf`](report/report.pdf)