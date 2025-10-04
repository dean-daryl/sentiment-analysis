# Sentiment Analysis Project

A comprehensive text classification system comparing traditional machine learning and deep learning approaches for sentiment analysis.

## 📋 Project Overview

This project implements and compares various machine learning models for sentiment analysis, including:

- **Traditional ML Models**: Logistic Regression, SVM, Naive Bayes
- **Deep Learning Models**: RNN, LSTM, GRU, Bidirectional LSTM
- **Feature Engineering**: TF-IDF, Bag of Words, Word2Vec, GloVe embeddings

## 🎯 Assignment Requirements

- Compare traditional ML vs deep learning approaches
- Use publicly available datasets (IMDB, Twitter, Amazon reviews)
- Conduct comprehensive EDA with statistical analysis and visualizations
- Apply various text preprocessing and embedding techniques
- Include 2+ experiment tables with parameter variations
- Evaluate using appropriate metrics (MSE, cross-entropy, accuracy, F1-score)
- Submit PDF report + GitHub repository

## 📁 Project Structure

```
sentiment-analysis/

├── .ipynb_checkpoints/         # Jupyter notebook checkpoints
├── results/                    # Results from experiments
├── visualization/              # Visualizations and plots
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── sentiment_analysis.ipynb    # Main notebook

```

## 🚀 Getting Started

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

### 4. Start Jupyter Notebook

```bash
jupyter notebook sentiment_analysis_template.ipynb
```

## 📊 Dataset Options

Choose one of the following datasets for your analysis:

### 1. IMDB Movie Reviews
- **Size**: 50,000 reviews
- **Classes**: Binary (positive/negative)
- **Source**: Keras datasets or download from Stanford
- **Access**: `tensorflow.keras.datasets.imdb`

### 2. Twitter Sentiment Analysis
- **Options**: Sentiment140, Twitter US Airline Sentiment
- **Classes**: Binary or multi-class
- **Source**: Kaggle, academic datasets

### 3. Amazon Product Reviews
- **Size**: Various sizes available
- **Classes**: 5-star ratings (can be converted to binary)
- **Source**: Amazon review datasets, Kaggle

### 4. Custom Dataset
- Ensure it has text and sentiment labels
- Minimum 1000+ samples recommended

## 🔧 Implementation Guide

### Phase 1: Data Exploration (Week 1)
- [ ] Load and explore chosen dataset
- [ ] Perform EDA with visualizations
- [ ] Analyze class distribution and text characteristics
- [ ] Create word frequency analysis

### Phase 2: Preprocessing (Week 1-2)
- [ ] Implement text cleaning pipeline
- [ ] Handle missing values and duplicates
- [ ] Apply tokenization and normalization
- [ ] Remove stopwords and perform stemming/lemmatization

### Phase 3: Feature Engineering (Week 2)
- [ ] Implement TF-IDF vectorization
- [ ] Create Bag of Words representations
- [ ] Train Word2Vec embeddings
- [ ] Optional: Use pre-trained GloVe embeddings

### Phase 4: Traditional ML Models (Week 2-3)
- [ ] Implement Logistic Regression
- [ ] Implement SVM with different kernels
- [ ] Implement Naive Bayes
- [ ] Perform hyperparameter tuning
- [ ] Create Experiment Table 1

### Phase 5: Deep Learning Models (Week 3-4)
- [ ] Prepare sequences for neural networks
- [ ] Implement LSTM model
- [ ] Implement GRU model
- [ ] Implement Bidirectional LSTM
- [ ] Experiment with architectures
- [ ] Create Experiment Table 2

### Phase 6: Evaluation and Analysis (Week 4)
- [ ] Compare all models using multiple metrics
- [ ] Create confusion matrices and classification reports
- [ ] Analyze results and discuss findings
- [ ] Suggest improvements and future work

## 📈 Evaluation Metrics

### Required Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **MSE**: Mean Squared Error
- **Cross-Entropy Loss**: Logarithmic loss for probability predictions

### Metric Justifications
Document why each metric is appropriate for your specific task and dataset characteristics.

## 🧪 Experiment Tables

### Experiment 1: Traditional ML Hyperparameter Tuning
| Model | Feature Type | Hyperparameters | Accuracy | Precision | Recall | F1-Score | MSE | Cross-Entropy |
|-------|--------------|-----------------|----------|-----------|--------|----------|-----|---------------|

### Experiment 2: Deep Learning Architecture Variations
| Model | Embedding Dim | Hidden Units | Dropout | Batch Size | Epochs | Accuracy | F1-Score | Cross-Entropy |
|-------|---------------|--------------|---------|------------|--------|----------|----------|---------------|
| GRU   | 128           | 64           | 0.5     | 32         | 3      | 0.882    | 0.885    | 0.296         |


## 👥 Team Collaboration

### Team Member Responsibilities
- **Member 1**: Data collection, EDA, preprocessing
- **Member 2**: Traditional ML models, hyperparameter tuning
- **Member 3**: Deep learning models, architecture experiments
- **Member 4**: Evaluation, analysis, report writing

### Git Workflow
1. Create feature branches for different components
2. Use descriptive commit messages
3. Review code before merging
4. Document all changes

## 📝 Report Requirements

### PDF Report Sections
1. **Introduction**: Problem statement and objectives
2. **Dataset**: Description, selection justification, EDA
3. **Methodology**: Preprocessing, feature engineering, models
4. **Experiments**: Two experiment tables with analysis
5. **Results**: Performance comparison and evaluation
6. **Discussion**: Key findings, limitations, improvements
7. **Conclusion**: Summary and future work
8. **References**: All sources used
9. **Team Contributions**: Individual member contributions

## 🚨 Common Issues and Solutions

### Environment Issues
- **NLTK download errors**: Run downloads in Python script, not notebook
- **TensorFlow GPU issues**: Ensure CUDA compatibility
- **Memory errors**: Reduce batch size or use data generators

### Data Issues
- **Large datasets**: Use sampling for initial development
- **Imbalanced classes**: Consider stratified sampling and appropriate metrics
- **Text encoding**: Handle UTF-8 encoding issues

### Model Issues
- **Overfitting**: Use dropout, early stopping, cross-validation
- **Poor performance**: Check preprocessing, try different features
- **Training time**: Start with smaller models, use GPU if available

## 📚 Resources

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide)
- [NLTK Documentation](https://www.nltk.org/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)

### Tutorials
- [Text Classification with TF-IDF](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [LSTM for Text Classification](https://www.tensorflow.org/text/tutorials/text_classification_rnn)
- [Word2Vec Tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)

### Datasets
- [IMDB Reviews](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Sentiment140 Twitter](http://help.sentiment140.com/for-students)
- [Amazon Reviews](https://nijianmo.github.io/amazon/index.html)

## 📋 Submission Checklist

### Before Final Submission
- [ ] All code cells execute without errors
- [ ] Results are reproducible (set random seeds)
- [ ] Code is well-documented with comments
- [ ] All visualizations are clear and labeled
- [ ] Experiment tables are complete
- [ ] Performance metrics are calculated correctly
- [ ] Discussion section analyzes results thoroughly
- [ ] Team contributions are documented
- [ ] README is updated with final results
- [ ] Repository is organized and clean
- [ ] PDF report is professionally formatted

### Final Deliverables
1. **GitHub Repository**: Complete code, data, results
2. **PDF Report**: Comprehensive analysis and findings
3. **Presentation**: Summary of key results (if required)

## 📞 Contact

For questions about this project template:
- Create GitHub issues for technical problems
- Contact team members for collaboration questions
- Refer to course materials for assignment-specific requirements

---

**Good luck with your sentiment analysis project! 🎉**
