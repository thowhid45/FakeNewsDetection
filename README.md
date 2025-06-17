# Fake News Detection System
A machine learning-based web app built with Streamlit that classifies news articles as Fake or Genuine using Logistic Regression and Random Forest classifiers. It also includes text preprocessing, visualizations, and model evaluation metrics.

# Features
Manual News Detection: Input a news article and instantly classify it.

Word Cloud Visualization: Compare frequent terms in Fake vs Real news.

Model Performance Metrics: View precision, recall, F1-score, and confusion matrices.

Dual Model Evaluation: Logistic Regression and Random Forest Classifier.


├── app.py                # Streamlit app file
├── True.csv              # Dataset of genuine news articles
├── Fake.csv              # Dataset of fake news articles
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation


# Machine Learning Pipeline
Data Loading:

Merges True.csv and Fake.csv into one DataFrame with labels.

# Text Preprocessing:

Lowercasing, URL removal, HTML cleaning, punctuation and digit removal.

# Feature Engineering:

TF-IDF Vectorization with bi-grams and minimum document frequency.

# Model Training:

Logistic Regression

Random Forest Classifier (140 trees, max depth 30)

# Evaluation:

Classification Report

Confusion Matrix

Word Cloud for fake and real articles
