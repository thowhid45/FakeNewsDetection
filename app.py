# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Page Config ---
st.set_page_config(page_title="Fake News Detection", layout="wide")

# --- Header ---
st.title("📰 Fake News Detection System")
st.markdown("Detect whether a news article is **Fake** or **Genuine** using Machine Learning.")

# --- Load Data ---
@st.cache_data
def load_data():
    true = pd.read_csv("True.csv")
    fake = pd.read_csv("Fake.csv")
    true['label'] = 1
    fake['label'] = 0
    news = pd.concat([true, fake], axis=0)
    news.drop(['title', 'subject', 'date'], axis=1, inplace=True)
    news = news.sample(frac=1).reset_index(drop=True)
    return news

# --- Preprocessing ---
def wordsopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

# --- Train Models ---
@st.cache_resource
def train_models(news):
    news['text'] = news['text'].apply(wordsopt)
    x = news['text']
    y = news['label']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

    vect = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5, ngram_range=(1, 2))
    xvt = vect.fit_transform(xtrain)
    xvtest = vect.transform(xtest)

    logistic = LogisticRegression()
    logistic.fit(xvt, ytrain)
    rf = RandomForestClassifier(n_estimators=140, max_depth=30, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(xvt, ytrain)

    return logistic, rf, vect, xvt, ytrain, xvtest, ytest

# --- WordCloud Plot ---
def plot_wordcloud(text, title, bg='white'):
    wc = WordCloud(width=800, height=400, background_color=bg).generate(text)
    plt.figure(figsize=(10, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=18)
    st.pyplot(plt)

# --- Manual Testing ---
def predict_news(text, vect, logistic, rf):
    df = pd.DataFrame({'text': [text]})
    df['text'] = df['text'].apply(wordsopt)
    tfidf = vect.transform(df['text'])
    pred1 = logistic.predict(tfidf)[0]
    pred2 = rf.predict(tfidf)[0]
    return pred1, pred2

# --- Load and Train ---
news = load_data()
logistic, rf, vect, xvt, ytrain, xvtest, ytest = train_models(news)

# --- Tabs for UI ---
tab1, tab2, tab3 = st.tabs(["🔍 Detect News", "📊 Visualize", "📈 Model Evaluation"])

# --- Tab 1: Manual Detection ---
with tab1:
    st.header("Enter News Article")
    user_input = st.text_area("Paste the news content below:", height=250)
    if st.button("Classify"):
        if user_input.strip() == "":
            st.warning("Please enter some news content.")
        else:
            pred1, pred2 = predict_news(user_input, vect, logistic, rf)
            st.markdown("### Results:")
            st.info(f"**Logistic Regression:** {'Genuine' if pred1 else 'Fake'}")
            st.info(f"**Random Forest:** {'Genuine' if pred2 else 'Fake'}")

# --- Tab 2: WordCloud Visualization ---
with tab2:
    st.header("WordCloud Comparison")
    col1, col2 = st.columns(2)
    with col1:
        fake_text = " ".join(news[news['label'] == 0]['text'])
        st.subheader("Fake News")
        plot_wordcloud(fake_text, "Fake News", bg='black')

    with col2:
        true_text = " ".join(news[news['label'] == 1]['text'])
        st.subheader("True News")
        plot_wordcloud(true_text, "Genuine News")

# --- Tab 3: Evaluation ---
with tab3:
    st.header("Model Evaluation Metrics")
    pred_lr = logistic.predict(xvtest)
    pred_rf = rf.predict(xvtest)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Logistic Regression")
        st.text(classification_report(ytest, pred_lr))
        cm_lr = confusion_matrix(ytest, pred_lr)
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
        st.pyplot(plt)

    with col2:
        st.subheader("Random Forest")
        st.text(classification_report(ytest, pred_rf))
        cm_rf = confusion_matrix(ytest, pred_rf)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
        st.pyplot(plt)
