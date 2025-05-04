# 📊 Sentiment Analysis Using Machine Learning

This project implements a **Sentiment Analysis** system using **Machine Learning** to classify IMDB movie reviews as either *positive* or *negative*. It uses **TF-IDF** for feature extraction and a **Logistic Regression** classifier for prediction. A simple **Streamlit** UI allows for real-time sentiment prediction on user input.

---

## 📌 Features

* Data preprocessing (cleaning, tokenization, stopword removal)
* Feature extraction using TF-IDF
* Logistic Regression classifier (trained and tested)
* Real-time sentiment prediction using Streamlit
* Displays prediction result: ✅ Positive or ❌ Negative

---

## 🖼 Sample Output

| Input Review                                 | Prediction  |
|---------------------------------------------|-------------|
| *"This movie was truly inspiring and moving."* | ✅ Positive  |
| *"Boring plot, weak characters. Waste of time."* | ❌ Negative  |
| ![sample](images/image_01.png) | ✅ Genuine  |
| ![sample](images/image_02.png)  | ❌ Forged   |

> Add screenshots like accuracy charts or confusion matrix once uploaded:


---

## 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn
* Joblib

---

## 🗂️ Project Structure

sentiment-analysis/
│
├── IMDB-Dataset.csv                # Raw dataset of movie reviews
│
├── sentiment_analysis.ipynb        # Jupyter notebook for EDA, training, and evaluation
│
├── app.py                          # Streamlit app for user review input and prediction
│
├── logistic_regression_model.pkl   # Trained Logistic Regression model saved using joblib
│
└── tfidf_vectorizer.pkl            # Serialized TF-IDF vectorizer for transforming input data

---

## 📥 Dataset Link

Download the dataset from:  
🔗 [https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)

---




