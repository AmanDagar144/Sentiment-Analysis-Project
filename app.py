import streamlit as st
import joblib
import re

# Load the saved model and vectorizer
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean input text (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Streamlit App
st.set_page_config(page_title="IMDb Sentiment Analyzer", layout="centered")
st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.write("Enter a movie review below and click **Predict** to see the sentiment.")

user_input = st.text_area("✏️ Your Review", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        sentiment = "😊 Positive" if prediction == 1 else "☹️ Negative"
        st.success(f"**Predicted Sentiment:** {sentiment}")
