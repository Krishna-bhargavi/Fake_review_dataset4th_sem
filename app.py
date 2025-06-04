# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("🕵️‍♀️ Fake Review Detector")
st.write("Paste a review and I'll tell you if it's fake or real!")

user_input = st.text_area("Enter your review here:")

if st.button("Check if it's Fake or Real"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        review_vector = vectorizer.transform([user_input])
        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("✅ This review looks REAL!")
        else:
            st.error("❌ This review might be FAKE!")
