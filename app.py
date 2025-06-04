{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cbfeef67",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'streamlit'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# app.py\u001b[39;00m\n\u001b[0;32m----> 2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mstreamlit\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mst\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mpandas\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mpd\u001b[39;00m\n\u001b[1;32m      4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mjoblib\u001b[39;00m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'"
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# Load the trained model and vectorizer\n",
    "model = joblib.load(\"model.pkl\")\n",
    "vectorizer = joblib.load(\"vectorizer.pkl\")\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"🕵️‍♀️ Fake Review Detector\")\n",
    "st.write(\"Paste a review and I'll tell you if it's fake or real!\")\n",
    "\n",
    "# Text input box\n",
    "user_input = st.text_area(\"Enter your review here:\")\n",
    "\n",
    "# Button\n",
    "if st.button(\"Check if it's Fake or Real\"):\n",
    "    if user_input.strip() == \"\":\n",
    "        st.warning(\"Please enter some text!\")\n",
    "    else:\n",
    "        # Transform and predict\n",
    "        review_vector = vectorizer.transform([user_input])\n",
    "        prediction = model.predict(review_vector)\n",
    "\n",
    "        if prediction[0] == 1:\n",
    "            st.success(\"✅ This review looks REAL!\")\n",
    "        else:\n",
    "            st.error(\"❌ This review might be FAKE!\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
