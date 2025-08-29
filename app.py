from flask import Flask, render_template, request, redirect, flash
import joblib
import os
import numpy as np
import re
import string
import nltk

# Download stopwords (first time only)
nltk.download('stopwords')
from nltk.corpus import stopwords

# Flask App Setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # needed for flash messages

# -------------------- Load Model & Vectorizer --------------------
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    model = None
    vectorizer = None
    print("❌ Error loading model or vectorizer:", e)

# -------------------- Text Cleaning Function --------------------
def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
        return text
    except Exception as e:
        print("❌ Error in cleaning text:", e)
        return ""

# -------------------- Routes --------------------

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Check News Page (Form)
@app.route('/check', methods=['GET', 'POST'])
def check():
    return render_template('check.html')

# Result Page (POST)
@app.route('/result', methods=['POST'])
def result():
    # Model check
    if model is None or vectorizer is None:
        flash("Model not loaded. Please try again later.")
        return redirect('/check')

    # Get form inputs
    title = request.form.get("title", "").strip()
    article = request.form.get("article", "").strip()

    # Empty check
    if not title and not article:
        flash("Please enter at least a title or article content.")
        return redirect('/check')

    try:
        # Combine and clean
        full_text = title + " " + article
        cleaned_text = clean_text(full_text)

        # After cleaning, check length
        if len(cleaned_text.split()) < 3:
            flash("Please enter more meaningful text. Your input is too short.")
            return redirect('/check')

        # Transform to vector
        vect_text = vectorizer.transform([cleaned_text])

        # Check if all-zero vector (words not in training vocab)
        if vect_text.nnz == 0:
            flash("Input text does not contain meaningful words for analysis.")
            return redirect('/check')

        # Predict
        prediction = model.predict(vect_text)[0]
        prob = model.predict_proba(vect_text)[0]
        label = "REAL" if prediction == 1 else "FAKE"
        confidence = round(np.max(prob) * 100, 2)

        # Dynamic reason
        reason = f"Based on language patterns, the model is {confidence}% confident this news is {label} using TF-IDF features and Logistic Regression."

        # Show result
        return render_template("result.html", label=label, confidence=confidence, reason=reason)

    except Exception as e:
        print("❌ Unexpected error during prediction:", e)
        flash("Something went wrong while analyzing the news. Please try again.")
        return redirect('/check')

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
