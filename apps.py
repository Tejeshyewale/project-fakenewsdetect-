import streamlit as st
import joblib
from PIL import Image

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #1e1e1e;
            color: #f1f1f1;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h2, h3 {
            color: #f45b69;
        }

        .stTextArea, .stTextInput > div > div > input {
            background-color: #2e2e2e;
            color: white;
        }

        .stButton > button {
            background-color: #f45b69;
            color: white;
            padding: 10px 20px;
            border-radius: 12px;
            font-weight: bold;
            border: none;
        }

        .stButton > button:hover {
            background-color: #d73748;
        }

        .result-box {
            background-color: #2e2e2e;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 style='text-align: center;'>🧠 Fake News Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Paste your news content below and let AI tell you if it's real or fake.</p>", unsafe_allow_html=True)

# Text Input
user_input = st.text_area("📝 Enter News Article:", height=250, placeholder="Paste or type the news content here...")

# Predict Button
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        vectorized_input = tfidf.transform([user_input])
        prediction = model.predict(vectorized_input)
        label = "Real" if prediction[0] == 1 else "Fake"

        if label == "Real":
            st.markdown(
                "<div class='result-box'><h2 style='color:#27ae60;'>✅ This news is REAL.</h2></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box'><h2 style='color:#e74c3c;'>🚨 This news is FAKE.</h2></div>",
                unsafe_allow_html=True
            )

# Footer
st.markdown("""
    <hr style='border: 1px solid #444; margin-top: 30px;'>
    <p style='text-align: center; font-size: 14px;'>Made with ❤️ by <strong>You</strong> | Powered by Streamlit & Machine Learning</p>
""", unsafe_allow_html=True)
