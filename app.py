# ---------------------------------------
# Velora AI — FINAL CLEAN VERSION
# ---------------------------------------

import streamlit as st
import joblib
import re
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Velora AI", layout="wide")

# -----------------------------
# PREMIUM UI STYLE
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
textarea, input {
    background-color: #111827 !important;
    color: white !important;
}
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

# -----------------------------
# RULE-BASED FIX (IMPORTANT)
# -----------------------------
NEGATIVE_PHRASES = [
    "very bad", "worst", "terrible", "awful",
    "not good", "hate", "poor quality", "bad product"
]

def rule_override(text, prediction):
    text_lower = text.lower()
    for phrase in NEGATIVE_PHRASES:
        if phrase in text_lower:
            return 0
    return prediction

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load('best_random_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# ANALYSIS FUNCTION
# -----------------------------
def analyze_text(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    prediction = rule_override(text, prediction)

    proba = model.predict_proba(vectorized)[0]
    confidence = round(max(proba) * 100)

    # cap confidence
    if confidence > 98:
        confidence = 98

    result = "Positive" if prediction == 1 else "Negative"

    # confidence level
    if confidence >= 90:
        level = "High"
    elif confidence >= 70:
        level = "Moderate"
    else:
        level = "Low"

    # important words
    feature_names = vectorizer.get_feature_names_out()
    weights = vectorized.toarray()[0]

    important_words = []
    for i in weights.argsort()[::-1]:
        if weights[i] > 0:
            word = feature_names[i]
            if word not in important_words:
                important_words.append(word)
        if len(important_words) == 5:
            break

    return result, confidence, level, important_words

# -----------------------------
# UI
# -----------------------------
st.title("Velora AI")
st.caption("Intelligent Text Analysis System")

# -----------------------------
# SINGLE TEXT
# -----------------------------
st.subheader("🔍 Analyze Single Text")

user_input = st.text_area(
    "Enter text:",
    placeholder="Type a review or comment..."
)

if st.button("Analyze", disabled=(user_input.strip() == "")):
    with st.spinner("Analyzing..."):

        result, confidence, level, words = analyze_text(user_input)

        if result == "Positive":
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")

        st.progress(confidence)
        st.write(f"Confidence: ~{confidence}% ({level})")

        st.write("Why this prediction:")
        st.info(", ".join(words))

        # highlight
        highlighted = user_input
        for w in words:
            if result == "Positive":
                highlighted = highlighted.replace(
                    w,
                    f"<span style='color:#22c55e;font-weight:bold'>{w}</span>"
                )
            else:
                highlighted = highlighted.replace(
                    w,
                    f"<span style='color:#ef4444;font-weight:bold'>{w}</span>"
                )

        st.markdown(highlighted, unsafe_allow_html=True)

# -----------------------------
# CSV UPLOAD
# -----------------------------
st.subheader("📂 Batch Analysis (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload CSV (must contain 'text' column)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column")
    else:
        results = []
        confidences = []

        with st.spinner("Processing file..."):
            for text in df['text']:
                result, confidence, _, _ = analyze_text(str(text))
                results.append(result)
                confidences.append(confidence)

        df['Prediction'] = results
        df['Confidence'] = confidences

        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results",
            csv,
            "velora_results.csv",
            "text/csv"
        )