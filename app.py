# ---------------------------------------
# Velora AI - FINAL POLISHED VERSION
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
# STYLING
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
textarea {
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
# CLEAN FUNCTION
# -----------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

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
# TITLE
# -----------------------------
st.title("Velora AI")
st.caption("Intelligent Text Analysis System")

# -----------------------------
# ANALYSIS FUNCTION
# -----------------------------
def analyze_text(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]

    confidence = float(max(proba) * 100)
    result = "Positive" if prediction == 1 else "Negative"

    # --- Feature importance ---
    feature_names = vectorizer.get_feature_names_out()
    weights = vectorized.toarray()[0]

    # Get top features
    top_indices = weights.argsort()[::-1]

    keywords = []
    used_words = set()

    for i in top_indices:
        if weights[i] == 0:
            continue
        word = feature_names[i]

        # Deduplicate overlapping tokens
        base_word = word.split()[0]
        if base_word not in used_words:
            keywords.append(word)
            used_words.add(base_word)

        if len(keywords) == 5:
            break

    return result, confidence, keywords

# -----------------------------
# CONFIDENCE LABEL
# -----------------------------
def confidence_label(score):
    if score >= 90:
        return "High"
    elif score >= 70:
        return "Moderate"
    else:
        return "Low"

# -----------------------------
# HIGHLIGHT FUNCTION
# -----------------------------
def highlight_text(text, keywords, sentiment):
    color = "#22c55e" if sentiment == "Positive" else "#ef4444"

    for word in keywords:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(
            f"<span style='color:{color}; font-weight:bold;'>{word}</span>",
            text
        )

    return text

# -----------------------------
# SINGLE INPUT UI
# -----------------------------
st.subheader("🔍 Analyze Single Text")

user_input = st.text_area(
    "Enter text:",
    placeholder="Type a review or comment..."
)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        with st.spinner("Analyzing..."):
            result, confidence, keywords = analyze_text(user_input)

        # --- Display ---
        st.success(f"Prediction: {result}")

        st.progress(int(confidence))
        st.write(f"Confidence: ~{int(confidence)}% ({confidence_label(confidence)})")

        # Explanation
        st.write("Why this prediction:")
        st.info(", ".join(keywords))

        # Highlighted text
        highlighted = highlight_text(user_input, keywords, result)
        st.markdown(highlighted, unsafe_allow_html=True)

# -----------------------------
# CSV BATCH ANALYSIS
# -----------------------------
st.subheader("📂 Batch Analysis (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload CSV (must contain 'text' column)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'text' not in df.columns:
        st.error("CSV must contain a column named 'text'")
    else:
        results = []
        confidences = []

        with st.spinner("Processing file..."):
            for text in df['text']:
                result, confidence, _ = analyze_text(str(text))
                results.append(result)
                confidences.append(round(confidence, 2))

        df['Prediction'] = results
        df['Confidence'] = confidences

        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "results.csv", "text/csv")