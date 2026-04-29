# Velora AI

Velora AI is a lightweight intelligent text analysis system built with machine learning and deployed using Streamlit.

## Features

* Sentiment prediction (Positive / Negative)
* Confidence score visualization
* Explanation of predictions (important keywords)
* Keyword highlighting
* Batch analysis via CSV upload
* Downloadable results

## Tech Stack

* Python
* Scikit-learn
* Streamlit
* Pandas
* Joblib

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
Velora-AI/
│── app.py
│── best_random_model.pkl
│── vectorizer.pkl
│── requirements.txt
│── README.md
```

## Deployment

This app can be deployed easily using Streamlit Cloud.

---

Built as part of a machine learning project.
