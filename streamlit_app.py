import io
import os
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import matplotlib.pyplot as plt

st.set_page_config(page_title="Spam Detector (Naive Bayes)", page_icon="📩", layout="centered")

st.title("📩 Spam Detection — Naive Bayes")
st.write("A simple Streamlit app based on a Multinomial Naive Bayes model with CountVectorizer, following your notebook setup.")

# --- Session state ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "label_map" not in st.session_state:
    st.session_state.label_map = None

with st.sidebar:
    st.header("Model")
    choice = st.radio("Choose action", ["Train New Model", "Load Saved Model (.pkl)"])

    if choice == "Load Saved Model (.pkl)":
        uploaded_model = st.file_uploader("Upload pipeline .pkl", type=["pkl"])
        if uploaded_model is not None:
            try:
                st.session_state.pipeline = pickle.load(uploaded_model)
                st.success("Loaded saved model successfully.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if choice == "Train New Model":
        st.caption("Upload a CSV with columns **Category** (labels) and **Message** (text).")
        data_file = st.file_uploader("Upload dataset (.csv)", type=["csv"], key="dataset")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", min_value=0, value=5, step=1)

        if st.button("Train"):
            if data_file is None:
                st.warning("Please upload a CSV file with columns: Category, Message.")
            else:
                try:
                    df = pd.read_csv(data_file)
                    # basic validation
                    if not {"Category", "Message"}.issubset(df.columns):
                        raise ValueError("CSV must contain columns: 'Category' and 'Message'.")

                    # Optionally normalize column names and strip NA
                    df = df[["Category", "Message"]].dropna().reset_index(drop=True)
                    X = df["Message"]
                    y = df["Category"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )

                    # Pipeline mirrors notebook: CountVectorizer + MultinomialNB
                    pipe = Pipeline([
                        ("vectorizer", CountVectorizer()),
                        ("nb", MultinomialNB())
                    ])

                    start = time.time()
                    pipe.fit(X_train, y_train)
                    train_time = time.time() - start

                    # Evaluate
                    y_pred = pipe.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
                    report = classification_report(y_test, y_pred, zero_division=0)

                    st.session_state.pipeline = pipe

                    st.subheader("Validation Metrics")
                    st.metric("Accuracy", f"{acc:.3f}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision (weighted)", f"{pr:.3f}")
                    with col2:
                        st.metric("Recall (weighted)", f"{rc:.3f}")
                    with col3:
                        st.metric("F1-score (weighted)", f"{f1:.3f}")

                    st.caption(f"Training time: {train_time:.2f}s")

                    st.text("Classification Report")
                    st.code(report, language="text")

                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
                    fig = plt.figure()
                    plt.imshow(cm, interpolation="nearest")
                    plt.title("Confusion Matrix")
                    plt.colorbar()
                    tick_marks = np.arange(len(sorted(y.unique())))
                    plt.xticks(tick_marks, sorted(y.unique()), rotation=45)
                    plt.yticks(tick_marks, sorted(y.unique()))
                    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, format(cm[i, j], "d"),
                                     horizontalalignment="center",
                                     color="white" if cm[i, j] > thresh else "black")
                    plt.tight_layout()
                    plt.ylabel("True label")
                    plt.xlabel("Predicted label")
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Training failed: {e}")

    if st.session_state.pipeline is not None:
        st.divider()
        st.subheader("Save Current Model")
        file_name = st.text_input("File name", value="spam_nb_pipeline.pkl")
        if st.button("Save as .pkl"):
            try:
                bytes_io = io.BytesIO()
                pickle.dump(st.session_state.pipeline, bytes_io)
                bytes_io.seek(0)
                st.download_button("Download .pkl", data=bytes_io.read(), file_name=file_name, mime="application/octet-stream")
            except Exception as e:
                st.error(f"Failed to export model: {e}")

st.divider()
st.header("Try It")
if st.session_state.pipeline is None:
    st.info("Load or train a model in the sidebar first.")
else:
    sample_text = st.text_area("Enter an SMS/email message to classify", height=120, placeholder="Type a message here...")
    if st.button("Predict"):
        if not sample_text.strip():
            st.warning("Please enter a message.")
        else:
            try:
                pred = st.session_state.pipeline.predict([sample_text])[0]
                proba = None
                # try to get probabilities if available
                try:
                    proba = st.session_state.pipeline.predict_proba([sample_text])[0]
                    # build a dataframe of class probabilities
                    classes = list(st.session_state.pipeline.classes_)
                    prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
                except Exception:
                    prob_df = None

                st.success(f"Prediction: **{pred}**")
                if prob_df is not None:
                    st.write("Class probabilities:")
                    st.dataframe(prob_df.sort_values("Probability", ascending=False).reset_index(drop=True))

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.caption("Tip: This app mirrors your notebook's approach: CountVectorizer feeding into MultinomialNB, with a simple train/test split.")
