"""
Mini NLP Project — Language Detection
File: mini-language-detector.py

Overview
--------
A compact, self-contained project to build a mini language-detection tool using character n-gram
features and a Multinomial Naive Bayes classifier (scikit-learn). It includes:
 - sample embedded training data (several languages) so you can run immediately
 - an option to train on a larger CSV dataset (if you provide one)
 - save/load model with joblib
 - a simple CLI to train and predict
 - an optional Streamlit UI for interactive testing

Requirements
------------
pip install scikit-learn pandas joblib streamlit

Usage examples
--------------
# Train using the embedded sample data and save model to model.joblib
python mini-language-detector.py --mode train --model-out model.joblib

# Predict language for a sentence
python mini-language-detector.py --mode predict --model model.joblib --text "This is a test"

# Run a small Streamlit UI (after saving model)
streamlit run mini-language-detector.py -- --mode serve --model model.joblib

Notes
-----
- Character n-grams are effective for language identification because they capture orthographic patterns.
- If you have a CSV dataset, prepare it with two columns: 'text' and 'label'. Use --data-file to pass path.

"""

from typing import List, Tuple
import argparse
import os
import sys
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ---------------------------
# Embedded sample dataset
# ---------------------------
# Small examples for quick experimentation (language codes: en, es, fr, de, hi, it, pt)
SAMPLE_DATA = [
    ("This is a test sentence in English.", "en"),
    ("How are you doing today?", "en"),
    ("I love programming and data science.", "en"),
    ("¿Cómo estás hoy?", "es"),
    ("Este es un texto en español.", "es"),
    ("Me encanta aprender nuevas cosas.", "es"),
    ("C'est une phrase en français.", "fr"),
    ("Bonjour, comment ça va?", "fr"),
    ("J'aime la programmation.", "fr"),
    ("Das ist ein deutscher Satz.", "de"),
    ("Wie geht es dir?", "de"),
    ("Ich lerne gerne.", "de"),
    ("यह एक हिंदी वाक्य है।", "hi"),
    ("आप कैसे हैं?", "hi"),
    ("मैं डेटा विज्ञान सीख रहा हूँ।", "hi"),
    ("Questo è un testo in italiano.", "it"),
    ("Mi piace programmare.", "it"),
    ("Este é um texto em português.", "pt"),
    ("Como você está hoje?", "pt"),
]


# ---------------------------
# Model utilities
# ---------------------------

def build_pipeline() -> Pipeline:
    """Return a scikit-learn Pipeline using character n-gram TF-IDF and MultinomialNB."""
    vect = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=True)
    clf = MultinomialNB()
    pipeline = Pipeline([("vect", vect), ("clf", clf)])
    return pipeline


def load_dataset_from_csv(path: str) -> Tuple[List[str], List[str]]:
    """Load a CSV with columns 'text' and 'label'."""
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    return texts, labels


def prepare_data(data_file: str = None) -> Tuple[List[str], List[str]]:
    """Return texts and labels. If data_file provided, load from CSV, otherwise use SAMPLE_DATA."""
    if data_file and os.path.exists(data_file):
        print(f"Loading dataset from {data_file}")
        return load_dataset_from_csv(data_file)
    else:
        texts, labels = zip(*SAMPLE_DATA)
        return list(texts), list(labels)


def train_and_save(model_out: str, data_file: str = None, test_size: float = 0.2, random_state: int = 42):
    texts, labels = prepare_data(data_file)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels if len(set(labels)) > 1 else None)

    pipeline = build_pipeline()
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Evaluation on held-out test set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(pipeline, model_out)
    print(f"Model saved to {model_out}")


def load_model(model_path: str) -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_texts(model_path: str, texts: List[str]):
    pipeline = load_model(model_path)
    preds = pipeline.predict(texts)
    probs = None
    # Some classifiers support predict_proba
    if hasattr(pipeline.named_steps['clf'], "predict_proba"):
        probs = pipeline.predict_proba(texts)
    results = []
    for i, t in enumerate(texts):
        entry = {"text": t, "pred": preds[i]}
        if probs is not None:
            # map class -> probability
            classes = pipeline.named_steps['clf'].classes_
            probs_map = {c: float(probs[i][j]) for j, c in enumerate(classes)}
            entry["probs"] = probs_map
        results.append(entry)
    return results


# ---------------------------
# Streamlit UI
# ---------------------------
def run_streamlit_app(model_path: str):
    try:
        import streamlit as st
    except Exception as e:
        print("Streamlit is required for the UI. Install with: pip install streamlit")
        raise

    pipeline = load_model(model_path)

    st.title("Mini Language Detector")
    st.write("Enter text and the model will predict the language.")

    text = st.text_area("Enter text", value="Type a sentence here to detect its language...", height=150)
    if st.button("Detect"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            pred = pipeline.predict([text])[0]
            st.success(f"Predicted language: {pred}")
            if hasattr(pipeline.named_steps['clf'], "predict_proba"):
                probs = pipeline.predict_proba([text])[0]
                classes = pipeline.named_steps['clf'].classes_
                prob_map = {c: float(p) for c, p in zip(classes, probs)}
                st.write("Probabilities:")
                st.json(prob_map)


# ---------------------------
# CLI
# ---------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Mini Language Detector")
    parser.add_argument("--mode", choices=["train", "predict", "serve"], required=True, help="Mode: train, predict, or serve (streamlit)")
    parser.add_argument("--data-file", type=str, help="Optional CSV file with 'text' and 'label' columns")
    parser.add_argument("--model-out", type=str, default="model.joblib", help="Where to save trained model")
    parser.add_argument("--model", type=str, help="Path to trained model for prediction/serve")
    parser.add_argument("--text", type=str, help="Text to predict (for predict mode)")

    args, unknown = parser.parse_known_args(argv)

    if args.mode == "train":
        train_and_save(args.model_out, data_file=args.data_file)

    elif args.mode == "predict":
        if not args.model:
            print("Please provide --model path for prediction")
            sys.exit(1)
        texts = [args.text] if args.text else []
        # if no text provided, read lines from stdin
        if not texts:
            print("Enter text to predict (end with Ctrl+D / Ctrl+Z):")
            input_text = sys.stdin.read().strip()
            if input_text:
                texts = [input_text]
        if not texts:
            print("No text to predict. Use --text or pipe input.")
            sys.exit(1)
        results = predict_texts(args.model, texts)
        for r in results:
            print("---")
            print(f"Text: {r['text']}")
            print(f"Predicted language: {r['pred']}")
            if "probs" in r:
                print("Probabilities:")
                for c, p in sorted(r["probs"].items(), key=lambda kv: -kv[1]):
                    print(f"  {c}: {p:.4f}")

    elif args.mode == "serve":
        if not args.model:
            print("Please provide --model path to serve (model must be trained first). Example: streamlit run mini-language-detector.py -- --mode serve --model model.joblib")
            sys.exit(1)
        # When launched via `streamlit run`, Streamlit runs the file and we intercept --mode serve to start UI
        run_streamlit_app(args.model)


if __name__ == "__main__":
    main()
