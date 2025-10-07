"""
sentiment_analysis_vader_extended.py
------------------------------------

An enhanced rule-based sentiment analysis tool using VADER 
(Valence Aware Dictionary and sEntiment Reasoner) from NLTK.

Features:
- Analyzes multiple text samples or custom user input
- Calculates detailed sentiment scores (positive, negative, neutral, compound)
- Displays color-coded terminal results for readability
- Provides summary statistics on sentiment distribution
- Optionally reads text from a file (one sentence per line)

Dependencies:
    pip install nltk colorama

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from colorama import Fore, Style, init

# Initialize colorama (for Windows terminal compatibility)
init(autoreset=True)

# Download the VADER lexicon (only once)
nltk.download('vader_lexicon', quiet=True)

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Example corpus
default_texts = [
    "I love this healthcare app, it's so intuitive!",
    "The diagnosis system is terrible and confusing.",
    "The new update is okay, nothing special but works fine.",
    "Amazing service by the doctors and staff.",
    "This machine learning model failed completely!",
    "The appointment booking process was smooth and quick.",
    "Customer support could be more responsive.",
    "Overall, I am satisfied with the service.",
    "Not bad, but definitely needs improvement.",
    "The interface is outdated and frustrating to use."
]


def analyze_text(text):
    """
    Analyze sentiment for a single text string.
    Returns a dictionary with detailed scores and final sentiment label.
    """
    score = analyzer.polarity_scores(text)
    compound = score['compound']

    # Classify compound score into sentiment categories
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "text": text,
        "positive": score['pos'],
        "neutral": score['neu'],
        "negative": score['neg'],
        "compound": compound,
        "sentiment": label
    }


def print_result(result):
    """
    Pretty-print the result with colors for clarity.
    """
    color = {
        "Positive": Fore.GREEN,
        "Negative": Fore.RED,
        "Neutral": Fore.YELLOW
    }.get(result["sentiment"], Fore.WHITE)

    print(f"{color}Text: {result['text']}")
    print(f"{color} -> Pos: {result['positive']:.2f} | Neu: {result['neutral']:.2f} "
          f"| Neg: {result['negative']:.2f} | Compound: {result['compound']:.3f}")
    print(f"{color} => Final Sentiment: {Style.BRIGHT}{result['sentiment']}\n")


def summarize(results):
    """
    Print summary statistics showing count of each sentiment.
    """
    total = len(results)
    positives = sum(1 for r in results if r["sentiment"] == "Positive")
    negatives = sum(1 for r in results if r["sentiment"] == "Negative")
    neutrals = total - positives - negatives

    print(Style.BRIGHT + "\n📊 Sentiment Summary:")
    print(Fore.GREEN + f"  Positive: {positives} ({positives / total * 100:.1f}%)")
    print(Fore.RED + f"  Negative: {negatives} ({negatives / total * 100:.1f}%)")
    print(Fore.YELLOW + f"  Neutral:  {neutrals} ({neutrals / total * 100:.1f}%)")
    print(Style.RESET_ALL)


def load_from_file(file_path):
    """
    Load text samples from a text file (one sentence per line).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except FileNotFoundError:
        print(Fore.RED + f"File not found: {file_path}")
        return []


def main():
    print(Style.BRIGHT + "\n🧠 Sentiment Analysis using VADER")
    print("------------------------------------------------\n")
    print("Choose an option:")
    print("1. Analyze default sample texts")
    print("2. Enter your own sentences")
    print("3. Analyze texts from a file (one per line)")
    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == "1":
        texts = default_texts
    elif choice == "2":
        print("\nEnter sentences (type 'done' to finish):")
        texts = []
        while True:
            line = input("> ")
            if line.lower() == "done":
                break
            texts.append(line)
    elif choice == "3":
        file_path = input("\nEnter file path: ").strip()
        texts = load_from_file(file_path)
        if not texts:
            print(Fore.RED + "No texts loaded. Exiting.")
            return
    else:
        print(Fore.RED + "Invalid choice. Exiting.")
        return

    print("\nAnalyzing sentiments...\n")
    results = [analyze_text(text) for text in texts]

    for res in results:
        print_result(res)

    summarize(results)
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
