# 💬 Project: Sentiment Analysis using VADER

A Python-based **rule-driven Sentiment Analysis System** built with the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** model from NLTK. It classifies text into **Positive**, **Negative**, or **Neutral** sentiments, providing detailed intensity scores for each emotion.

This enhanced version includes **colored terminal output**, **interactive input options**, and **summary statistics** to give a complete, real-time understanding of text sentiment (e.g., healthcare feedback, social media, or reviews).

---

## 🧠 Overview & Analysis Flow

Sentiment analysis is a core task in **Natural Language Processing (NLP)**. This project uses the **VADER** model, a robust, lexicon- and rule-based approach specifically tuned for sentiment expressed in social media and informal text, handling elements like **emojis**, **slang**, and **capitalization for emphasis**.

### Simple VADER Analysis Flow
+----------------+      +----------------+      +------------------+
|   Input Text   |      | Tokenization & |      | Lexicon Matching |
| (Sentence/Line)|  --> | Preprocessing  |  --> |     (VADER)      |
+----------------+      +----------------+      +------------------+
|                                                 |
v                                                 v
+------------------+      +------------------+      +----------------+
| Compound Scoring |  --> | Classification   |  --> | Summary Output |
| (Polarity Score) |      | (P/N/Neu Label)  |      | (Counts & %)   |
+------------------+      +------------------+      +----------------+


---

## 🚀 Features

✅ **Rule-based sentiment analysis** (no model training required).
✅ Supports custom input or batch analysis via text files.
✅ **Color-coded terminal output** for easy visualization.
✅ Displays detailed intensity scores (`pos`, `neu`, `neg`, **`compound`**).
✅ Summary statistics of overall sentiment distribution.
✅ **Fully offline**, lightweight, and extremely fast.

---

## 🧩 Technologies Used

| Component | Description | Library |
| :--- | :--- | :--- |
| **NLTK** | Natural Language Toolkit, provides VADER sentiment analyzer | `nltk` |
| **VADER Lexicon** | Predefined dictionary of sentiment-weighted words | Built into NLTK |
| **Colorama** | Adds terminal colors for better visualization | `colorama` |
| **Python 3.x** | Core programming environment | - |

---

## ⚙️ How It Works (Classification Logic)

### Scoring
VADER calculates a **compound score**, which is a single, normalized measure of polarity in the range **$[-1, 1]$**. This score is adjusted based on detected linguistic rules (e.g., boosters, punctuation, negations).

### Classification
The final sentiment label is determined by applying a fixed threshold to the compound score:

$$\text{Sentiment Label} =
\begin{cases}
\text{Positive,} & \text{if compound} \ge 0.05 \\
\text{Negative,} & \text{if compound} \le -0.05 \\
\text{Neutral,} & \text{otherwise}
\end{cases}$$

| Score | Meaning |
| :--- | :--- |
| `pos` | Fraction of positive words |
| `neu` | Fraction of neutral words |
| `neg` | Fraction of negative words |
| `compound` | Overall normalized sentiment polarity |
| `label` | Final sentiment classification |

---

## 📂 Project Structure

.
├── sentiment_analysis_vader_extended.py  # Main script
├── reviews.txt                           # Optional input file (one sentence per line)
└── README.md                             # Documentation report


---

## 🧰 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/sentiment-analyzer.git](https://github.com/yourusername/sentiment-analyzer.git)
cd sentiment-analyzer
2. Install dependencies
Bash

pip install nltk colorama
3. Run the script
Bash

python sentiment_analysis_vader_extended.py
💻 Example Usage
Console Output: Interactive Mode
🧠 Sentiment Analysis using VADER
------------------------------------------------

Choose an option:
1. Analyze default sample texts
2. Enter your own sentences
3. Analyze texts from a file (one per line)

Enter your choice (1/2/3): 1
Sample Analysis Output:

Text: I love this healthcare app, it's so intuitive!
  -> Pos: 0.66 | Neu: 0.34 | Neg: 0.00 | Compound: 0.778
  => Final Sentiment: Positive

Text: The diagnosis system is terrible and confusing.
  -> Pos: 0.00 | Neu: 0.37 | Neg: 0.63 | Compound: -0.647
  => Final Sentiment: Negative

📊 Sentiment Summary:
  Positive: 4 (40.0%)
  Negative: 3 (30.0%)
  Neutral:  3 (30.0%)

✅ Analysis complete!