# Sentiment Analysis of Tweets
**Comparing three classifiers on three-class tweet sentiment.**

Capstone project of the İTÜ Data Science Certificate Program (2023), revised in 2026. The script classifies tweets as negative, neutral or positive, and compares three classifiers trained on the same TF-IDF features: Linear SVM, Multinomial Naive Bayes and Logistic Regression.

Built with Python · pandas · NLTK · scikit-learn · matplotlib · seaborn

---

## What it does

1. **Explores the data:** plots how the three labels are distributed and how long the tweets are.
2. **Preprocesses the text:** lowercasing, punctuation removal, tokenisation with NLTK, and removal of English stopwords, except the negations *not*, *no* and *nor*, which matter for sentiment.
3. **Splits before vectorising:** 80% of the tweets are used for training and 20% for testing. The TF-IDF vectoriser (5,000 features) is fitted on the training set only, so no information from the test set leaks into training.
4. **Trains and evaluates three classifiers:** for each, it reports accuracy and per-class precision, recall and F1, and draws one-vs-rest ROC curves with the AUC of each class.
5. **Predicts two example sentences** with all three models.

---

## Results

On the test set (32,594 tweets):

| Model | Accuracy | Macro F1 | AUC (negative / neutral / positive) |
|---|---|---|---|
| Linear SVM | 0.90 | 0.89 | 0.95 / 0.97 / 0.95 |
| Logistic Regression | 0.89 | 0.88 | 0.95 / 0.97 / 0.95 |
| Multinomial Naive Bayes | 0.73 | 0.69 | 0.90 / 0.94 / 0.91 |

- Linear SVM and Logistic Regression perform almost the same.
- Multinomial Naive Bayes misses more than half of the negative tweets (recall 0.43) and labels too many tweets as positive (precision 0.65 on the positive class).
- Neutral is the easiest class for all three models (recall 0.97 for Linear SVM and Logistic Regression).

---

## Dataset

`Twitter_Data.csv`, a public Kaggle dataset of about 163,000 tweets about Indian politics, each labelled −1 (negative), 0 (neutral) or 1 (positive). The dataset is not included in this repository.

- The script expects the columns to be named `text` and `label`. In the original download they are called `clean_text` and `category`, so rename them first.
- The dataset does not document how the labels were assigned. The models learn to reproduce these labels, whatever produced them.

---

## Installation

```
pip install pandas matplotlib seaborn nltk scikit-learn
```

Download NLTK's stopword list and tokeniser once:

```
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

---

## How to use

Put `Twitter_Data.csv` in the same folder as `sentiment_analysis.py`, open a terminal in that folder and run:

```
python sentiment_analysis.py
```

The results are printed in the terminal. Each plot opens in its own window, and the script continues once you close it.

---

## Revision history

**2023:** submitted as the capstone project of the İTÜ Data Science Certificate Program and accepted for the Certificate of Expertise.

**2026 revision:**
- **Negations are kept.** NLTK's English stopword list includes *not*, *no* and *nor*, so removing stopwords turned "not good" into "good". These three words are now excluded from the list.
- **ROC curves are corrected.** Each curve now compares the test labels with the classifier's own class labels (−1, 0, 1). Before, the code compared the labels with column numbers (0, 1, 2), so the curves did not match the classes they were named after.
- **A `NameError` is fixed.** The SVM evaluation referred to a variable that did not exist, which stopped the script.
- **Warnings are no longer hidden.** This showed that Logistic Regression did not converge within its default 100 iterations; it now trains for up to 1,000.
- **The dataset is loaded from the folder the script is run in**, not from a fixed path on one computer.

---

## License

CC BY-NC 4.0. See `LICENSE`.

---

## Built by

Ece Erkmen (github.com/ecerkmen)

Claude (Anthropic): code review for the 2026 revision, and this README. Not affiliated with or endorsed by Anthropic.
