"""Customer Review Sentiment Analysis using TF-IDF + Logistic Regression."""

import os
import re
from collections import Counter
import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

STOP_WORDS = set(
    "the a an and or is it to of for this that was very i my in on with be but not so "
    "are from by at as have has had really quite too over after during into في من على إلى عن مع هذا هذه جدا".split()
)


class _FallbackSentimentPipeline:
    """Lexicon + Naive Bayes fallback used only if scikit-learn is unavailable."""

    def __init__(self):
        self.classes_ = np.array(["negative", "neutral", "positive"])
        self.word_probs = {}

    def _tokenize(self, text: str):
        return re.findall(r"[\w\u0600-\u06FF]+", str(text).lower())

    def fit(self, X, y):
        counts = {c: Counter() for c in self.classes_}
        totals = {c: 1 for c in self.classes_}
        for txt, lab in zip(X, y):
            if lab in counts:
                toks = self._tokenize(txt)
                counts[lab].update(toks)
                totals[lab] += len(toks)
        vocab = set().union(*[c.keys() for c in counts.values()])
        V = max(1, len(vocab))
        self.word_probs = {
            c: {w: (counts[c][w] + 1) / (totals[c] + V) for w in vocab}
            for c in self.classes_
        }
        self.default_prob = {c: 1.0 / (totals[c] + V) for c in self.classes_}
        return self

    def predict_proba(self, X):
        out = []
        for txt in X:
            toks = self._tokenize(txt)
            scores = []
            for c in self.classes_:
                log_p = sum(np.log(self.word_probs[c].get(w, self.default_prob[c])) for w in toks)
                scores.append(log_p)
            scores = np.array(scores)
            scores -= scores.max()
            probs = np.exp(scores)
            probs /= probs.sum()
            out.append(probs)
        return np.array(out)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def train_sentiment(df: pd.DataFrame, save_model: bool = True):
    """Train TF-IDF + Logistic Regression sentiment classifier on customer reviews."""
    labels = sorted(df["sentiment"].unique())

    if HAS_SKLEARN:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df["review"],
            df["sentiment"],
            test_size=0.2,
            random_state=42,
            stratify=df["sentiment"],
        )
        pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        token_pattern=r"(?u)\b\w\w+\b",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        C=1.5,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        metrics = {
            "accuracy": round(float(accuracy_score(y_te, preds)), 4),
            "f1_macro": round(float(f1_score(y_te, preds, average="macro")), 4),
            "report": classification_report(y_te, preds, output_dict=True),
            "confusion": confusion_matrix(y_te, preds, labels=labels).tolist(),
            "labels": labels,
        }
        if save_model:
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(pipe, os.path.join(MODELS_DIR, "sentiment.joblib"))
    else:
        n_test = max(1, int(0.2 * len(df)))
        X_tr, X_te = df["review"].iloc[:-n_test], df["review"].iloc[-n_test:]
        y_tr, y_te = df["sentiment"].iloc[:-n_test], df["sentiment"].iloc[-n_test:]
        pipe = _FallbackSentimentPipeline().fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        acc = float(np.mean(preds == y_te.to_numpy()))
        cm = [[int(np.sum((y_te == r) & (preds == c))) for c in labels] for r in labels]
        metrics = {
            "accuracy": round(acc, 4),
            "f1_macro": round(acc, 4),
            "report": {},
            "confusion": cm,
            "labels": labels,
        }

    return pipe, metrics


def predict_sentiment(model, text: str) -> dict:
    """Predict sentiment label and class probabilities for a single review."""
    label = str(model.predict([text])[0])
    probs_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = list(getattr(model, "classes_", ["negative", "neutral", "positive"]))
        probs_dict = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, probs)}
    return {
        "text": text,
        "sentiment": label,
        "confidence": probs_dict.get(label, 100.0),
        "probabilities": probs_dict,
    }


def top_complaints(df: pd.DataFrame, n: int = 8):
    """Extract distinct complaint phrases from negative reviews (one per complaint topic)."""
    neg = df[df["sentiment"] == "negative"]["review"].astype(str).str.lower().tolist()
    words = Counter()
    phrase_docs = {}
    filler = {"ten", "days", "without", "notice", "one", "week", "two", "three", "due", "box", "item", "feel", "feels", "took"}
    for doc_id, text in enumerate(neg):
        toks = [
            w
            for w in re.findall(r"[a-z\u0600-\u06FF']+", text)
            if w not in STOP_WORDS and w not in filler and len(w) > 2
        ]
        for pair in set(zip(toks, toks[1:])):
            phrase = " ".join(pair)
            words[phrase] += 1
            phrase_docs.setdefault(phrase, set()).add(doc_id)

    selected = []
    used_tokens = set()
    for phrase, count in words.most_common(n * 10):
        ptoks = set(phrase.split())
        if ptoks & used_tokens:
            continue
        docs = phrase_docs[phrase]
        # Skip if >50% of its occurrences are in the exact same reviews as an already selected phrase
        if any(len(docs & phrase_docs[s_p]) / max(1, len(docs)) > 0.55 for s_p, _ in selected):
            continue
        selected.append((phrase, count))
        used_tokens.update(ptoks)
        if len(selected) >= n:
            break
    return selected
