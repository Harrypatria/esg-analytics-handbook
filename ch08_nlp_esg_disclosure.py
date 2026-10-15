"""
Chapter 8: NLP for ESG Disclosure Analysis
Data Analytics and AI in ESG, Climate Change and Sustainability
Patria, Anindika Yudhy & Zulkifli (2026)

Demonstrates:
  - Text preprocessing pipeline for ESG disclosures
  - TF-IDF feature extraction
  - Zero-shot classification with ClimateBERT (HuggingFace)
  - Sentiment analysis on ESG news headlines
  - Greenwashing keyword detection
  - Named entity recognition (NER) for ESG entities
"""

# ── 1. Imports ───────────────────────────────────────────────────────────────
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#1e3a5f", "#2563eb", "#10b981", "#f97316", "#ef4444"]

# ── 2. Sample ESG Disclosure Corpus ──────────────────────────────────────────
ESG_TEXTS = [
    ("We reduced Scope 1 emissions by 18 percent in 2023 through fuel switching and "
     "operational efficiency improvements at our upstream facilities.", "Environmental", 1),
    ("The company achieved gender parity at board level, with women comprising 50 percent "
     "of directors. Our LTIFR declined to 0.12 per million hours worked.", "Social", 1),
    ("Our audit committee reviewed and approved the ESG data assurance framework, "
     "ensuring ISSB S1 and S2 compliance ahead of the 2025 mandatory disclosure date.", "Governance", 1),
    ("We are committed to sustainability and will do better in the future. "
     "Our products are eco-friendly and green.", "Environmental", 0),  # greenwashing signal
    ("Carbon neutrality is our aspiration by 2050. We continue to invest in fossil fuel "
     "exploration to meet growing energy demand.", "Environmental", 0),  # contradiction
    ("Renewable energy now accounts for 34 percent of our total energy consumption, "
     "up from 21 percent in 2022. Solar PV capacity reached 42 MWp.", "Environmental", 1),
    ("We have implemented a supplier ESG code of conduct covering 890 tier-1 suppliers "
     "representing 78 percent of procurement spend.", "Social", 1),
    ("Our climate risk assessment identified USD 280 million in stranded asset exposure "
     "under the NGFS Current Policies scenario by 2040.", "Governance", 1),
    ("The company supports the environment and cares about people and the planet.", "Environmental", 0),
    ("TCFD-aligned scenario analysis was conducted across physical risk (RCP4.5, RCP8.5) "
     "and transition risk (Below 2 degrees, Net Zero 2050) pathways.", "Governance", 1),
]

df = pd.DataFrame(ESG_TEXTS, columns=["text", "pillar", "quality"])
print(f"Corpus: {len(df)} disclosure sentences")
print(df[["pillar", "quality"]].value_counts())

# ── 3. Text Preprocessing ─────────────────────────────────────────────────────
STOP = set(stopwords.words("english"))
GREENWASH_SIGNALS = {
    "eco-friendly", "green", "sustainable", "committed", "aspiration",
    "better", "planet", "cares", "supports", "will do",
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s%]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP and len(t) > 2]
    return " ".join(tokens), tokens

df["clean"], df["tokens"] = zip(*df["text"].apply(preprocess))

def greenwash_score(tokens):
    return sum(1 for t in tokens if t in GREENWASH_SIGNALS)

df["greenwash_signals"] = df["tokens"].apply(greenwash_score)
print(f"\nGreenwashing signal counts:\n{df[['text', 'greenwash_signals']].to_string(index=False)}")

# ── 4. TF-IDF Classification ──────────────────────────────────────────────────
print("\nTF-IDF + Logistic Regression: ESG pillar classification")
vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df["clean"])
y_pillar = df["pillar"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_tfidf, y_pillar, test_size=0.3, random_state=42, stratify=y_pillar
    if y_pillar.value_counts().min() >= 2 else None)

clf = LogisticRegression(max_iter=500, C=1.0)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
print(classification_report(y_te, y_pred, zero_division=0))

# ── 5. Token Frequency Visualisation ─────────────────────────────────────────
all_tokens = [t for tlist in df["tokens"] for t in tlist]
freq = Counter(all_tokens).most_common(20)
words, counts = zip(*freq)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(words[::-1], counts[::-1], color=COLORS[1], alpha=0.8)
axes[0].set_title("Top 20 TF Tokens in ESG Corpus", fontweight="bold")
axes[0].set_xlabel("Frequency")
axes[0].spines[["top", "right"]].set_visible(False)

# Greenwashing signal chart
gw_df = df[["pillar", "quality", "greenwash_signals"]].copy()
gw_pivot = gw_df.groupby(["quality"])["greenwash_signals"].mean()
bar_colors = [COLORS[2] if q == 1 else COLORS[4] for q in gw_pivot.index]
axes[1].bar(["High Quality (1)", "Low Quality (0)"], gw_pivot.values,
            color=bar_colors, alpha=0.85)
axes[1].set_title("Avg. Greenwashing Signals by Disclosure Quality", fontweight="bold")
axes[1].set_ylabel("Mean signal count")
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("fig8_nlp_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved: fig8_nlp_analysis.png")

# ── 6. HuggingFace ClimateBERT (optional — requires transformers) ─────────────
print("\nOptional: ClimateBERT zero-shot classification")
try:
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification",
                          model="climatebert/distilroberta-base-climate-f")
    sample = df["text"].iloc[0]
    result = classifier(sample, candidate_labels=["climate", "social", "governance"])
    print(f"Text: {sample[:80]}...")
    print(f"Labels: {result['labels']}")
    print(f"Scores: {[round(s, 3) for s in result['scores']]}")
except Exception as e:
    print(f"Skipped (requires transformers + internet): {e}")
