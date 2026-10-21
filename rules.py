import re
import numpy as np
import pandas as pd
import textstat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load, filter (>=5 votes), dedupe, label (helpful = >50% of voters)
df = pd.read_csv('Reviews.csv.xz')
df = df[(df['HelpfulnessDenominator'] >= 5) &
        (df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator'])]
df = df.drop_duplicates(subset='Text')
df['helpful'] = (df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'] > 0.5).astype(int)

# 2. Split 70/15/15, stratified
train, temp = train_test_split(df, test_size=0.30, stratify=df['helpful'], random_state=42)
dev, test = train_test_split(temp, test_size=0.50, stratify=temp['helpful'], random_state=42)

# 3. Features
HTML = re.compile(r'<[^>]+>')
TOK  = re.compile(r"[a-z']+")
SPEC = re.compile(r'\$\s?\d+|\b\d+(?:\.\d+)?\s?(?:%|oz|lbs?|g|mg|grams?|calories|cups?|pack|count)?\b', re.I)
COMP = re.compile(r'\b(?:than|compared (?:to|with)|instead of|similar to|unlike|better|worse|prefer)\b', re.I)

# Can play around with these (placeholders)
HELPFUL_WORDS = {'ingredients', 'texture', 'sodium', 'sugar', 'protein', 'expiration',
                 'shipping', 'package', 'brand', 'aftertaste', 'calories'}
VAGUE_WORDS   = {'yummy', 'awesome', 'love', 'amazing', 'best', 'great', 'delicious'}

def featurize(texts):
    rows = []
    for t in texts:
        clean = HTML.sub(' ', str(t))
        toks = TOK.findall(clean.lower())
        n = max(len(toks), 1)
        rows.append({
            'length':      len(toks),
            'lexicon':     (sum(w in HELPFUL_WORDS for w in toks) - sum(w in VAGUE_WORDS for w in toks)) / n,
            'specificity': len(SPEC.findall(clean)) / n * 100,
            'comparison':  len(COMP.findall(clean)) / n * 100,
            'readability': np.clip(textstat.flesch_reading_ease(clean), 0, 100),
        })
    return pd.DataFrame(rows)

X_tr, X_dv, X_te = (featurize(d['Text']) for d in (train, dev, test))

# 4. Scale to [0, 1]: clip at 5th/95th percentile of train
lo, hi = X_tr.quantile(0.05), X_tr.quantile(0.95)
scale = lambda X: (X.clip(lo, hi, axis=1) - lo) / (hi - lo + 1e-9)

# 5. Hand-set weights (placeholders)
WEIGHTS = {'length': 2.0, 'lexicon': 1.0, 'specificity': 1.0, 'comparison': 0.5, 'readability': -0.5}
score = lambda X: sum(w * scale(X)[c] for c, w in WEIGHTS.items())

# 6. Cutoff that maximizes macro-F1 on dev
s_dv = score(X_dv)
grid = np.quantile(s_dv, np.linspace(0.01, 0.99, 99))
cutoff = grid[np.argmax([f1_score(dev['helpful'], s_dv >= t, average='macro') for t in grid])]

# 7. Evaluate
pred = (score(X_te) >= cutoff).astype(int)
print(f"n={len(df)}  helpful rate={df['helpful'].mean():.3f}")
print(f"majority baseline acc={test['helpful'].mean():.3f}  macro-F1={f1_score(test['helpful'], np.ones(len(test)), average='macro'):.3f}")
print(f"rules test acc={accuracy_score(test['helpful'], pred):.3f}  macro-F1={f1_score(test['helpful'], pred, average='macro'):.3f}")
print(classification_report(test['helpful'], pred, target_names=['not helpful', 'helpful']))