"""
Shared train/test split for all three methods.

Everyone (LogReg, rule-based, LLM) should score on the SAME test set, so this
script writes the split to disk as lists of review Ids. Run once, commit the
splits/ folder, and have every method load from it.

    python make_split.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA = "Reviews.csv.xz"
SPLIT_DIR = Path("splits")
LLM_SAMPLE_SIZE = 600   # LLM is scored on a stratified subset of the test set (cost)
MIN_VOTES = 5           # from the proposal: only reviews with >= 5 helpfulness votes


def load_labeled(path: str = DATA) -> pd.DataFrame:
    """Load reviews, apply the proposal's filters, and add the binary label.

    TODO: once clean.py's output is settled, swap this to load that file
    instead, so all three methods share exactly one cleaning step.
    """
    df = pd.read_csv(path)  # pandas reads .xz directly

    # Filters
    df = df[df.HelpfulnessDenominator >= MIN_VOTES]
    df = df[df.HelpfulnessNumerator <= df.HelpfulnessDenominator]  # a few rows are corrupt
    # This dataset repeats the same review across product variants; drop
    # duplicates BEFORE splitting or the same text lands in train and test.
    df = df.drop_duplicates(subset=["Text"])

    # Light text cleanup
    df["Text"] = df.Text.str.replace(r"<br\s*/?>", " ", regex=True).str.strip()
    df["Summary"] = df.Summary.fillna("").astype(str)

    # Label: helpful if a majority of voters found it helpful
    df["helpful"] = (df.HelpfulnessNumerator / df.HelpfulnessDenominator > 0.5).astype(int)
    return df.reset_index(drop=True)


def main():
    df = load_labeled()
    print(f"{len(df):,} reviews after filtering")
    print("label balance:\n", df.helpful.value_counts(normalize=True).round(3))

    train, test = train_test_split(
        df, test_size=0.2, stratify=df.helpful, random_state=SEED
    )

    # LLM subset: stratify on label x star rating so the sample keeps the
    # test set's mix (also needed for the bias-by-star-rating analysis).
    strata = test.helpful.astype(str) + "_" + test.Score.astype(str)
    _, llm_sample = train_test_split(
        test, test_size=LLM_SAMPLE_SIZE, stratify=strata, random_state=SEED
    )

    SPLIT_DIR.mkdir(exist_ok=True)
    train[["Id"]].to_csv(SPLIT_DIR / "train_ids.csv", index=False)
    test[["Id"]].to_csv(SPLIT_DIR / "test_ids.csv", index=False)
    llm_sample[["Id"]].to_csv(SPLIT_DIR / "llm_sample_ids.csv", index=False)
    print(f"train={len(train):,}  test={len(test):,}  llm_sample={len(llm_sample):,}")


if __name__ == "__main__":
    main()