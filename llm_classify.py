"""
Method 3: Claude as a zero-shot / few-shot / rubric-guided helpfulness classifier.

Setup:
    pip install anthropic pandas scikit-learn
    export ANTHROPIC_API_KEY=...        # never commit this
    python make_split.py                # once, creates splits/

Usage:
    python llm_classify.py run --strategy zero_shot --limit 20   # sanity check first
    python llm_classify.py run --strategy zero_shot
    python llm_classify.py run --strategy few_shot --shots 6
    python llm_classify.py run --strategy rubric
    python llm_classify.py run --strategy zero_shot --no-score   # ablation: hide star rating
    python llm_classify.py eval

Results are appended to results/llm/<run>.jsonl one review at a time, so a
crashed or interrupted run picks up where it left off without re-paying.
"""
import argparse
import json
from pathlib import Path

import anthropic
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from make_split import SEED, load_labeled

MODEL = "claude-haiku-5-5"   # cheap + fast; try a bigger model later as an extra axis
OUT_DIR = Path("results/llm")
SPLIT_DIR = Path("splits")

SYSTEM = (
    "You are labeling Amazon fine-food product reviews. A review is 'helpful' if "
    "most shoppers who read it would find it useful when deciding whether to buy "
    "the product. Respond with exactly one word: helpful or not_helpful."
)

RUBRIC = """When judging, consider what shoppers tend to value:
- Specific details about taste, texture, ingredients, packaging, or quantity
- Concrete numbers such as prices, sizes, or amounts
- Comparisons to other products
- Explaining WHY the reviewer liked or disliked the product
- Clear, readable writing rather than only an emotional reaction"""

TASK = "Is the following review helpful or not_helpful?"


# ---------- prompt construction ----------

def format_review(row, include_score: bool) -> str:
    lines = []
    if include_score:
        lines.append(f"Star rating: {row.Score}/5")
    lines.append(f"Summary: {row.Summary}")
    lines.append(f"Review: {row.Text}")
    return "\n".join(lines)


def build_messages(row, strategy: str, shots: pd.DataFrame, include_score: bool):
    task = f"{RUBRIC}\n\n{TASK}" if strategy == "rubric" else TASK
    messages = []
    if strategy == "few_shot":
        # Examples are given as prior conversation turns
        for _, ex in shots.iterrows():
            messages.append({"role": "user", "content": f"{task}\n\n{format_review(ex, include_score)}"})
            messages.append({"role": "assistant", "content": "helpful" if ex.helpful else "not_helpful"})
    messages.append({"role": "user", "content": f"{task}\n\n{format_review(row, include_score)}"})
    return messages


def pick_shots(train: pd.DataFrame, n: int) -> pd.DataFrame:
    """Balanced few-shot examples from TRAIN only (never the test set).
    Prefers shorter reviews to keep the prompt cheap."""
    short = train[train.Text.str.len() < 800]
    pos = short[short.helpful == 1].sample(n // 2, random_state=SEED)
    neg = short[short.helpful == 0].sample(n - n // 2, random_state=SEED)
    return pd.concat([pos, neg]).sample(frac=1, random_state=SEED)  # shuffle order


def parse_label(text: str):
    t = text.strip().lower()
    if t.startswith("not"):
        return 0
    if t.startswith("helpful"):
        return 1
    return None  # unparseable -> counted separately in eval


# ---------- running ----------

def run_name(strategy: str, include_score: bool, shots: int) -> str:
    name = strategy
    if strategy == "few_shot":
        name += f"_{shots}"
    if not include_score:
        name += "_noscore"
    return name


def run(strategy: str, include_score: bool, n_shots: int, limit: int | None):
    df = load_labeled().set_index("Id", drop=False)
    sample = df.loc[pd.read_csv(SPLIT_DIR / "llm_sample_ids.csv").Id]
    if limit:
        sample = sample.sample(limit, random_state=SEED)
    train = df.loc[pd.read_csv(SPLIT_DIR / "train_ids.csv").Id]
    shots = pick_shots(train, n_shots) if strategy == "few_shot" else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{run_name(strategy, include_score, n_shots)}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["Id"] for line in out.open()}

    client = anthropic.Anthropic(max_retries=5)  # SDK retries rate limits/5xx
    todo = sample[~sample.Id.isin(done)]
    print(f"{out.name}: {len(done)} done, {len(todo)} to go")

    with out.open("a") as f:
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1024,  # room for thinking; the answer itself is one word
                # No temperature: SDK v1.0 removed it, and newer models
                # reject it anyway. Determinism is checked by re-running instead.
                system=SYSTEM,
                messages=build_messages(row, strategy, shots, include_score),
            )
            # The model may return a thinking block before its answer, so
            # take only the text blocks.
            raw = "".join(b.text for b in resp.content if b.type == "text")
            thought = any(b.type == "thinking" for b in resp.content)
            f.write(json.dumps({
                "Id": int(row.Id),
                "raw": raw,
                "pred": parse_label(raw),
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "thought": thought,
                "stop_reason": resp.stop_reason,
            }) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")


# ---------- evaluation ----------

def evaluate():
    df = load_labeled().set_index("Id", drop=False)
    rows = []
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        preds = pd.read_json(path, lines=True).set_index("Id")
        merged = preds.join(df[["helpful", "Score", "Text"]])
        ok = merged.dropna(subset=["pred"])
        y, p = ok.helpful, ok.pred.astype(int)
        rows.append({
            "run": path.stem,
            "n": len(merged),
            "unparsed": int(merged.pred.isna().sum()),
            "accuracy": accuracy_score(y, p),
            "f1_helpful": f1_score(y, p),
            "f1_macro": f1_score(y, p, average="macro"),
            "avg_input_tokens": merged.input_tokens.mean(),
        })

        # Bias checks from the proposal: accuracy by star rating and by length
        ok = ok.assign(
            correct=(y == p),
            length=pd.qcut(ok.Text.str.split().str.len(), 4,
                           labels=["short", "med", "long", "very long"]),
        )
        print(f"\n== {path.stem} ==")
        print("accuracy by star rating:\n", ok.groupby("Score").correct.mean().round(3).to_string())
        print("accuracy by length quartile:\n", ok.groupby("length", observed=True).correct.mean().round(3).to_string())
        print("predicted-helpful rate by star rating:\n", ok.groupby("Score").pred.mean().round(3).to_string())

    summary = pd.DataFrame(rows).round(3)
    print("\n", summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--strategy", choices=["zero_shot", "few_shot", "rubric"], required=True)
    r.add_argument("--shots", type=int, default=6)
    r.add_argument("--no-score", action="store_true", help="hide the star rating from the prompt")
    r.add_argument("--limit", type=int, default=None, help="only run on N reviews (for testing)")
    sub.add_parser("eval")
    args = ap.parse_args()

    if args.cmd == "run":
        run(args.strategy, not args.no_score, args.shots, args.limit)
    else:
        evaluate()


if __name__ == "__main__":
    main()