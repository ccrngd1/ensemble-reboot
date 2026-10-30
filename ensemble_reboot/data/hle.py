"""HLE (Humanity's Last Exam) text-only loader.

Source: HuggingFace cais/hle-rolling (preferred) or cais/hle as fallback.
Filter to text-only items (no image field). For items > 500, stratified sample
to ~250 by subject.

MC items: shuffle choices with per-item seed (same pattern as GPQA), grade by
exact letter extraction. Short-answer items: exact-match grading with
normalization.

Dataset text is NEVER committed (data/ is gitignored); manifest keeps item ids
+ sha256 only.
"""

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]  # HLE can have up to 8 choices

# Try hle-rolling first (Sept 2026 cleaned version), fall back to hle
HF_REPOS = ["cais/hle-rolling", "cais/hle"]
TARGET_SAMPLE_SIZE = 250
TEXT_ONLY_LIMIT = 500


@dataclass
class HLEItem:
    item_id: str
    question: str
    choices: dict | None      # letter -> text (MC only); None for short-answer
    correct_letter: str | None  # MC only
    correct_answer: str | None  # short-answer only
    question_type: str          # "multiple_choice" or "short_answer"
    subject: str
    item_sha256: str

    @property
    def prompt(self) -> str:
        lines = []
        if self.question_type == "multiple_choice":
            lines = [
                "Answer the following multiple-choice question.",
                "",
                self.question.strip(),
                "",
            ]
            for letter in sorted(self.choices.keys()):
                lines.append(f"{letter}) {self.choices[letter]}")
            lines += [
                "",
                'End your response with a line of the form "Answer: X" where X is'
                " one of the choice letters.",
            ]
        else:  # short_answer
            lines = [
                "Answer the following question concisely.",
                "",
                self.question.strip(),
                "",
                "Give your final answer on a line starting with 'Answer:'.",
            ]
        return "\n".join(lines)


def _item_id(question: str, idx: int) -> str:
    """Generate item_id from question hash and index."""
    return f"hle-{hashlib.sha256(question.encode()).hexdigest()[:12]}-{idx}"


def _normalize_answer(text: str) -> str:
    """Normalize short-answer text for exact-match comparison."""
    # Remove extra whitespace, lowercase, strip punctuation at boundaries
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip('.,;:!?')
    return text


def download_hle(data_dir) -> Path:
    """Download HLE dataset from HuggingFace, trying rolling first."""
    from huggingface_hub import hf_hub_download
    import shutil

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Try each repo in order
    for repo in HF_REPOS:
        try:
            dest = data_dir / f"hle_{repo.replace('/', '_')}.parquet"
            if dest.exists():
                return dest

            # Try to download the parquet file
            src = hf_hub_download(
                repo,
                "data/train-00000-of-00001.parquet",
                repo_type="dataset"
            )
            shutil.copy(src, dest)
            print(f"Downloaded HLE from {repo}")
            return dest
        except Exception as e:
            print(f"Could not download from {repo}: {e}")
            continue

    raise RuntimeError(f"Failed to download HLE from any of: {HF_REPOS}")


def load_hle_items(data_dir, shuffle_salt: str = "hle-shuffle-v1") -> list:
    """Load HLE text-only items with deterministically shuffled choices.

    Filters to text-only (no image field or image is null/empty).
    If text-only items exceed TEXT_ONLY_LIMIT, stratified-sample to
    TARGET_SAMPLE_SIZE by subject.
    """
    import pyarrow.parquet as pq

    parquet_path = download_hle(data_dir)
    table = pq.read_table(parquet_path)
    all_items = []

    for idx, row in enumerate(table.to_pylist()):
        # Filter to text-only: skip if image field exists and is non-empty
        if row.get("image") and str(row["image"]).strip():
            continue

        question = row.get("question", "").strip()
        if not question:
            continue

        question_type = row.get("question_type", "multiple_choice")
        subject = row.get("subject", "unknown")
        iid = _item_id(question, idx)

        if question_type == "multiple_choice":
            # Parse choices and correct answer
            choices_raw = row.get("choices", [])
            if not choices_raw:
                continue

            correct_idx = row.get("correct_answer_idx")
            if correct_idx is None:
                continue

            # Shuffle choices with per-item seed
            rng = random.Random(f"{shuffle_salt}:{iid}")
            indices = list(range(len(choices_raw)))
            rng.shuffle(indices)

            # Map to letters
            choices = {}
            correct_letter = None
            for i, orig_idx in enumerate(indices):
                letter = LETTERS[i]
                choices[letter] = choices_raw[orig_idx]
                if orig_idx == correct_idx:
                    correct_letter = letter

            all_items.append(HLEItem(
                item_id=iid,
                question=question,
                choices=choices,
                correct_letter=correct_letter,
                correct_answer=None,
                question_type="multiple_choice",
                subject=subject,
                item_sha256=hashlib.sha256(question.encode()).hexdigest(),
            ))
        else:  # short_answer
            correct_answer = row.get("correct_answer", "").strip()
            if not correct_answer:
                continue

            all_items.append(HLEItem(
                item_id=iid,
                question=question,
                choices=None,
                correct_letter=None,
                correct_answer=_normalize_answer(correct_answer),
                question_type="short_answer",
                subject=subject,
                item_sha256=hashlib.sha256(question.encode()).hexdigest(),
            ))

    # If we have too many text-only items, stratified sample by subject
    if len(all_items) > TEXT_ONLY_LIMIT:
        by_subject = {}
        for item in all_items:
            by_subject.setdefault(item.subject, []).append(item)

        # Proportional sampling per subject
        sampled = []
        rng = random.Random(f"{shuffle_salt}:sample")
        for subject, items in sorted(by_subject.items()):
            n_from_subject = max(1, int(TARGET_SAMPLE_SIZE * len(items) / len(all_items)))
            if len(items) <= n_from_subject:
                sampled.extend(items)
            else:
                sampled.extend(rng.sample(items, n_from_subject))

        # If we're under target, top up from largest subjects
        while len(sampled) < TARGET_SAMPLE_SIZE:
            largest = max(by_subject.items(), key=lambda x: len(x[1]))
            subject, items = largest
            remaining = [i for i in items if i not in sampled]
            if remaining:
                sampled.append(rng.choice(remaining))
            else:
                break

        all_items = sampled[:TARGET_SAMPLE_SIZE]

    return sorted(all_items, key=lambda x: x.item_id)
