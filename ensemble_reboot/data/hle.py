"""HLE (Humanity's Last Exam) text-only loader.

Source: HuggingFace macabdul9/hle_text_only (community text-only mirror).
Falls back to cais/hle-rolling or cais/hle if the mirror is unavailable.

Filters out Math category (CC prefers deep reasoning over math benchmarks).
For items > 500, stratified-sample to ~250 by category.

MC items: choices are embedded in the question text after "Answer Choices:".
Parse them out, shuffle with per-item seed, record correct letter, grade by
exact letter extraction. Short-answer (exactMatch) items: normalize and
compare.

Dataset text is NEVER committed (data/ is gitignored); manifest keeps item ids
+ sha256 only.
"""

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Try community text-only mirror first (no auth needed), then gated repos
HF_REPOS = [
    ("macabdul9/hle_text_only", "data/test-00000-of-00001.parquet"),
    ("cais/hle-rolling", "data/train-00000-of-00001.parquet"),
    ("cais/hle", "data/train-00000-of-00001.parquet"),
]
TARGET_SAMPLE_SIZE = 250
TEXT_ONLY_LIMIT = 500
EXCLUDED_CATEGORIES = {"Math"}  # CC: "not interested in math"


@dataclass
class HLEItem:
    item_id: str
    question: str
    choices: dict | None       # letter -> text (MC only); None for short-answer
    correct_letter: str | None  # MC only
    correct_answer: str | None  # short-answer only (normalized)
    question_type: str          # "multiple_choice" or "short_answer"
    subject: str
    category: str
    item_sha256: str

    @property
    def prompt(self) -> str:
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
        else:
            lines = [
                "Answer the following question concisely.",
                "",
                self.question.strip(),
                "",
                "Give your final answer on a line starting with 'Answer:'.",
            ]
        return "\n".join(lines)


def _normalize_answer(text: str) -> str:
    """Normalize short-answer text for exact-match comparison."""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip('.,;:!?')
    return text


def _parse_mc_question(raw_question: str):
    """Parse MC question with embedded choices.

    HLE format: question text, then "Answer Choices:\\n" followed by
    lines like "A. choice text".

    Returns (question_text, [(letter, choice_text), ...]) or None if
    not parseable.
    """
    parts = re.split(r'\n\s*Answer Choices:\s*\n', raw_question)
    if len(parts) != 2:
        return None
    question_text = parts[0].strip()
    choices_text = parts[1].strip()
    choices = re.findall(r'^([A-H])\.\s+(.+?)$', choices_text, re.MULTILINE)
    if not choices:
        return None
    return question_text, choices


def download_hle(data_dir) -> Path:
    """Download HLE dataset from HuggingFace."""
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "hle_text_only.parquet"
    if dest.exists():
        return dest

    errors = []
    for repo, filename in HF_REPOS:
        try:
            src = hf_hub_download(repo, filename, repo_type="dataset",
                                  cache_dir=str(data_dir / ".hf_cache"))
            import shutil
            shutil.copy(src, dest)
            print(f"Downloaded HLE from {repo}")
            return dest
        except Exception as e:
            errors.append(f"{repo}: {e}")
            continue

    raise RuntimeError(
        f"Failed to download HLE from any source:\n" +
        "\n".join(errors)
    )


def load_hle_items(data_dir, shuffle_salt: str = "hle-shuffle-v1") -> list:
    """Load HLE text-only items.

    - Filters out Math category
    - MC items: parses choices from question text, shuffles with per-item seed
    - exactMatch items: normalizes answer
    - Stratified-samples to ~250 by category if > TEXT_ONLY_LIMIT items
    """
    import pyarrow.parquet as pq

    parquet_path = download_hle(data_dir)
    table = pq.read_table(parquet_path)
    all_items = []

    for row in table.to_pylist():
        # Skip items with images (should be none in text-only mirror, but guard)
        img = row.get("image", "")
        if img and str(img).strip():
            continue

        question_raw = row.get("question", "").strip()
        if not question_raw:
            continue

        category = row.get("category", "Other")
        if category in EXCLUDED_CATEGORIES:
            continue

        answer_type = row.get("answer_type", "")
        subject = row.get("raw_subject", "unknown")
        hle_id = row.get("id", "")
        sha = hashlib.sha256(question_raw.encode()).hexdigest()
        item_id = f"hle-{hle_id}" if hle_id else f"hle-{sha[:16]}"

        if answer_type == "multipleChoice":
            parsed = _parse_mc_question(question_raw)
            if parsed is None:
                continue
            question_text, orig_choices = parsed
            correct_letter_orig = row.get("answer", "").strip().upper()

            # Shuffle choices with per-item seed
            rng = random.Random(f"{shuffle_salt}:{item_id}")
            indices = list(range(len(orig_choices)))
            rng.shuffle(indices)

            choices = {}
            correct_letter = None
            for new_pos, orig_idx in enumerate(indices):
                letter = LETTERS[new_pos]
                choices[letter] = orig_choices[orig_idx][1]  # choice text
                if orig_choices[orig_idx][0] == correct_letter_orig:
                    correct_letter = letter

            if correct_letter is None:
                continue  # couldn't map the correct answer

            all_items.append(HLEItem(
                item_id=item_id,
                question=question_text,
                choices=choices,
                correct_letter=correct_letter,
                correct_answer=None,
                question_type="multiple_choice",
                subject=subject,
                category=category,
                item_sha256=sha,
            ))

        elif answer_type == "exactMatch":
            answer = row.get("answer", "").strip()
            if not answer:
                continue

            all_items.append(HLEItem(
                item_id=item_id,
                question=question_raw,
                choices=None,
                correct_letter=None,
                correct_answer=_normalize_answer(answer),
                question_type="short_answer",
                subject=subject,
                category=category,
                item_sha256=sha,
            ))

    # Stratified sample if too many items
    if len(all_items) > TEXT_ONLY_LIMIT:
        all_items = _stratified_sample(all_items, TARGET_SAMPLE_SIZE, shuffle_salt)

    return sorted(all_items, key=lambda x: x.item_id)


def _stratified_sample(items, target, shuffle_salt):
    """Proportional sampling by category, no single category > 30%."""
    by_cat = {}
    for item in items:
        by_cat.setdefault(item.category, []).append(item)

    max_per_cat = int(target * 0.30)
    rng = random.Random(f"{shuffle_salt}:sample")

    # First pass: proportional allocation, capped
    allocated = {}
    total = len(items)
    for cat, cat_items in by_cat.items():
        n = min(max_per_cat, max(1, round(target * len(cat_items) / total)))
        allocated[cat] = n

    # Scale to hit target
    alloc_total = sum(allocated.values())
    if alloc_total < target:
        deficit = target - alloc_total
        # Add to categories with room, largest first
        for cat in sorted(by_cat, key=lambda c: len(by_cat[c]), reverse=True):
            if deficit <= 0:
                break
            room = min(deficit, len(by_cat[cat]) - allocated[cat], max_per_cat - allocated[cat])
            if room > 0:
                allocated[cat] += room
                deficit -= room

    sampled = []
    for cat, cat_items in by_cat.items():
        n = allocated.get(cat, 0)
        if len(cat_items) <= n:
            sampled.extend(cat_items)
        else:
            sampled.extend(rng.sample(cat_items, n))

    return sampled[:target]
