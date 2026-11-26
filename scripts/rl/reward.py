
"""
Chest-ICPO rule-based reward for veRL GRPO.

Reward:
    Correct answer   -> 1.0
    Incorrect answer -> 0.0
    Invalid format   -> 0.0

Supports:
    B
    <answer>B</answer>
    <think>reasoning...</think>
    <answer>B</answer>
    <evidence>[T3, T4]</evidence>
    <flag>ok</flag>
"""

import re


ANSWER_PATTERN = re.compile(
    r"<answer>\s*([A-F])\s*</answer>",
    re.IGNORECASE,
)

THINK_PATTERN = re.compile(
    r"<think>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)

LETTER_PATTERN = re.compile(
    r"^[A-F]$",
    re.IGNORECASE,
)


def extract_answer(solution_str):
    """Extract the final answer without reading reasoning text."""

    if not isinstance(solution_str, str):
        return None

    text = solution_str.strip()

    # Remove common generation special tokens.
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")

    # Handle both complete thinking blocks and cases where
    # the opening <think> tag is already in the input prompt.
    parts = re.split(r"</think>", text, flags=re.IGNORECASE)

    if len(parts) > 1:
        text = parts[-1]

    # Reject unfinished thinking sections.
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        return None

    text = text.strip()

    # Prefer a structured answer tag outside reasoning.
    matches = ANSWER_PATTERN.findall(text)

    if matches:
        return matches[-1].upper()

    # Accept only a standalone answer letter.
    if LETTER_PATTERN.fullmatch(text):
        return text.upper()

    return None


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    """
    veRL custom reward interface.

    Args:
        data_source: Dataset identifier.
        solution_str: Generated response.
        ground_truth: Correct answer letter.
        extra_info: Optional dataset metadata.

    Returns:
        float: 1.0 if correct, 0.0 otherwise.
    """

    if data_source != "chest_icpo":
        raise ValueError(
            f"Unsupported data source: {data_source}"
        )

    if not isinstance(ground_truth, str):
        raise ValueError("Ground truth must be a string")

    ground_truth = ground_truth.strip().upper()

    if not LETTER_PATTERN.fullmatch(ground_truth):
        raise ValueError(
            f"Invalid ground truth: {ground_truth}"
        )

    prediction = extract_answer(solution_str)

    return float(prediction == ground_truth)
