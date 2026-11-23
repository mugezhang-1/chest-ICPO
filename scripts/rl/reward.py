
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

    # Prefer the structured answer tag.
    matches = ANSWER_PATTERN.findall(solution_str)

    if matches:
        return matches[-1].upper()

    # Remove complete and incomplete thinking sections.
    text = THINK_PATTERN.sub("", solution_str)
    text = re.sub(
        r"<think>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove common thinking-mode special tokens.
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.strip()

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
