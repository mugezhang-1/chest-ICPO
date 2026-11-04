"""MI-CXR items, prompts and scoring (Exp 1).

The paper (Cho et al., ACL Findings 2026) does not publish its evaluation prompt. It states: zero-shot, single step,
temperature 0, five images labeled T1-T5 in chronological order, models instructed to output exactly one choice
among A-E, one deterministic rule-based extraction, and invalid / multiple / ambiguous answers scored as wrong.
NATIVE_PROMPT is our reconstruction of that protocol; reproducing the paper's numbers is how we check it.
"""

import json
from pathlib import Path

# MI-CXR qtype -> the column of the paper's Table 1
FAMILY = {
    "single_emergence": "TEL_single", "single_resolution": "TEL_single",
    "multiple_emergence_1": "TEL_multiple", "multiple_emergence_2": "TEL_multiple",
    "multiple_resolution_1": "TEL_multiple", "multiple_resolution_2": "TEL_multiple",
    "emergence_to_resolution": "TEL_ordered", "resolution_to_emergence": "TEL_ordered",
    "icr_multi_abn": "ICR",
    "single_entity_interval_summary": "GTS_single", "multi_entity_interval_summary": "GTS_multi",
}
COLUMNS = ["TEL_single", "TEL_multiple", "TEL_ordered", "ICR", "GTS_single", "GTS_multi", "overall"]

# Table 1 of the MI-CXR paper (temperature 0). Column order follows Table 1's header.
PAPER_TABLE1 = {
    "QwenVL3-8B": dict(zip(COLUMNS, [0.237, 0.295, 0.185, 0.164, 0.328, 0.236, 0.234])),
    "Lingshu-7B": dict(zip(COLUMNS, [0.230, 0.260, 0.165, 0.189, 0.324, 0.194, 0.223])),
    "MedGemma-4B": dict(zip(COLUMNS, [0.174, 0.196, 0.301, 0.281, 0.259, 0.183, 0.237])),
}

NATIVE_INTRO = ("Below are five chest X-ray images from the same patient, labeled T1 through T5 in "
                "chronological order.")
NATIVE_INSTRUCTION = "Answer with exactly one option letter (A, B, C, D, or E) and nothing else."


def load_items(path: str | Path) -> list[dict]:
    with open(path) as f:
        items = [json.loads(line) for line in f]
    items.sort(key=lambda r: r["q_id"])
    for r in items:
        if sorted(r["images"]) != ["T1", "T2", "T3", "T4", "T5"] or sorted(r["choices"]) != list("ABCDE"):
            raise ValueError(f"q_id {r['q_id']}: unexpected images or choices")
        if r["answer"] not in r["choices"] or r["qtype"] not in FAMILY:
            raise ValueError(f"q_id {r['q_id']}: bad answer or qtype")
    return items


def native_messages(item: dict, image_root: Path, order: tuple = ("T1", "T2", "T3", "T4", "T5"),
                    choices: dict | None = None, intro: str = NATIVE_INTRO, instruction: str = NATIVE_INSTRUCTION,
                    show_images: bool = True) -> list[dict]:
    """One user turn: intro, then 'T1:' + image ... 'T5:' + image, then question, options and instruction.

    `order` lets probes present the images in another order while keeping the labels T1..T5 in place; `choices`
    replaces the option texts (presented letter -> text); `show_images=False` withholds every image. The defaults
    give the exact prompt of the part-A runs.
    """
    choices = item["choices"] if choices is None else choices
    content = [{"type": "text", "text": intro}]
    for label, src in zip(("T1", "T2", "T3", "T4", "T5"), order):
        if show_images:
            content.append({"type": "text", "text": f"\n{label}:"})
            content.append({"type": "image_url", "image_url": {"url": f"file://{image_root / item['images'][src]}"}})
        else:
            content.append({"type": "text", "text": f"\n{label}: [image withheld]"})
    opts = "\n".join(f"{k}. {choices[k]}" for k in "ABCDE")
    content.append({"type": "text", "text": f"\n\nQuestion: {item['question']}\n{opts}\n\n{instruction}"})
    return [{"role": "user", "content": content}]


def summarize(rows: list[dict]) -> dict:
    """Accuracy per Table-1 column and overall; rows need 'qtype' and 'correct'."""
    out = {}
    for col in COLUMNS[:-1]:
        sel = [r["correct"] for r in rows if FAMILY[r["qtype"]] == col]
        out[col] = {"n": len(sel), "acc": round(sum(sel) / len(sel), 4) if sel else None}
    out["overall"] = {"n": len(rows), "acc": round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None}
    return out
