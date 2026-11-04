"""Second readers (labels v2, R4; docs/label_fix_design.md): a language model answers an item from the full reports.

The reader sees every report of the window (T1..Tn, including the last), the question and our options with the
image-side "Cannot be assessed" option replaced by UNDETERMINED. It never sees our answer. An item is kept only when
the readers choose our answer. The prompt text is versioned; any change to it needs a new PROMPT_VERSION.
"""

import hashlib
import re

from icpo.data.reports import sections

PROMPT_VERSION = "reader_v1"
CANNOT = "Cannot be assessed from these images"
UNDETERMINED = "The reports do not determine the answer"
LETTERS = "ABCDEFGHIJ"
MAX_REPORT_CHARS = 4000

DEFINITIONS = {
    "Pleural Effusion": "pleural effusion (fluid in the pleural space) on either side, of any size",
    "Edema": "pulmonary edema of any degree, including interstitial edema",
    "Consolidation": "consolidation of the lung",
    "Atelectasis": "atelectasis or collapse of any part of the lung",
    "Pneumothorax": "pneumothorax (including hydropneumothorax) on either side, of any size",
    "Cardiomegaly": "an enlarged heart or cardiac silhouette",
    "Lung Opacity": ("any opacity of the lung parenchyma, including edema, consolidation, atelectasis, pneumonia, "
                     "masses and nodules"),
}

RULES = """How to read the options:
- Newly appeared: absent on the earlier exam, present on the later one. Resolved: present on the earlier exam, \
absent on the later one. Improved / Worsened / Unchanged: present on both exams and less severe / more severe / \
about the same on the later one.
- Course from the first to the last exam: "Improves or resolves, with no worsening" means at least one interval \
improves or resolves and no interval worsens or newly appears; "Worsens or newly appears, with no improvement" is \
the reverse; "First worsens, then improves" and "First improves, then worsens" mean exactly one change of \
direction; "Present and unchanged throughout" means present on every exam and never changing.
- "First newly appear": the first interval in which the finding goes from absent to present. "First resolve": the \
first interval in which it goes from present to absent. The "does not ... in this sequence" option means no such \
interval exists among these exams.
- A report usually compares with the immediately preceding exam; dates are redacted as ___, so use the context.
- When parts of the finding change in different directions (for example left worse, right better) and the report \
gives no overall direction, the answer is not determined.
- Choose "{undetermined}" when the reports do not state or clearly imply the answer: the finding is not mentioned \
on an exam the answer depends on, the wording is too hedged to decide, or the reports contradict each other."""


def report_text(raw: str) -> str:
    """Whole report, one line per section, whitespace normalised, capped at MAX_REPORT_CHARS."""
    t = "\n".join(f"{n}: {b}" if n else b for n, b in sections(raw))
    return t[:MAX_REPORT_CHARS]


def reader_options(options: list[str]) -> list[str]:
    return [UNDETERMINED if o == CANNOT else o for o in options]


def build_prompt(item: dict, reports: list[str]) -> str:
    """item: finding, question, options; reports: report text per window study, in order."""
    n = len(reports)
    f = item["finding"]
    opts = reader_options(list(item["options"]))
    body = "\n\n".join(f"T{k + 1} report:\n{r}" for k, r in enumerate(reports))
    return (
        "You are checking the labels of a chest X-ray dataset. Below are the complete radiology reports of one "
        f"patient's {n} chest X-ray exams, T1 to T{n} in chronological order. Answer the question using only these "
        f"reports.\n\nIn this question, \"{f.lower()}\" means {DEFINITIONS[f]}.\n\n"
        + RULES.format(undetermined=UNDETERMINED)
        + f"\n\n{body}\n\nQuestion: {item['question']}\n"
        + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        + "\n\nThink it through, then end your reply with <answer>X</answer>, where X is one option letter."
    )


def request_id(qid: str) -> str:
    return hashlib.sha256(f"{PROMPT_VERSION}|{qid}".encode()).hexdigest()[:16]


_ANSWER = re.compile(r"<answer>\s*\(?([A-J])\)?\s*</answer>")


def parse_answer(text: str, n_options: int) -> str | None:
    """Letter in the LAST <answer> tag, if it names one of the options; otherwise None (counts as disagreement)."""
    hits = _ANSWER.findall(text or "")
    if not hits:
        return None
    letter = hits[-1]
    return letter if LETTERS.index(letter) < n_options else None
