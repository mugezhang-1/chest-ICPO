"""Prompts for the internal test set (bases + variants; steps 7 and text variants).

Conditions (decided 2026-10-11):
  img_prior   images, the indication, and the impression of every prior study shown
  img         images and the indication only
  text_only   the indication and the prior impressions, no images (text-only baseline)
Rules:
  - the window's current (last) study's report is never shown, wherever a variant puts that study
  - an impression travels with its own study's image; a donor slot (v4) shows no impression
  - v5 replaces one prior impression with its validated rewrite; v6 replaces the indication
Answer format: the roadmap's structured format, parsed leniently (a bare letter is accepted).
"""

from pathlib import Path

CONDITIONS = ("img_prior", "img", "text_only", "img_oracle", "text_oracle")
ORACLE = ("img_oracle", "text_oracle")   # sanity check only: the current study's report is shown too
NO_IMAGES = ("text_only", "text_oracle")
INSTRUCTION_NATIVE = "Answer with exactly one option letter and nothing else."   # prompt-sensitivity check
INSTRUCTION = (
    "Answer in this format:\n<think>your reasoning</think>\n<answer>one option letter</answer>\n"
    "<evidence>[the T numbers of the images you relied on]</evidence>\n"
    "<flag>ok</flag>, or <flag>inconsistent</flag> if the images cannot all belong to one patient's timeline."
)
LETTERS = "ABCDEFGHIJ"


def render(base: dict, var: dict, cond: str, image_root: Path, impressions: dict, indication: str,
           v5: dict | None = None, instruction: str = INSTRUCTION) -> list[dict]:
    """One user turn. base: a bases row; var: a variants row (or v0); impressions: study_id -> prior impression;
    v5: {"slot": window index, "text": rewrite} to replace that study's impression."""
    win = [int(s) for s in base["study_ids"]]
    current = win[-1]
    slots = [int(s) for s in var["slots"]]
    m = len(slots)
    show_img = cond not in NO_IMAGES
    show_imp = cond in ("img_prior", "text_only") + ORACLE
    lead = (f"The following {m} chest X-ray images are from the same patient, labeled T1 to T{m} in chronological order."
            if show_img else
            f"A patient had {m} chest X-ray examinations, labeled T1 to T{m} in chronological order. "
            "The images are not available; only the text below is.")
    content = [{"type": "text", "text": lead + (f"\nClinical indication: {indication}" if indication else "")}]
    for k, (s, rel) in enumerate(zip(slots, var["slot_relpaths"])):
        if show_img:
            content.append({"type": "text", "text": f"\nT{k + 1}:"})
            content.append({"type": "image_url", "image_url": {"url": f"file://{image_root / rel}"}})
        if show_imp and s >= 0 and (win[s] != current or cond in ORACLE):
            text = v5["text"] if v5 is not None and v5["slot"] == s else impressions.get(win[s], "")
            if text:
                content.append({"type": "text", "text": f"\nReport impression for T{k + 1}: {text}"})
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(var["options"]))
    content.append({"type": "text", "text": f"\n\nQuestion: {base['question']}\n{opts}\n\n{instruction}"})
    return [{"role": "user", "content": content}]


def render_exist(finding: str, relpath: str, image_root: Path, instruction: str = INSTRUCTION) -> list[dict]:
    """Perception ceiling: one film, no text, 'Is <finding> present?' (Yes / No / Cannot)."""
    opts = ["Yes", "No", "Cannot be assessed from these images"]
    q = f"Is {finding.lower()} present on this chest X-ray?"
    return [{"role": "user", "content": [
        {"type": "text", "text": "T1:"}, {"type": "image_url", "image_url": {"url": f"file://{image_root / relpath}"}},
        {"type": "text", "text": "\n\nQuestion: " + q + "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
         + "\n\n" + instruction}]}], opts
