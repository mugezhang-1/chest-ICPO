"""Answer parsing shared by every evaluation and by the reward (roadmap §2.1, §6).

Output format required of trained models:
    <think> ... </think>
    <answer>LETTER</answer>
    <evidence>[study indices relied on]</evidence>
    <flag>ok</flag>  or  <flag>inconsistent</flag>

strict  : every tag exactly once, in a well-formed span; answer is one valid letter; evidence is a list of valid
          1-based study indices; flag is ok/inconsistent. Anything else -> format_ok False and all fields None.
lenient : for zero-shot models. Uses tags when present, else conservative patterns ("answer is C", a final
          line that is just a letter, an exact option text). If two different letters are equally plausible
          the answer is None (scored as wrong), never a guess.
"""

import re
from dataclasses import dataclass

TAGS = ("think", "answer", "evidence", "flag")
FLAGS = ("ok", "inconsistent")


@dataclass(frozen=True)
class Parsed:
    answer: str | None
    evidence: frozenset | None
    flag: str | None
    format_ok: bool
    error: str = ""


def letters(n_options: int) -> str:
    if not 2 <= n_options <= 26:
        raise ValueError(f"n_options must be in [2, 26], got {n_options}")
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options]


def _tag_spans(text: str, tag: str) -> list[str]:
    return re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S | re.I)


def _count_open_close(text: str, tag: str) -> tuple[int, int]:
    return (len(re.findall(rf"<{tag}>", text, flags=re.I)), len(re.findall(rf"</{tag}>", text, flags=re.I)))


def _letter(s: str, valid: str) -> str | None:
    m = re.fullmatch(r"\s*\(?\s*([A-Za-z])\s*\)?\s*\.?\s*", s)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    return None


def _evidence(s: str, n_studies: int) -> frozenset | None:
    s = s.strip()
    if not re.fullmatch(r"\[?\s*((T?\d+)(\s*,\s*T?\d+)*)?\s*\]?", s, flags=re.I):
        return None
    idx = [int(x) for x in re.findall(r"T?(\d+)", s, flags=re.I)]
    if any(not 1 <= i <= n_studies for i in idx) or len(idx) != len(set(idx)):
        return None
    return frozenset(idx)


def parse_strict(text: str, n_options: int, n_studies: int) -> Parsed:
    valid = letters(n_options)
    for tag in TAGS:
        o, c = _count_open_close(text, tag)
        if o != 1 or c != 1:
            return Parsed(None, None, None, False, f"<{tag}> appears {o}/{c} times")
    spans = {t: _tag_spans(text, t) for t in TAGS}
    if any(len(v) != 1 for v in spans.values()):
        return Parsed(None, None, None, False, "malformed tag span")
    ans = _letter(spans["answer"][0], valid)
    if ans is None:
        return Parsed(None, None, None, False, f"answer {spans['answer'][0]!r} is not one of {valid}")
    ev = _evidence(spans["evidence"][0], n_studies)
    if ev is None:
        return Parsed(None, None, None, False, f"evidence {spans['evidence'][0]!r} is not a list of 1..{n_studies}")
    flag = spans["flag"][0].strip().lower()
    if flag not in FLAGS:
        return Parsed(None, None, None, False, f"flag {flag!r} not in {FLAGS}")
    return Parsed(ans, ev, flag, True)


_LENIENT_PATTERNS = [
    # keywords are case-insensitive; the option letter must be uppercase ("the answer is a ..." is not A)
    r"(?i:(?:final\s+)?answer)\s*(?i:is|:)?\s*(?i:option\s*)?\(?\*{0,2}([A-Z])\*{0,2}\)?(?![A-Za-z])",
    r"(?i:(?:correct|best)\s+(?:option|choice|answer))\s*(?i:is|:)\s*\(?\*{0,2}([A-Z])(?![A-Za-z])",
    r"^\s*\(?\*{0,2}([A-Z])\*{0,2}\)?\s*[.:)]?\s*$",   # a line that is only a letter
    r"^\s*\(?([A-Z])[.)]\s+\S",                       # "C. Between T2 and T3 ..." at a line start
]


def _letter_with_text(s: str, valid: str, options: dict | None) -> str | None:
    """'B. Improved' / '(B) Improved': accepted only if the text matches option B itself (2026-10-11).
    Without the option list, or if the text names another option, nothing is guessed."""
    m = re.fullmatch(r"\s*\(?\s*([A-Z])\s*[.):]\s*(.+?)\s*", s, flags=re.S)
    if not m or m.group(1) not in valid or not options:
        return None
    letter, text = m.group(1), _norm(m.group(2))
    return letter if _norm(options.get(letter, "")) == text else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip(".").lower()


def parse_lenient(text: str, n_options: int, n_studies: int, options: dict | None = None) -> Parsed:
    """Best-effort answer extraction for models never trained on our format. Ambiguity -> None."""
    valid = letters(n_options)
    strict = parse_strict(text, n_options, n_studies)
    if strict.format_ok:
        return strict

    body = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    # thinking models whose chat template opens <think> in the prompt emit only the closing tag
    if re.search(r"</think>", body, flags=re.I):
        body = re.split(r"</think>", body, flags=re.I)[-1]
    ans = None
    spans = _tag_spans(body, "answer")
    if len(spans) == 1:
        ans = _letter(spans[0], valid) or _letter_with_text(spans[0], valid, options)
    if ans is None:
        for pat in _LENIENT_PATTERNS:
            found = set(re.findall(pat, body, flags=re.M))
            found &= set(valid)
            if len(found) == 1:
                ans = found.pop()
                break
            if len(found) > 1:
                return Parsed(None, None, None, False, f"ambiguous letters {sorted(found)}")
    if ans is None and options:
        hits = [k for k, v in options.items() if _norm(v) and _norm(v) == _norm(body)]
        if len(hits) == 1 and hits[0] in valid:
            ans = hits[0]

    ev_spans = _tag_spans(body, "evidence")
    ev = _evidence(ev_spans[0], n_studies) if len(ev_spans) == 1 else None
    fl_spans = _tag_spans(body, "flag")
    flag = fl_spans[0].strip().lower() if len(fl_spans) == 1 and fl_spans[0].strip().lower() in FLAGS else None
    return Parsed(ans, ev, flag, False, "" if ans else "no answer found")
