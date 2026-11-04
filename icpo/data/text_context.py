"""Pipeline step 4 (roadmap §3.2.4): text shown to the model alongside the images.

- indication: the current study's reason for exam, cut before the first sentence that reads like a finding or a
  comparison (some reports put the radiologist's reading inside the indication section, which would leak answers)
- impression: what a study contributes when it appears as a *prior* (impression, else findings)
Both are de-identification-cleaned: "___" placeholders, clock times and result-notification sentences removed.
The current study's own impression, findings and wet read are never used.
"""

import re

from icpo.data.reports import FINDINGS_NAMES, IMPRESSION_NAMES, sections

INDICATION_NAMES = {
    "INDICATION", "INDICATIONS", "HISTORY", "CLINICAL HISTORY", "REASON FOR EXAMINATION", "REASON FOR EXAM",
    "CLINICAL INFORMATION", "CLINICAL INDICATION",
}
# sentence boundary, but not after titles such as "Dr." (notification sentences name the physician)
SENTENCE = re.compile(r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<=[.!?])\s+")
DEID = re.compile(r"_{2,}")
CLOCK = re.compile(r"\b\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?", re.IGNORECASE)
NOTIFICATION = re.compile(
    r"\b(notified|discussed|telephone|paged|communicated|called|conveyed|informed|d/w)\b", re.IGNORECASE)
# sentences that report imaging findings or comparisons rather than the clinical question
FINDING_LIKE = re.compile(
    r"\b(compar\w*|there (is|are|has been|have been)|no (evidence|acute|new|change)|is seen|are seen|"
    r"unchanged|interval|again (seen|noted|demonstrated)|redemonstrat\w*|impression|findings?)\b",
    re.IGNORECASE)


def clean(text: str) -> str:
    """Remove de-identification placeholders, clock times and result-notification sentences."""
    keep = [s for s in SENTENCE.split(text) if not NOTIFICATION.search(s)]
    out = CLOCK.sub("", DEID.sub("", " ".join(keep)))
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"([,;:])\s*(?=[,.;:])", "", out)
    out = re.sub(r"^\W+", "", re.sub(r"\s+", " ", out)).strip()
    return out if re.search(r"[A-Za-z]{2,}", out) else ""


def indication(report: str) -> tuple[str, bool]:
    """(cleaned indication, whether it was truncated at a finding-like sentence)."""
    body = " ".join(b for n, b in sections(report) if n in INDICATION_NAMES)
    sents = [s for s in SENTENCE.split(body) if s.strip()]
    for i, s in enumerate(sents):
        if FINDING_LIKE.search(s):
            return clean(" ".join(sents[:i])), True
    return clean(" ".join(sents)), False


def prior_impression(report: str) -> tuple[str, str]:
    """(cleaned text, source in {"impression", "findings", "none"}) for use when this study is a prior."""
    secs = sections(report)
    for names, src in ((IMPRESSION_NAMES, "impression"), (FINDINGS_NAMES, "findings")):
        body = clean(" ".join(b for n, b in secs if n in names))
        if body:
            return body, src
    return "", "none"
