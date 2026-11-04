"""Section parsing for MIMIC-CXR free-text reports (files/pXX/pXXXXXXXX/sXXXXXXXX.txt)."""

import re
import zipfile

# A section heading is an upper-case phrase at the start of a line followed by a colon, e.g. "IMPRESSION:".
HEADING = re.compile(r"^[ \t]*([A-Z][A-Z /&()-]{2,40}?)[ \t]*:", re.MULTILINE)
IMPRESSION_NAMES = {"IMPRESSION", "IMPRESSIONS", "CONCLUSION", "CONCLUSIONS", "SUMMARY"}
FINDINGS_NAMES = {"FINDINGS", "FINDING"}


def sections(text: str) -> list[tuple[str, str]]:
    """Ordered (HEADING, body) pairs. Text before the first heading is returned under ''."""
    out, last_name, last_end = [], "", 0
    for m in HEADING.finditer(text):
        out.append((last_name, text[last_end:m.start()]))
        last_name, last_end = m.group(1).strip(), m.end()
    out.append((last_name, text[last_end:]))
    return [(n, " ".join(b.split())) for n, b in out if b.strip()]


def labeling_text(text: str) -> tuple[str, str]:
    """Text used for labeling, following the MIMIC-CXR-JPG labels: impression, else findings, else the last
    section. Returns (source, text) with source in {"impression", "findings", "last", "empty"}."""
    secs = sections(text)
    for names, src in ((IMPRESSION_NAMES, "impression"), (FINDINGS_NAMES, "findings")):
        body = " ".join(b for n, b in secs if n in names)
        if body:
            return src, body
    return ("last", secs[-1][1]) if secs else ("empty", "")


def iter_reports(zip_path):
    """Yield (study_id, text) for every report in mimic-cxr-reports.zip."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if base.startswith("s") and base.endswith(".txt"):
                yield int(base[1:-4]), z.read(name).decode("utf-8", errors="replace")
