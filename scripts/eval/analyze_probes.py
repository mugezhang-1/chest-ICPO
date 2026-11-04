"""Exp 1 part B: analyse MI-CXR probe runs for one model.

Label-free  : how often each probe changes the model's answer relative to `orig` (null probes give the baseline
              instability a structural probe must exceed to count as order sensitivity).
Gold-valid  : accuracy under options / reword / no_image (the gold letter cannot change).
Anchored    : TEL items on our seven findings whose full presence sequence is known from QBA labels and whose
              re-derived answer equals MI-CXR's gold letter on the original order. For these the correct answer
              after reverse / swap / constant is re-derived from the permuted sequence (icpo.eval.micxr_tel), so
              we can ask whether the model's answer moves the way the timeline requires.
Constant    : with five identical images nothing can appear or resolve, so every TEL answer should be the
              "none" option; this needs no labels at all.
Output: <work_root>/eval/micxr/<model>/probes_summary.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from icpo import config
from icpo.eval.micxr import load_items
from icpo.eval.micxr_tel import TO_OURS, derive, finding_of, pair_letter
from icpo.eval.probes import permuted_states, present

STRUCTURAL = ("reverse", "swap", "constant")
NULL = ("options", "reword")


def boot(x, rng, n=2000):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def stat(x, rng):
    x = list(x)
    return {"n": len(x), "rate": round(float(np.mean(x)), 4) if x else None, "ci95": boot(x, rng)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    d = config.path("work_root") / "eval" / "micxr" / args.model
    sfx = f"_{args.tag}" if args.tag else ""
    runs = {}
    for p in ("orig",) + STRUCTURAL + NULL + ("no_image",):
        f = d / f"native{'' if p == 'orig' else '_' + p}{sfx}.parquet"
        if f.exists():
            runs[p] = pd.read_parquet(f).set_index("q_id")
    orig = runs["orig"]
    items = {it["q_id"]: it for it in load_items(config.path("external_root") / "MI-CXR" / "micxr_test.jsonl")}

    out = {"model": args.model, "probes": sorted(runs), "orig_accuracy": stat(orig.correct, rng)}

    # 1. label-free answer changes
    out["answer_change_vs_orig"] = {}
    for p, r in runs.items():
        if p == "orig":
            continue
        ch = (r.answer.reindex(orig.index) != orig.answer)
        out["answer_change_vs_orig"][p] = {"all": stat(ch, rng),
                                           **{fam: stat(ch[orig.family == fam], rng) for fam in sorted(orig.family.unique())}}

    # 2. accuracy where the gold letter is unchanged
    out["accuracy"] = {p: stat(runs[p].correct, rng) for p in ("orig",) + NULL + ("no_image",) if p in runs}

    # 3. anchored TEL items (QBA presence sequence reproduces the gold answer on the original order)
    q = pd.read_parquet(config.path("work_root") / "labelers" / "qba.parquet").set_index("study_id")
    anchored = {}
    for qid, it in items.items():
        f = TO_OURS.get(finding_of(it["question"]) or "")
        if it["group_type"] != "TEL" or not f:
            continue
        sids = [int(it["images"][f"T{i}"].split("/")[3][1:]) for i in range(1, 6)]
        vals = [q[f].get(s) for s in sids]
        seq = ["pos" if v == 1.0 else "neg" if v == 0.0 else "unk" for v in vals]
        if "unk" in seq or derive(it["qtype"], seq, it["choices"]) != it["answer"]:
            continue
        anchored[qid] = seq
    out["anchored_tel_items"] = len(anchored)

    out["anchored"] = {}
    for p in STRUCTURAL:
        if p not in runs:
            continue
        rows = []
        for qid, seq in anchored.items():
            it = items[qid]
            exp = derive(it["qtype"], permuted_states(seq, present(it, p).order), it["choices"])
            if exp in (None, "ambiguous"):
                continue
            a0, a1 = orig.at[qid, "answer"], runs[p].at[qid, "answer"]
            rows.append({"qid": qid, "family": it["qtype"], "expected": exp, "gold": it["answer"],
                         "expected_changed": exp != it["answer"], "a0": a0, "a1": a1,
                         "correct0": a0 == it["answer"], "correct1": a1 == exp})
        r = pd.DataFrame(rows)
        moved = r[r.expected_changed]
        out["anchored"][p] = {
            "items": len(r),
            "expected_answer_changes": stat(r.expected_changed, rng),
            "accuracy_after": stat(r.correct1, rng),
            "accuracy_before_same_items": stat(r.correct0, rng),
            # the core order-sensitivity numbers, on items whose correct answer must move
            "when_answer_must_change": {
                "model_changed_answer": stat(moved.a1 != moved.a0, rng),
                "model_gave_new_correct_answer": stat(moved.correct1, rng),
                "model_kept_old_answer": stat(moved.a1 == moved.a0, rng),
                "still_correct_given_correct_before": stat(moved[moved.correct0].correct1, rng),
            },
        }

    # 4. constant sequence: every TEL answer must be the "none" option
    if "constant" in runs:
        c = runs["constant"]
        rows = []
        for qid, it in items.items():
            if it["group_type"] != "TEL":
                continue
            none = "E" if it["qtype"] not in ("emergence_to_resolution", "resolution_to_emergence") \
                else pair_letter(it["choices"], None)
            rows.append({"none_chosen": c.at[qid, "answer"] == none, "none_chosen_orig": orig.at[qid, "answer"] == none})
        r = pd.DataFrame(rows)
        out["constant_tel"] = {"chose_none": stat(r.none_chosen, rng), "chose_none_on_orig": stat(r.none_chosen_orig, rng)}

    dst = d / f"probes_summary{sfx}.json"
    dst.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
