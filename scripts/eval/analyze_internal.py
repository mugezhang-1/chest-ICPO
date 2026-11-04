"""Exp 1 part C metrics on the internal set (definitions fixed 2026-10-11, before any test-set result was seen).

Per condition (img_prior, img):
  accuracy            v0 accuracy, overall and per family
  gate1_icr_flip      ICR items: v1 (swapped interval, direction must reverse) answered with the reversed answer.
                      Roadmap Gate 1 stops the project if untrained models exceed ~0.80 here.
  tel_onset_shift     TEL items whose v1 / v2 answer must change: variant answered correctly
  null_flip           v3 (duplicate) and v6 (reworded indication): answer differs from the model's own v0 answer (ideal 0)
  abstain_v4          v4 (another patient's film): answered "Cannot be assessed"; flag_v4: flagged inconsistent
  false_abstain_v0    v0 answered "Cannot be assessed" (ideal 0)
  text_follow_v5      img_prior only, v5 (a prior impression flipped): answer differs from the model's v0 answer (ideal 0)
  consistent_given_correct  P(variant correct | v0 correct), per variant: the reward's consistency term
  citation            v0 with an <evidence> list: share citing every required study
Text-only baseline: v0 accuracy without images, overall and split by the step-6 text_determined flag.
All rates come with 95% bootstrap intervals over bases.
"""

import argparse
import json

import numpy as np
import pandas as pd

from icpo import config

CANNOT = "Cannot be assessed from these images"


def ci(x, rng, n=2000):
    x = np.asarray(list(x), dtype=float)
    if len(x) == 0:
        return {"n": 0, "rate": None, "ci95": None}
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"n": int(len(x)), "rate": round(float(x.mean()), 4),
            "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    W = config.path("work_root")
    name = args.split + (f"_{args.tag}" if args.tag else "")   # e.g. --tag rescored
    d = pd.read_parquet(W / "eval" / "internal" / args.model / f"{name}.parquet")
    req = pd.read_parquet(W / "pipeline" / "bases.parquet", columns=["qid", "required_evidence"]).set_index("qid")
    out = {"model": args.model, "split": args.split, "parse_failures": int(d.answer.isna().sum()),
           "truncated": int((d.finish_reason == "length").sum()), "conditions": {}}

    for cond in ("img_prior", "img"):
        c = d[d.condition == cond]
        if c.empty or "v0" not in set(c.variant):
            continue
        piv = c.pivot_table(index="qid", columns="variant", values="answer", aggfunc="first")
        cor = c.pivot_table(index="qid", columns="variant", values="correct", aggfunc="first")
        exp = c.pivot_table(index="qid", columns="variant", values="expected", aggfunc="first")
        fam = c.drop_duplicates("qid").set_index("qid").family
        v0 = c[c.variant == "v0"].set_index("qid")
        r = {"accuracy": {"all": ci(v0.correct, rng),
                          **{f: ci(g.correct, rng) for f, g in v0.groupby("family")}}}
        icr = [q for q in cor.index if fam[q] == "ICR" and pd.notna(exp.at[q, "v1"] if "v1" in exp else np.nan)]
        r["gate1_icr_flip"] = ci((cor.at[q, "v1"] for q in icr), rng)
        r["gate1_icr_flip_given_v0_correct"] = ci((cor.at[q, "v1"] for q in icr if cor.at[q, "v0"]), rng)
        for v in ("v1", "v2"):
            if v in exp:
                q = [q for q in exp.index if fam[q] == "TEL" and pd.notna(exp.at[q, v]) and exp.at[q, v] != exp.at[q, "v0"]]
                r[f"tel_onset_shift_{v}"] = ci((cor.at[x, v] for x in q), rng)
        # answer changes are counted only where both answers parsed (two unparsed answers are not a "change")
        both = lambda v: piv.index[piv[v].notna() & piv["v0"].notna()]
        for v in ("v3", "v6"):
            if v in piv:
                r[f"null_flip_{v}"] = ci((piv.at[x, v] != piv.at[x, "v0"] for x in both(v)), rng)
        if "v4" in piv:
            v4 = c[c.variant == "v4"]
            r["abstain_v4"] = ci(v4.answer == CANNOT, rng)
            r["flag_v4"] = ci(v4.flag == "inconsistent", rng)
        r["false_abstain_v0"] = ci(v0.answer == CANNOT, rng)
        if "v5" in piv and cond == "img_prior":
            r["text_follow_v5"] = ci((piv.at[x, "v5"] != piv.at[x, "v0"] for x in both("v5")), rng)
        r["consistent_given_correct"] = {
            v: ci((cor.at[q, v] for q in cor.index if cor.at[q, "v0"] is True and pd.notna(cor.at[q, v])), rng)
            for v in ("v1", "v2", "v3", "v4", "v5", "v6") if v in cor}
        ev = v0[v0.evidence.notna()]
        r["citation_present"] = ci(v0.evidence.notna(), rng)
        r["citation_covers_required"] = ci((set(int(i) for i in req.at[q, "required_evidence"]) <= set(e)
                                            for q, e in zip(ev.index, ev.evidence)), rng)
        r["parse_failure_rate"] = ci(c.answer.isna(), rng)
        out["conditions"][cond] = r

    # sanity checks (2026-10-11): oracle conditions show the current report; "single" is one film, no text
    for cond in ("img_oracle", "text_oracle", "single"):
        c = d[d.condition == cond]
        if c.empty:
            continue
        r = {"all": ci(c.correct, rng), **{f"family_{f}": ci(g.correct, rng) for f, g in c.groupby("family")},
             **{f"finding_{f}": ci(g.correct, rng) for f, g in c.groupby("finding")},
             "parse_failure_rate": ci(c.answer.isna(), rng)}
        if cond == "single":
            r["recall_present"] = ci(c[c.expected == "Yes"].correct, rng)
            r["recall_absent"] = ci(c[c.expected == "No"].correct, rng)
            r["balanced_accuracy"] = round((r["recall_present"]["rate"] + r["recall_absent"]["rate"]) / 2, 4)
        out.setdefault("sanity", {})[cond] = r

    t = d[(d.condition == "text_only") & (d.variant == "v0")]
    if not t.empty:
        out["text_only"] = {"all": ci(t.correct, rng),
                            "text_determined": ci(t[t.text_determined].correct, rng),
                            "not_text_determined": ci(t[~t.text_determined].correct, rng),
                            **{f: ci(g.correct, rng) for f, g in t.groupby("family")}}
    dst = W / "eval" / "internal" / args.model / f"{name}_summary.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
