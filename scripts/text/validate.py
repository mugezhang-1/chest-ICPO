"""Text step 3: validate every generated rewrite; only passing rewrites become v5 / v6 variants.

CheXbert (the project's labeler, same pinned weights and code as scripts/labelers/run_chexbert.py) labels the original
and the rewrite:
  v6: no CheXbert label changes its EXPLICIT status (present/absent), plus the rule checks
  v5: the target finding becomes explicit in the requested direction; no other label changes its explicit status
      (the target's CheXpert-hierarchy relatives excepted), plus the rule checks
Outputs in <work_root>/text/: validated.parquet (request_id, kind, text, passed, failures) and
text_variants.parquet (qid, variant v5/v6, slot, text) for passing rewrites. Both with provenance.
"""

import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer

from icpo import config
from icpo.data.text_variants import rule_checks
from icpo.provenance import add_cli, require_clean_tree, write_table

_spec = importlib.util.spec_from_file_location("run_chexbert", config.REPO_ROOT / "scripts" / "labelers" / "run_chexbert.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def chexbert_labels(texts, model, tok, device):
    preds = cb.label(model, cb.encode(list(texts), tok), device, 64)
    return np.vectorize(lambda p: cb.DECODE[int(p)])(preds)          # n x 14, values 1/0/-1/nan


def explicit(x):
    return "pos" if x == 1.0 else "neg" if x == 0.0 else None


def same(a, b):
    """No EXPLICIT status changes: present/absent must match; uncertain vs not-mentioned is labeler noise on
    rephrased text and is ignored (decided 2026-10-08 after reading the smoke-test failures)."""
    return all(explicit(x) == explicit(y) for x, y in zip(a, b))


# CheXpert hierarchy: flipping a finding legitimately moves its parent / children
RELATIVES = {
    "Cardiomegaly": {"Enlarged Cardiomediastinum"},
    "Lung Opacity": {"Edema", "Consolidation", "Pneumonia", "Atelectasis", "Lung Lesion"},
    "Edema": {"Lung Opacity"}, "Consolidation": {"Lung Opacity", "Pneumonia"}, "Atelectasis": {"Lung Opacity"},
    "Pleural Effusion": {"Pleural Other"}, "Pneumothorax": set(), "Support Devices": set(),
}


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--model", default="qwen3.5-9b-nothink")
    ap.add_argument("--round", default="", help='"" for round 1, "_round2" for the retry round')
    args = ap.parse_args()
    git = require_clean_tree(args.allow_dirty)
    W = config.path("work_root") / "text"
    req = pd.read_parquet(W / f"requests{args.round}.parquet")
    gen = pd.read_parquet(W / f"generated_{args.model}{args.round}.parquet")
    df = req.merge(gen[["request_id", "text", "finish_reason"]], on="request_id", how="inner", validate="one_to_one")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizer.from_pretrained("bert-base-uncased", revision=cb.TOKENIZER_REVISION)
    model = cb.load_model(device)
    lab_o = chexbert_labels(df.source_text, model, tok, device)
    lab_n = chexbert_labels(df.text, model, tok, device)
    idx = {c: i for i, c in enumerate(cb.CONDITIONS)}

    fails = []
    for k, r in enumerate(df.itertuples(index=False)):
        f = rule_checks(r.kind, r.source_text, r.text, r.finding)
        if r.finish_reason == "length":
            f.append("truncated")
        if r.kind == "v6":
            if not same(lab_o[k], lab_n[k]):
                f.append("chexbert_changed")
        else:
            t = idx[r.finding]
            want = 0.0 if r.target == "neg" else 1.0
            if lab_n[k][t] != want:
                f.append("target_not_flipped")
            skip = {r.finding, "No Finding"} | RELATIVES[r.finding]
            others = [i for i, c in enumerate(cb.CONDITIONS) if c not in skip]
            if not same(lab_o[k][others], lab_n[k][others]):
                f.append("other_findings_changed")
        fails.append(f)
    df["failures"] = fails
    df["passed"] = [not f for f in fails]
    val = df[["request_id", "kind", "study_id", "finding", "target", "source_text", "text", "passed", "failures"]]

    ins = [W / f"requests{args.round}.parquet", W / f"generated_{args.model}{args.round}.parquet"]
    vstats = {"pass_rate_by_kind": val.groupby("kind").passed.mean().round(4).to_dict(),
              "failures_by_kind": {k: pd.Series([x for fs in g.failures for x in fs]).value_counts().to_dict()
                                   for k, g in val.groupby("kind")}}
    vstats = json.loads(json.dumps(vstats, default=int))
    write_table(val, W / f"validated{args.round}.parquet", step="text_validate", git=git, inputs=ins, stats=vstats,
                params={"labeler": "CheXbert (scripts/labelers/run_chexbert.py)", "generator": args.model})

    # final table from every validated round: per (base, kind) the passing rewrite with the lowest priority
    rounds = [("", 0)] + ([("_round2", None)] if (W / "links_round2.parquet").exists() and
                          (W / "validated_round2.parquet").exists() else [])
    allv, alll = [], []
    for sfx, prio in rounds:
        allv.append(pd.read_parquet(W / f"validated{sfx}.parquet"))
        lk = pd.read_parquet(W / f"links{sfx}.parquet")
        alll.append(lk.assign(priority=prio) if prio is not None else lk)
    allv = pd.concat(allv).drop_duplicates("request_id")
    alll = pd.concat(alll, ignore_index=True)
    ok = allv[allv.passed][["request_id", "text"]]
    tv = (alll.merge(ok, on="request_id").sort_values(["qid", "kind", "priority"])
          .drop_duplicates(["qid", "kind"]).rename(columns={"kind": "variant"}).reset_index(drop=True))
    stats = {**vstats, "rounds": [r for r, _ in rounds], "bases_with_text_variant": tv.groupby("variant").qid.nunique().to_dict(),
             "chosen_priority": {f"{v}/p{p}": int(n) for (v, p), n in tv.groupby(["variant", "priority"]).size().items()}}
    write_table(tv, W / "text_variants.parquet", step="text_variants", git=git, stats=stats,
                inputs=[W / f"validated{r}.parquet" for r, _ in rounds] + [W / f"links{r}.parquet" for r, _ in rounds],
                params={"generator": args.model})
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
