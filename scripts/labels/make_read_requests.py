"""Second readers, step 1 (CPU): build one reader prompt per item from the full reports of its window.

  python scripts/labels/make_read_requests.py --items <parquet> --name <name> [--split val] [--qids <csv with qid>]
Items need qid, finding, question, options, answer, study_ids. Output: <work_root>/second_read/requests_<name>.parquet
(request_id, qid, prompt, n_options, our_letter) with provenance. Contains report text: data directory only.
"""

import argparse
from pathlib import Path

import pandas as pd

from icpo import config
from icpo.data.mint import NO_LONGITUDINAL
from icpo.data.reports import iter_reports
from icpo.data.second_read import LETTERS, PROMPT_VERSION, build_prompt, reader_options, report_text, request_id
from icpo.provenance import add_cli, require_clean_tree, write_table


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--items", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--qids", default="", help="CSV with a qid column; keep only these items")
    args = ap.parse_args()
    git = require_clean_tree(args.allow_dirty)
    items = pd.read_parquet(args.items)
    items = items[(items.family != "EXIST") & ~items.finding.isin(NO_LONGITUDINAL)]   # v1 tables still hold them
    if args.split:
        items = items[items.split == args.split]
    if args.qids:
        items = items[items.qid.isin(set(pd.read_csv(args.qids).qid))]
    items = items.sort_values("qid").reset_index(drop=True)
    need = {int(s) for ids in items.study_ids for s in ids}
    rep = {}
    for sid, text in iter_reports(config.path("physionet_root") / "mimic-cxr" / "2.1.0" / "mimic-cxr-reports.zip"):
        if sid in need:
            rep[sid] = report_text(text)
    missing = need - set(rep)
    if missing:
        raise SystemExit(f"{len(missing)} reports missing, e.g. {sorted(missing)[:5]}")
    rows = []
    for it in items.to_dict("records"):
        opts = reader_options(list(it["options"]))
        rows.append({"request_id": request_id(it["qid"]), "qid": it["qid"], "family": it["family"],
                     "finding": it["finding"], "prompt": build_prompt(it, [rep[int(s)] for s in it["study_ids"]]),
                     "n_options": len(opts), "our_letter": LETTERS[list(it["options"]).index(it["answer"])]})
    df = pd.DataFrame(rows)
    out = config.path("work_root") / "second_read" / f"requests_{args.name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_table(df, out, step="second_read_requests", git=git, inputs=[Path(args.items)],
                params={"prompt_version": PROMPT_VERSION, "split": args.split, "qids": args.qids},
                stats={"items": len(df), "by_family": df.family.value_counts().to_dict(),
                       "prompt_chars_max": int(df.prompt.str.len().max())})
    print(len(df), "requests ->", out)


if __name__ == "__main__":
    main()
