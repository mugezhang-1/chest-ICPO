"""Re-parse stored raw outputs of an internal-set run with the current parser (no GPU).

The original run file is left untouched; the re-scored table is written next to it as <name>_rescored.parquet with
provenance recording the original's content hash and the parser commit.
"""

import argparse

import pandas as pd

from icpo import config
from icpo.eval.internal import LETTERS
from icpo.eval.parser import parse_lenient
from icpo.provenance import add_cli, require_clean_tree, write_table


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="test")
    args = ap.parse_args()
    git = require_clean_tree(args.allow_dirty)
    src = config.path("work_root") / "eval" / "internal" / args.model / f"{args.name}.parquet"
    d = pd.read_parquet(src)
    before = int(d.answer.isna().sum())
    new = []
    for r in d.itertuples(index=False):
        opts = {LETTERS[i]: t for i, t in enumerate(r.options)}
        p = parse_lenient(r.raw, n_options=len(r.options), n_studies=int(r.n_slots), options=opts)
        new.append((p.answer, opts.get(p.answer) if p.answer else None, p.flag,
                    sorted(p.evidence) if p.evidence is not None else None, p.format_ok))
    d[["answer_letter", "answer", "flag", "evidence", "format_ok"]] = pd.DataFrame(new, index=d.index)
    d["correct"] = d.answer == d.expected
    stats = {"parse_failures_before": before, "parse_failures_after": int(d.answer.isna().sum())}
    write_table(d, src.with_name(f"{args.name}_rescored.parquet"), step="exp1_internal_rescore", git=git,
                inputs=[src], params={"parser": "icpo.eval.parser.parse_lenient"}, stats=stats)
    print(args.model, stats)


if __name__ == "__main__":
    main()
