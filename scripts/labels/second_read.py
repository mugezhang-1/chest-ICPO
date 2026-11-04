"""Second readers, step 2 (GPU): run one pinned reader over a request table, greedy, once; outputs are stored.

  python scripts/labels/second_read.py --reader gpt-oss-120b --requests <name> [--limit 20]
Output: <work_root>/second_read/<reader>/<name>.parquet (request_id, qid, raw, letter, agrees, finish_reason).
An unparsed or truncated reply counts as disagreement.
"""

import argparse
import hashlib
import os

import pandas as pd

from icpo import config
from icpo.data.second_read import PROMPT_VERSION, parse_answer
from icpo.provenance import add_cli, require_clean_tree, write_table


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--reader", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()
    git = require_clean_tree(args.allow_dirty or bool(args.limit))
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    import vllm

    m = config.load("labels/readers")[args.reader]
    W = config.path("work_root") / "second_read"
    req = pd.read_parquet(W / f"requests_{args.requests}.parquet")
    if args.limit:
        req = req.head(args.limit)
    # local_files_only: these snapshots were fetched outside this pipeline and are verified by
    # scripts/repro/verify_snapshot.py; huggingface_hub >= 1.x needs it to resolve a revision offline
    snap = snapshot_download(m["hf_id"], revision=m["revision"], cache_dir=m["cache_dir"], local_files_only=True)
    llm = LLM(model=snap, tokenizer=snap, seed=0, **m["vllm"])
    sp = SamplingParams(temperature=0.0, max_tokens=m["max_tokens"], seed=0)
    outs = []
    for s in range(0, len(req), args.chunk):
        convs = [[{"role": "user", "content": p}] for p in req.prompt.iloc[s:s + args.chunk]]
        outs += llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs=m.get("chat_template_kwargs"))
        print(f"{min(s + args.chunk, len(req))}/{len(req)}", flush=True)
    df = req[["request_id", "qid", "family", "finding", "n_options", "our_letter"]].copy()
    df["raw"] = [o.outputs[0].text for o in outs]
    df["finish_reason"] = [o.outputs[0].finish_reason for o in outs]
    df["n_tokens"] = [len(o.outputs[0].token_ids) for o in outs]
    df["letter"] = [None if f == "length" else parse_answer(t, n)
                    for t, f, n in zip(df.raw, df.finish_reason, df.n_options)]
    df["agrees"] = df.letter == df.our_letter
    name = args.requests + (f"_limit{args.limit}" if args.limit else "")
    stats = {"rows": len(df), "agree_rate": round(float(df.agrees.mean()), 4),
             "unparsed": int(df.letter.isna().sum()), "truncated": int((df.finish_reason == "length").sum()),
             "agree_by_family": df.groupby("family").agrees.mean().round(4).to_dict(),
             "tokens_mean": round(float(df.n_tokens.mean()), 1)}
    write_table(df, W / args.reader / f"{name}.parquet", step="second_read", git=git,
                inputs=[W / f"requests_{args.requests}.parquet"],
                params={"reader": m, "prompt_version": PROMPT_VERSION,
                        "sampling": {"temperature": 0.0, "max_tokens": m["max_tokens"], "seed": 0},
                        "prompts_sha256": hashlib.sha256("".join(req.prompt).encode()).hexdigest(),
                        "vllm": vllm.__version__, "env": {k: os.environ.get(k) for k in ("VLLM_USE_FLASHINFER_SAMPLER",)}},
                stats=stats)
    print(stats)


if __name__ == "__main__":
    main()
