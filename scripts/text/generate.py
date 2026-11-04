"""Text step 2: generate every requested rewrite with a pinned model (greedy), once; outputs are stored and reused.

  python scripts/text/generate.py --model qwen3.5-9b-nothink [--limit 50]
Output: <work_root>/text/generated_<model>.parquet (request_id, raw, text) with provenance (model commit, prompts
hash, sampling, vLLM version).
"""

import argparse
import hashlib
import os

import pandas as pd

from icpo import config
from icpo.data.text_variants import clean_output
from icpo.provenance import add_cli, require_clean_tree, write_table

HF_CACHE = "/fs/scratch/PAS2836/mugezhang/hf_cache_large"


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--model", default="qwen3.5-9b-nothink")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--requests", default="requests", help="requests table name, e.g. requests_round2")
    args = ap.parse_args()
    git = require_clean_tree(args.allow_dirty or bool(args.limit))
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    import vllm

    m = config.load("eval/models")[args.model]
    W = config.path("work_root") / "text"
    req = pd.read_parquet(W / f"{args.requests}.parquet")
    if args.limit:
        req = req.sample(n=min(args.limit, len(req)), random_state=0).sort_values("request_id")
    snap = snapshot_download(m["hf_id"], revision=m["revision"], cache_dir=HF_CACHE)
    llm = LLM(model=snap, tokenizer=snap, seed=0, limit_mm_per_prompt={"image": 0}, **m["vllm"])
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    convs = [[{"role": "user", "content": p}] for p in req.prompt]
    outs = llm.chat(convs, sp, use_tqdm=True, chat_template_kwargs=m.get("chat_template_kwargs"))
    df = pd.DataFrame({"request_id": req.request_id.values, "raw": [o.outputs[0].text for o in outs],
                       "finish_reason": [o.outputs[0].finish_reason for o in outs]})
    df["text"] = df.raw.map(clean_output)
    sfx = args.requests[len("requests"):]                                # "" or "_round2"
    name = f"generated_{args.model}{sfx}" + (f"_limit{args.limit}" if args.limit else "")
    write_table(df, W / f"{name}.parquet", step="text_generate", git=git, inputs=[W / f"{args.requests}.parquet"],
                params={"model": m, "sampling": {"temperature": 0.0, "max_tokens": args.max_tokens, "seed": 0},
                        "prompts_sha256": hashlib.sha256("".join(req.prompt).encode()).hexdigest(),
                        "vllm": vllm.__version__, "env": {k: os.environ.get(k) for k in ("VLLM_USE_FLASHINFER_SAMPLER",)}},
                stats={"truncated": int((df.finish_reason == "length").sum()), "rows": len(df)})
    print(df.head(5).to_string()[:2000])


if __name__ == "__main__":
    main()
