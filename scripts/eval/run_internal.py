"""Exp 1 part C: models on the internal test set (bases + variants), zero-shot.

  python scripts/eval/run_internal.py --model qwen3-vl-8b --split test [--limit 30]
For every base of the split: v0-v4 under img_prior and img; v5 under img_prior (it rewrites a prior impression);
v6 under img_prior and img; v0 under text_only. Expected answers come from the variants table (v0-v4) and equal
the base answer for v5/v6. Every image is SHA-256-checked; the visual-token count is checked on the first items.
Output: <work_root>/eval/internal/<model>/<split>[_tag].parquet + provenance.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd

from icpo import config
from icpo.eval.internal import CONDITIONS, INSTRUCTION, INSTRUCTION_NATIVE, LETTERS, NO_IMAGES, ORACLE, render, render_exist
from icpo.eval.parser import parse_lenient
from icpo.provenance import add_cli, require_clean_tree, write_table

_spec = importlib.util.spec_from_file_location("run_micxr", Path(__file__).with_name("run_micxr.py"))
rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rm)


def build_requests(split: str, limit: int, conditions: list[str], instruction: str = INSTRUCTION):
    W = config.path("work_root")
    bases = pd.read_parquet(W / "pipeline" / "bases.parquet").query("split == @split").sort_values("qid")
    if limit:
        bases = bases.head(limit)
    var = pd.read_parquet(W / "pipeline" / "variants.parquet")
    var = var[var.qid.isin(set(bases.qid))]
    tv = pd.read_parquet(W / "text" / "text_variants.parquet")
    tv = {(q, v): r for q, v, r in zip(tv.qid, tv.variant, tv.to_dict("records"))}
    tc = pd.read_parquet(W / "pipeline" / "text_context.parquet").set_index("study_id")
    reqs = []
    for b in bases.to_dict("records"):
        win = [int(s) for s in b["study_ids"]]
        imps = {s: tc.prior_impression.get(s, "") or "" for s in win}
        ind = tc.indication.get(win[-1], "") or ""
        vb = {r["variant"]: r for r in var[var.qid == b["qid"]].to_dict("records")}
        v0 = vb["v0"]

        def add(vname, vrow, cond, indication, v5=None, expected=None):
            reqs.append({"qid": b["qid"], "family": b["family"], "finding": b["finding"], "variant": vname,
                         "condition": cond, "n_slots": len(vrow["slots"]), "options": list(vrow["options"]),
                         "expected": expected or vrow["answer"], "relation": vrow.get("relation", "SAME"),
                         "text_determined": bool(b["text_determined"]), "slot_relpaths": list(vrow["slot_relpaths"]),
                         "messages": render(b, vrow, cond, rm.config.path("images_root"), imps, indication, v5,
                                            instruction)})

        for cond in [c for c in ("img_prior", "img") if c in conditions]:
            for name in ("v0", "v1", "v2", "v3", "v4"):
                if name in vb:
                    add(name, vb[name], cond, ind)
            if (b["qid"], "v6") in tv:
                add("v6", {**v0, "relation": "SAME"}, cond, tv[(b["qid"], "v6")]["text"])
        if "img_prior" in conditions and (b["qid"], "v5") in tv:
            t = tv[(b["qid"], "v5")]
            add("v5", {**v0, "relation": "SAME"}, "img_prior", ind, v5={"slot": int(t["slot"]), "text": t["text"]})
        if "text_only" in conditions:
            add("v0", v0, "text_only", ind)
        for cond in [c for c in ORACLE if c in conditions]:      # sanity check: v0 with the current report shown
            add("v0", v0, cond, ind)
    return bases, reqs


def build_exist_requests(split: str, limit: int, instruction: str):
    """Perception ceiling: every single-film EXIST item of the split's sampled windows and findings, one per
    (study, finding); one image, no text."""
    W = config.path("work_root")
    bases = pd.read_parquet(W / "pipeline" / "bases.parquet").query("split == @split").sort_values("qid")
    if limit:
        bases = bases.head(limit)
    keys = set(zip(bases.window_id, bases.finding))
    pool = pd.read_parquet(W / "pipeline" / "questions_pool.parquet",
                           columns=["qid", "family", "subtype", "window_id", "finding", "answer", "study_ids", "dicom_ids"])
    in_keys = pd.Series([k in keys for k in zip(pool.window_id, pool.finding)], index=pool.index)
    ex = pool[(pool.family == "EXIST") & in_keys].copy()
    ex["k"] = [int(s[len("study"):]) for s in ex.subtype]
    ex["study"] = [int(ids[k]) for ids, k in zip(ex.study_ids, ex.k)]
    ex = ex.sort_values("qid").drop_duplicates(["study", "finding"])
    hashes = pd.read_parquet(W / "manifests" / "images_v1.hashes.parquet", columns=["dicom_id", "relpath"])
    rel = dict(zip(hashes.dicom_id, hashes.relpath))
    reqs = []
    for r in ex.itertuples(index=False):
        p = rel[r.dicom_ids[r.k]]
        msg, opts = render_exist(r.finding, p, config.path("images_root"), instruction)
        reqs.append({"qid": r.qid, "family": "EXIST", "finding": r.finding, "variant": "exist", "condition": "single",
                     "n_slots": 1, "options": opts, "expected": r.answer, "relation": "SAME", "text_determined": False,
                     "slot_relpaths": [p], "messages": msg})
    return bases, reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--conditions", default="img_prior+img+text_only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--check-tokens", type=int, default=20)
    ap.add_argument("--format", default="structured", choices=["structured", "native"])
    ap.add_argument("--task", default="bundles", choices=["bundles", "exist"])
    args = add_cli(ap).parse_args()
    git = require_clean_tree(args.allow_dirty or bool(args.limit))
    conditions = [c for c in args.conditions.split("+") if c]
    assert all(c in CONDITIONS for c in conditions), conditions

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    import vllm

    m = config.load("eval/models")[args.model]
    vis = config.load("vision")["model_input"]
    image_root = config.path("images_root")
    instruction = INSTRUCTION if args.format == "structured" else INSTRUCTION_NATIVE
    if args.task == "exist":
        bases, reqs = build_exist_requests(args.split, args.limit, instruction)
    else:
        bases, reqs = build_requests(args.split, args.limit, conditions, instruction)
    paths = sorted({p for r in reqs for p in r["slot_relpaths"]})
    n_img, image_set_hash = rm.verify_images([{"images": {str(i): p for i, p in enumerate(paths)}}], image_root)
    print(f"{len(bases)} bases, {len(reqs)} prompts, {n_img} images verified", flush=True)

    sub = m.get("subfolder")
    snapshot = snapshot_download(m["hf_id"], revision=m["revision"], cache_dir=rm.HF_CACHE,
                                 allow_patterns=[f"{sub}/*"] if sub else None)
    if sub:
        snapshot = str(Path(snapshot) / sub)
    if m.get("config_patch"):
        snapshot, patch = rm.patched_overlay(snapshot, m["config_patch"])
        m = {**m, "config_patch_applied": patch}
    family = m.get("family", "qwen")
    size = rm.processor_size(snapshot, vis, family)
    llm = LLM(model=snapshot, tokenizer=snapshot, seed=0, limit_mm_per_prompt={"image": 6},
              allowed_local_media_path=str(image_root), mm_processor_kwargs={"size": size} if size else None, **m["vllm"])
    max_tokens = max(args.max_tokens, m.get("max_tokens", 0))
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
    pad_id = llm.get_tokenizer().convert_tokens_to_ids("<|image_pad|>" if family == "qwen" else "<image_soft_token>")

    outs = []
    for s in range(0, len(reqs), args.chunk):
        outs += llm.chat([r["messages"] for r in reqs[s:s + args.chunk]], sp, use_tqdm=False,
                         chat_template_kwargs=m.get("chat_template_kwargs"))
        print(f"{min(s + args.chunk, len(reqs))}/{len(reqs)}", flush=True)

    rows = []
    for r, o in zip(reqs, outs):
        text = o.outputs[0].text
        opts = {LETTERS[i]: t for i, t in enumerate(r["options"])}
        p = parse_lenient(text, n_options=len(r["options"]), n_studies=r["n_slots"], options=opts)
        ans = opts.get(p.answer) if p.answer else None
        rows.append({k: r[k] for k in ("qid", "family", "finding", "variant", "condition", "n_slots", "expected",
                                       "relation", "text_determined")}
                    | {"options": r["options"], "answer_letter": p.answer, "answer": ans, "correct": ans == r["expected"],
                       "flag": p.flag, "evidence": sorted(p.evidence) if p.evidence is not None else None,
                       "format_ok": p.format_ok, "raw": text, "finish_reason": o.outputs[0].finish_reason,
                       "visual_tokens": int(sum(t == pad_id for t in o.prompt_token_ids))})
    if args.check_tokens:
        img_rows = [(r, x) for r, x in zip(reqs, rows) if r["condition"] not in NO_IMAGES][:args.check_tokens]
        for r, x in img_rows:
            exp = 256 * r["n_slots"] if size is None else sum(rm.expected_visual_tokens(
                snapshot, [image_root / p for p in r["slot_relpaths"]], size))
            if exp != x["visual_tokens"]:
                raise SystemExit(f"{r['qid']} {r['variant']}: {x['visual_tokens']} visual tokens, expected {exp}")
        if any(x["visual_tokens"] for r, x in zip(reqs, rows) if r["condition"] in NO_IMAGES):
            raise SystemExit("text-only prompts contain visual tokens")
        print("visual-token check passed", flush=True)

    df = pd.DataFrame(rows)
    stats = {"accuracy_by_condition_variant": df.groupby(["condition", "variant"]).correct.mean().round(4)
             .reset_index().to_dict("records"),
             "parse_failures": int(df.answer.isna().sum()), "truncated": int((df.finish_reason == "length").sum())}
    name = args.split + (f"_limit{args.limit}" if args.limit else "") + (f"_{args.tag}" if args.tag else "")
    W = config.path("work_root")
    write_table(df, W / "eval" / "internal" / args.model / f"{name}.parquet", step="exp1_internal", git=git,
                inputs=[W / "pipeline" / "bases.parquet", W / "pipeline" / "variants.parquet",
                        W / "text" / "text_variants.parquet", W / "pipeline" / "text_context.parquet"],
                params={"model": m, "split": args.split, "conditions": conditions, "vision": vis, "task": args.task,
                        "format": args.format,
                        "sampling": {"temperature": 0.0, "max_tokens": max_tokens, "seed": 0, "chunk": args.chunk},
                        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                        "image_set_sha256": image_set_hash, "n_images": n_img, "vllm": vllm.__version__,
                        "env": {k: os.environ.get(k) for k in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_BATCH_INVARIANT")}},
                stats=json.loads(json.dumps(stats, default=str)))
    print(json.dumps(stats, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
