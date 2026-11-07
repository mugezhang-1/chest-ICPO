"""Exp 1, part A: MI-CXR accuracy under the paper's protocol (reconstructed; see icpo/eval/micxr.py).

  python scripts/eval/run_micxr.py --model qwen3-vl-8b [--limit 50] [--check-tokens 20]

Before inference every MI-CXR image is checked against the SHA-256 recorded when it was resized, so the exact
pixels used are known. --check-tokens N compares, for the first N items, the number of visual tokens vLLM put in
the prompt with the number the model's own Hugging Face processor gives at the budget in configs/vision.yaml;
any mismatch aborts the run. Output: <work_root>/eval/micxr/<model>/<protocol>.parquet + provenance + summary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from icpo import config
from icpo.eval.micxr import NATIVE_INSTRUCTION, NATIVE_INTRO, PAPER_TABLE1, FAMILY, load_items, native_messages, summarize
from icpo.eval.parser import parse_lenient
from icpo.eval.probes import PROBES, REWORD_INSTRUCTION, REWORD_INTRO, present, presented_choices
from icpo.provenance import add_cli, require_clean_tree, write_table

# HF_CACHE = "/fs/scratch/PAS2836/mugezhang/hf_cache_large"
HF_CACHE = "/fs/scratch/PAS2836/yusenpeng_cache"

def verify_images(items, image_root):
    rec = pd.read_parquet(config.path("work_root") / "manifests" / "images_v1.resized.parquet")
    want = dict(zip(rec.relpath, rec.sha256))
    paths = sorted({p for it in items for p in it["images"].values()})
    bad = [p for p in paths if p not in want or not (image_root / p).exists()]
    if bad:
        raise SystemExit(f"{len(bad)} MI-CXR images missing from the resized set, e.g. {bad[:3]}")
    h = hashlib.sha256()
    for p in paths:
        digest = hashlib.sha256((image_root / p).read_bytes()).hexdigest()
        if digest != want[p]:
            raise SystemExit(f"{p}: file on disk differs from the resize record")
        h.update(f"{p}:{digest}\n".encode())
    return len(paths), h.hexdigest()


def patched_overlay(snapshot, patch):
    """A directory that mirrors `snapshot` with symlinks, except config.json, which gets `patch` applied.

    Used only for config fields that newer transformers reject but that do not affect vLLM inference. The original
    snapshot is never modified; both config checksums are returned for the provenance record.
    """
    src = Path(snapshot)
    cfg_text = (src / "config.json").read_text()
    cfg = json.loads(cfg_text)
    cfg.update(patch)
    new_text = json.dumps(cfg, indent=2, sort_keys=True)
    tag = hashlib.sha256((str(src.resolve()) + new_text).encode()).hexdigest()[:12]
    dst = Path(HF_CACHE) / "overlays" / tag
    if not (dst / "config.json").exists():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.name != "config.json" and not (dst / f.name).exists():
                (dst / f.name).symlink_to(f.resolve())
        (dst / "config.json").write_text(new_text)
    return str(dst), {"patch": patch, "original_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
                      "patched_sha256": hashlib.sha256(new_text.encode()).hexdigest()}


def processor_size(model_path, vis, family):
    """Per-model pixel budget from the token budget in configs/vision.yaml (None: fixed-resolution encoder)."""
    if family != "qwen":
        return None
    pp = json.load(open(Path(model_path) / "preprocessor_config.json"))
    px = (pp["patch_size"] * pp["merge_size"]) ** 2
    return {"shortest_edge": vis["min_tokens_per_image"] * px, "longest_edge": vis["tokens_per_image"] * px}


def expected_visual_tokens(snapshot, image_paths, size):
    from PIL import Image
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(snapshot)
    out = []
    for p in image_paths:
        with Image.open(p) as im:
            g = proc(images=[im.convert("RGB")], size=size, return_tensors="np")["image_grid_thw"][0]
        out.append(int(g[0] * g[1] * g[2]) // (proc.merge_size ** 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", default="native", choices=["native"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-tokens", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tag", default="", help="suffix for the output name, e.g. run2 for a repeat run")
    ap.add_argument("--chunk", type=int, default=256, help="items per vLLM call (recorded; keep fixed across runs)")
    ap.add_argument("--probes", default="orig", help=f"subset of {PROBES} joined by + (sbatch --export splits on commas); one model load")
    args = add_cli(ap).parse_args()
    git = require_clean_tree(args.allow_dirty or bool(args.limit))

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    import vllm

    m = config.load("eval/models")[args.model]
    vis = config.load("vision")["model_input"]
    image_root = config.path("images_root")
    items_path = config.path("external_root") / "MI-CXR" / "micxr_test.jsonl"
    items = load_items(items_path)
    if args.limit:
        items = items[:args.limit]
    n_img, image_set_hash = verify_images(items, image_root)
    print(f"{len(items)} items, {n_img} images verified", flush=True)

    sub = m.get("subfolder")
    snapshot = snapshot_download(m["hf_id"], revision=m["revision"], cache_dir=HF_CACHE,
                                 allow_patterns=[f"{sub}/*"] if sub else None)
    if sub:
        snapshot = str(Path(snapshot) / sub)
    config_patch = None
    if m.get("config_patch"):
        snapshot, config_patch = patched_overlay(snapshot, m["config_patch"])
        m = {**m, "config_patch_applied": config_patch}     # recorded in every output's provenance
    family = m.get("family", "qwen")
    size = processor_size(snapshot, vis, family)
    print(f"model {args.model}: family {family}, image size kwargs {size}", flush=True)
    llm = LLM(model=snapshot, tokenizer=snapshot, seed=0, limit_mm_per_prompt={"image": 5},
              allowed_local_media_path=str(image_root),
              mm_processor_kwargs={"size": size} if size else None, **m["vllm"])
    args.max_tokens = m.get("max_tokens", args.max_tokens)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    tok = llm.get_tokenizer()
    pad_id = tok.convert_tokens_to_ids("<|image_pad|>" if family == "qwen" else "<image_soft_token>")
    probes = [p for p in args.probes.replace(",", "+").split("+") if p]
    assert all(p in PROBES for p in probes), probes
    for probe in probes:
        run_probe(args, probe, items, llm, sp, pad_id, snapshot, size, image_root, m, vis, git, items_path,
                  n_img, image_set_hash, vllm.__version__)


def run_probe(args, probe, items, llm, sp, pad_id, snapshot, size, image_root, m, vis, git, items_path, n_img,
              image_set_hash, vllm_version):
    pres = [present(it, probe) for it in items]
    # fixed-size chunks: one call with every item decodes all 26k images into host memory at once
    outs = []
    for start in range(0, len(items), args.chunk):
        convs = []
        for it, p in zip(items[start:start + args.chunk], pres[start:start + args.chunk]):
            convs.append(native_messages(it, image_root, p.order, choices=presented_choices(it, p),
                                         intro=REWORD_INTRO if p.reword else NATIVE_INTRO,
                                         instruction=REWORD_INSTRUCTION if p.reword else NATIVE_INSTRUCTION,
                                         show_images=p.show_images))
        outs += llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs=m.get("chat_template_kwargs"))
        print(f"[{probe}] {min(start + args.chunk, len(items))}/{len(items)}", flush=True)

    # the gold letter is unchanged by null and modality probes; structural probes are scored in the analysis
    gold_valid = probe in ("orig", "options", "reword", "no_image")
    rows = []
    for it, p, o in zip(items, pres, outs):
        text = o.outputs[0].text
        parsed = parse_lenient(text, n_options=5, n_studies=5, options=presented_choices(it, p))
        ans = p.letters.get(parsed.answer) if parsed.answer else None     # back to the original letter
        rows.append({
            "q_id": it["q_id"], "qtype": it["qtype"], "family": FAMILY[it["qtype"]], "probe": probe,
            "order": "".join(x[1] for x in p.order), "gold": it["answer"], "raw": text,
            "answer_presented": parsed.answer, "answer": ans,
            "correct": (ans == it["answer"]) if gold_valid else None, "parse_error": parsed.error,
            "finish_reason": o.outputs[0].finish_reason, "prompt_tokens": len(o.prompt_token_ids),
            "visual_tokens": int(sum(t == pad_id for t in o.prompt_token_ids)),
        })

    if args.check_tokens and probe == "orig":
        for it, r in list(zip(items, rows))[:args.check_tokens]:
            exp = 256 * 5 if size is None else sum(expected_visual_tokens(
                snapshot, [image_root / it["images"][t] for t in sorted(it["images"])], size))
            if exp != r["visual_tokens"]:
                raise SystemExit(f"q_id {it['q_id']}: vLLM used {r['visual_tokens']} visual tokens, "
                                 f"the HF processor gives {exp} at the configured budget")
        print(f"visual-token check passed on {args.check_tokens} items", flush=True)
    if probe == "no_image" and any(r["visual_tokens"] for r in rows):
        raise SystemExit("no_image probe still contains visual tokens")

    df = pd.DataFrame(rows)
    summary = summarize([r for r in rows if r["correct"] is not None]) if gold_valid else {}
    summary["parse_failures"] = int(df.answer.isna().sum())
    summary["truncated"] = int((df.finish_reason == "length").sum())
    summary["answer_distribution"] = df.answer.fillna("none").value_counts().sort_index().to_dict()
    summary["paper_table1"] = PAPER_TABLE1.get(m.get("paper_name"))
    name = args.protocol + ("" if probe == "orig" else f"_{probe}") + (f"_limit{args.limit}" if args.limit else "") \
        + (f"_{args.tag}" if args.tag else "")
    out = config.path("work_root") / "eval" / "micxr" / args.model / f"{name}.parquet"
    prompt_text = NATIVE_INTRO + "\n" + NATIVE_INSTRUCTION
    write_table(df, out, step="exp1_micxr", git=git, stats=summary, inputs=[items_path],
                params={"model": m, "protocol": args.protocol, "probe": probe, "vision": vis, "sampling": {
                            "temperature": 0.0, "max_tokens": args.max_tokens, "seed": 0, "chunk": args.chunk},
                        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                        "image_set_sha256": image_set_hash, "n_images": n_img, "vllm": vllm_version,
                        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "VLLM_USE_FLASHINFER_SAMPLER",
                                                                "VLLM_BATCH_INVARIANT", "HF_HUB_OFFLINE")}})
    print(json.dumps({k: summary.get(k) for k in ("overall", "parse_failures", "answer_distribution")}, default=str))


if __name__ == "__main__":
    main()
