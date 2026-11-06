"""
High-level design choices: we want to do rejection SFT on qwen3.5-9b-think (thinking mode)

**1.0**:
train_predictions_without_answer.jsonl - inference on qwen3.5-9b-think, the model is not exposed to the correct answer

    **1.1**:
    train_predictions_without_answer_correct_only.jsonl - filter on correct predictions only

    **1.2**:
    train_predictions_without_answer_correct_only_img_prior.jsonl - filter on samples with available reports


**2.0**:
train_predictions_with_answer.jsonl - inference on qwen3.5-9b-think, the model is exposed to the correct answer during generation

    **2.1**:
    train_predictions_with_answer_correct_only.jsonl - filter on correct predictions only

    **2.2**:
    train_predictions_with_answer_correct_only_img_prior.jsonl - filter on samples with available reports
    
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
import vllm

from vllm import LLM, SamplingParams
from tqdm import tqdm


from icpo import config
from icpo.eval.internal import INSTRUCTION, INSTRUCTION_NATIVE, LETTERS, NO_IMAGES
from icpo.eval.parser import parse_lenient
from icpo.provenance import add_cli, require_clean_tree, write_table
from scripts.eval.run_micxr import HF_CACHE, verify_images, processor_size, expected_visual_tokens, patched_overlay

SUPPORTED_CONDITIONS = ("img_prior", "img", "text_only")

def optional_text(value):
    """JSONL occasionally encodes missing indications as NaN (nonstandard JSON)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Expected optional text, got {type(value).__name__}: {value!r}")
    return value

def load_records(path: Path, split: str, limit: int) -> tuple[list[dict], int]:
    records = []
    seen = set()
    for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
        if r.get("split") != split:
            continue
        required = (
            "bundle_id", "question_id", "family", "finding", "variant", "relation",
            "question", "options", "answer", "answer_letter", "images",
            "prior_impressions", "indication", "conditions",
        )
        missing = [k for k in required if k not in r]
        if missing:
            raise ValueError(f"{path}:{lineno}: missing fields {missing}")
        if r["question_id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate question_id {r['question_id']}")
        seen.add(r["question_id"])
        if not isinstance(r["images"], list) or not 1 <= len(r["images"]) <= 6:
            raise ValueError(f"{r['question_id']}: expected 1 to 6 images")
        if not isinstance(r["prior_impressions"], list) or len(r["prior_impressions"]) != len(r["images"]):
            raise ValueError(f"{r['question_id']}: prior_impressions must align with images")
        if "slot_states" in r and len(r["slot_states"]) != len(r["images"]):
            raise ValueError(f"{r['question_id']}: slot_states must align with images")
        if not isinstance(r["options"], list) or not 1 <= len(r["options"]) <= len(LETTERS):
            raise ValueError(f"{r['question_id']}: invalid options")
        expected_letter = LETTERS[r["options"].index(r["answer"])] if r["answer"] in r["options"] else None
        if expected_letter != r["answer_letter"]:
            raise ValueError(f"{r['question_id']}: answer/answer_letter/options disagree")
        if not set(r["conditions"]).issubset(SUPPORTED_CONDITIONS):
            raise ValueError(f"{r['question_id']}: unsupported conditions {r['conditions']}")
        r["indication"] = optional_text(r["indication"])
        r["prior_impressions"] = [optional_text(x) for x in r["prior_impressions"]]
        records.append(r)

    if limit:
        # --limit applies to BASE QUESTIONS (bundles), not JSONL lines.
        # Preserve first-seen order and include every variant in selected bundles.
        chosen = set(list(dict.fromkeys(r["bundle_id"] for r in records))[:limit])
        records = [r for r in records if r["bundle_id"] in chosen]
    n_bundles = len(set(r["bundle_id"] for r in records))
    if not records:
        raise ValueError(f"No records in {path} for split={split!r}")
    return records, n_bundles

def select_image_root(image_root: Path, records: list[dict]) -> tuple[Path, bool]:
    """Accept either dataset root or its images/ directory for images/files/... paths."""
    first = records[0]["images"][0]
    root = image_root.expanduser().resolve()
    p = Path(first)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe JSONL image path: {first}")
    candidates = [(root, False)]
    if p.parts[0] == "images":
        candidates.append((root, True))
    for candidate_root, strip in candidates:
        test_rel = Path(*p.parts[1:]) if strip else p
        if (candidate_root / test_rel).is_file():
            return candidate_root, strip
    raise FileNotFoundError(
        f"Cannot find first image {first!r} under {root}. "
        "--image-root should point to the dataset root (containing images/) "
        "or the images directory itself."
    )


def relative_image_path(raw: str, strip_images_prefix: bool) -> str:
    p = Path(raw)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe JSONL image path: {raw}")
    if strip_images_prefix:
        if not p.parts or p.parts[0] != "images":
            raise ValueError(f"Expected images/ prefix: {raw}")
        p = Path(*p.parts[1:])
    return p.as_posix()

def render_jsonl(r: dict, cond: str, image_root: Path, instruction: str, answer_mode: str = "without") -> list[dict]:
    """Same text layout as icpo.eval.internal.render, with slot-aligned impressions."""
    m = len(r["images"])
    show_img = cond not in NO_IMAGES
    show_imp = cond in ("img_prior", "text_only")
    lead = (
        f"The following {m} chest X-ray images are from the same patient, labeled T1 to T{m} in chronological order."
        if show_img else
        f"A patient had {m} chest X-ray examinations, labeled T1 to T{m} in chronological order. "
        "The images are not available; only the text below is."
    )
    ind = r["indication"]
    content = [{"type": "text", "text": lead + (f"\nClinical indication: {ind}" if ind else "")}]
    for k, (rel, impression) in enumerate(zip(r["image_relpaths"], r["prior_impressions"])):
        if show_img:
            content.append({"type": "text", "text": f"\nT{k + 1}:"})
            content.append({"type": "image_url", "image_url": {"url": f"file://{image_root / rel}"}})
        if show_imp and impression:
            content.append({"type": "text", "text": f"\nReport impression for T{k + 1}: {impression}"})
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))

    prompt = f"\n\nQuestion: {r['question']}\n{opts}\n\n"
    if answer_mode == "with":
        prompt += (
            f"Ground-truth answer: {r['answer_letter']}. {r['answer']}\n"
            "The ground-truth answer is provided as supervision. "
            "Explain the answer using the available clinical evidence, "
            "then provide the final answer in the required format.\n\n"
        )

    prompt += instruction
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_requests(records: list[dict], conditions: list[str], image_root: Path, instruction: str, answer_mode: str = "without"):
    reqs = []
    for r in records:
        for cond in r["conditions"]:
            if cond not in conditions:
                continue
            reqs.append({
                "qid": r["bundle_id"],
                "question_id": r["question_id"],
                "family": r["family"],
                "finding": r["finding"],
                "variant": r["variant"],
                "condition": cond,
                "n_slots": len(r["images"]),
                "options": r["options"],
                "expected": r["answer"],
                "relation": r["relation"],
                "slot_relpaths": r["image_relpaths"],
                "messages": render_jsonl(r, cond, image_root, instruction, answer_mode),
                "has_prior_report": any(impression.strip() for impression in r["prior_impressions"]),
            })
    if not reqs:
        raise ValueError(f"No requests for conditions={conditions}")
    return reqs

def verify_images(reqs: list[dict], image_root: Path) -> tuple[int, str]:
    """Verify files and fingerprint their bytes (no original manifest to compare against)."""
    paths = sorted({p for r in reqs for p in r["slot_relpaths"]})
    h = hashlib.sha256()
    for idx, rel in enumerate(paths, 1):
        path = image_root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        h.update(f"{rel}\t{digest}\n".encode("utf-8"))
        if idx % 1000 == 0:
            print(f"Verified {idx}/{len(paths)} unique images", flush=True)
    return len(paths), h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, default=None,
                    help="Dataset root containing images/, or the images/ directory. Default: config images_root.")
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--conditions", default="img_prior+img+text_only")
    ap.add_argument("--limit", type=int, default=0, help="Number of unique QA bundles, with all variants")
    ap.add_argument("--tag", default="jsonl")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--check-tokens", type=int, default=20)
    ap.add_argument("--format", default="structured", choices=["structured", "native"])

    ap.add_argument(
        "--answer-mode",
        choices=["without", "with"],
        default="without",
        help="Whether to expose the ground-truth answer during generation.",
    )

    args = add_cli(ap).parse_args()
    if args.chunk < 1 or args.limit < 0 or args.check_tokens < 0:
        ap.error("--chunk must be positive; --limit and --check-tokens cannot be negative")
    conditions = [x for x in args.conditions.split("+") if x]
    if not conditions or not set(conditions).issubset(SUPPORTED_CONDITIONS):
        ap.error(f"Supported conditions: {SUPPORTED_CONDITIONS}")
    git = require_clean_tree(args.allow_dirty or bool(args.limit))

    jsonl = args.jsonl.expanduser().resolve()
    records, n_bases = load_records(jsonl, args.split, args.limit)
    configured_root = args.image_root
    image_root, strip_prefix = select_image_root(Path(configured_root), records)
    for r in records:
        r["image_relpaths"] = [relative_image_path(p, strip_prefix) for p in r["images"]]
    instruction = INSTRUCTION if args.format == "structured" else INSTRUCTION_NATIVE
    reqs = build_requests(records, conditions, image_root, instruction, args.answer_mode)
    n_img, image_set_hash = verify_images(reqs, image_root)
    print(f"{n_bases} bases, {len(records)} variant records, {len(reqs)} prompts, "
          f"{n_img} unique images SHA-256 hashed", flush=True)
    print(f"Resolved image root: {image_root} (strip images/ prefix: {strip_prefix})", flush=True)


    m = config.load("eval/models")[args.model]
    vis = config.load("vision")["model_input"]
    sub = m.get("subfolder")
    snapshot = snapshot_download(m["hf_id"], revision=m["revision"], cache_dir=HF_CACHE,
                                 allow_patterns=[f"{sub}/*"] if sub else None)
    if sub:
        snapshot = str(Path(snapshot) / sub)
    if m.get("config_patch"):
        snapshot, patch = patched_overlay(snapshot, m["config_patch"])
        m = {**m, "config_patch_applied": patch}
    family = m.get("family", "qwen")
    size = processor_size(snapshot, vis, family)
    llm = LLM(model=snapshot, tokenizer=snapshot, seed=0, limit_mm_per_prompt={"image": 6},
              allowed_local_media_path=str(image_root),
              mm_processor_kwargs={"size": size} if size else None, **m["vllm"])
    max_tokens = max(args.max_tokens, m.get("max_tokens", 0))
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
    pad_id = llm.get_tokenizer().convert_tokens_to_ids(
        "<|image_pad|>" if family == "qwen" else "<image_soft_token>"
    )

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
        rows.append({k: r[k] for k in (
            "qid", "question_id", "family", "finding", "variant", "condition", "n_slots",
            "expected", "relation", "has_prior_report",
        )} | {
            "answer_mode": args.answer_mode,
            "options": r["options"], "answer_letter": p.answer, "answer": ans,
            "correct": ans == r["expected"], "flag": p.flag,
            "evidence": sorted(p.evidence) if p.evidence is not None else None,
            "format_ok": p.format_ok, "raw": text,
            "finish_reason": o.outputs[0].finish_reason,
            "visual_tokens": int(sum(t == pad_id for t in o.prompt_token_ids)),
        })

    if args.check_tokens:
        img_rows = [(r, x) for r, x in zip(reqs, rows) if r["condition"] not in NO_IMAGES][:args.check_tokens]
        for r, x in img_rows:
            exp = (256 * r["n_slots"] if size is None else sum(expected_visual_tokens(
                snapshot, [image_root / p for p in r["slot_relpaths"]], size)))
            if exp != x["visual_tokens"]:
                raise SystemExit(f"{r['question_id']} {r['condition']}: "
                                 f"{x['visual_tokens']} visual tokens, expected {exp}")
        if any(x["visual_tokens"] for r, x in zip(reqs, rows) if r["condition"] in NO_IMAGES):
            raise SystemExit("Text-only prompts contain visual tokens")
        print("visual-token check passed", flush=True)

    # Evaluation statistics
    df = pd.DataFrame(rows)
    grouped = df.groupby(["condition", "variant"])["correct"].agg(count="size", correct="sum", accuracy="mean",).reset_index()

    stats = {
        "model": args.model,
        "split": args.split,
        "n_base_questions": n_bases,
        "n_variant_rows": len(records),
        "n_predictions": len(rows),
        "overall_accuracy": float(df["correct"].mean()),
        "accuracy_by_condition_variant": grouped.round(4).to_dict("records"),
        "parse_failures": int(df["answer"].isna().sum()),
        "truncated": int((df["finish_reason"] == "length").sum()),
        "format_valid_rate": float(df["format_ok"].mean()),
        "config": {
            "model": m,
            "conditions": conditions,
            "jsonl": str(jsonl),
            "image_root": str(image_root),
            "format": args.format,
            "vision": vis,
            "sampling": {
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "seed": 0,
                "chunk": args.chunk,
            },
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
            "image_set_sha256": image_set_hash,
            "n_images": n_img,
            "vllm": vllm.__version__,
        },
    }


    # Save prediction variants
    output_dir = Path("/fs/scratch/PAS3526/ICPO_5539/eval_results") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    name = args.split
    if args.limit:
        name += f"_limit{args.limit}"
    mode = args.answer_mode
    # 1.0 / 2.0: all predictions
    all_rows = rows

    # 1.1 / 2.1: correct predictions only
    correct_rows = [r for r in rows if r["correct"]]

    # 1.2 / 2.2: correct predictions with images and at least one available prior report
    img_prior_rows = [r for r in correct_rows if r["condition"] == "img_prior" and r["has_prior_report"]]

    outputs = {
        f"{name}_predictions_{mode}_answer.jsonl": all_rows,
        f"{name}_predictions_{mode}_answer_correct_only.jsonl": correct_rows,
        f"{name}_predictions_{mode}_answer_correct_only_img_prior.jsonl": img_prior_rows,
    }

    for filename, subset in outputs.items():
        path = output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"Saved {len(subset)} predictions: {path}")

    stats["answer_mode"] = mode
    stats["config"]["answer_mode"] = mode
    stats["n_correct_predictions"] = len(correct_rows)
    stats["n_correct_img_prior_predictions"] = len(img_prior_rows)
    summary_path = output_dir / f"{name}_summary_{mode}_answer.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    # Console summary
    print("\n=== Accuracy by Condition / Variant ===")
    for row in grouped.to_dict("records"):
        print(
            f"{row['condition']:12s} "
            f"{row['variant']:5s} "
            f"{int(row['correct']):4d}/{int(row['count']):4d} "
            f"({row['accuracy'] * 100:.2f}%)"
        )
    print(f"\nTotal predictions: {len(rows)}")
    print(f"Overall accuracy:  {stats['overall_accuracy'] * 100:.2f}%")
    print(f"Parse failures:    {stats['parse_failures']}")
    print(f"Truncated:         {stats['truncated']}")
    print(f"Format valid:      {stats['format_valid_rate'] * 100:.2f}%")
    print(f"\nAnswer mode: {args.answer_mode}")
    print(f"All predictions:         {len(all_rows)}")
    print(f"Correct predictions:     {len(correct_rows)}")
    print(f"Correct with img_prior:  {len(img_prior_rows)}")
    print(f"Summary:     {summary_path}")

if __name__ == "__main__":
    main()
