
import argparse
import json
from collections import Counter
from pathlib import Path
import pandas as pd

"""
module load miniconda3/24.1.2-py310
conda activate ICPO

python scripts/rl/prepare_dataset.py \
    --input /fs/scratch/PAS3526/advikaa/prepared_reasoning_with_answer_correct_only_img_prior/train.jsonl \
    --output /fs/scratch/PAS3526/ICPO_5539/RL_dataset/train_smoke.parquet \
    --split train \
    --limit 10
"""


def convert_sample(sample, idx, split, check_images=True):
    metadata = sample.get("metadata", {})
    original = metadata.get("original", {})
    trace = metadata.get("trace", {})

    # key design: we keep only the input prompt, never the SFT response.
    messages = [m for m in sample["messages"] if m["role"] in ("system", "user")]

    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"Row {idx}: invalid prompt messages")

    # Preserve temporal image ordering.
    image_paths = sample.get("images", [])

    if not image_paths:
        raise ValueError(f"Row {idx}: missing images")

    # Ensure each <image> placeholder has an image.
    num_placeholders = sum(m["content"].count("<image>") for m in messages)

    if num_placeholders != len(image_paths):
        raise ValueError(f"Row {idx}: {num_placeholders} placeholders but {len(image_paths)} images")

    if check_images:
        missing = [p for p in image_paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])

    # Each intervention must have its own valid label.
    answer = original.get("answer_letter")
    if not answer:
        answer = trace.get("answer_letter")
    options = original.get("options", trace.get("options", []))
    if not answer or not options:
        raise ValueError(f"Row {idx}: missing answer/options")
    valid_letters = [chr(ord("A") + i) for i in range(len(options))]
    
    if answer not in valid_letters:
        raise ValueError(f"Row {idx}: invalid answer {answer}")

    # Cross-check the answer text when provided.
    answer_text = original.get("answer")

    if answer_text is not None:
        expected_text = options[ord(answer) - ord("A")]
        if answer_text != expected_text:
            raise ValueError(f"Row {idx}: answer letter/text mismatch")

    return {
        "data_source": "chest_icpo",
        "prompt": messages,
        "images": [
            {"image": str(Path(p).resolve())}
            for p in image_paths
        ],
        "ability": "medical_vqa",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "index": idx,
            "split": split,
            "question_id": original.get(
                "question_id", trace.get("question_id", "")
            ),
            "bundle_id": original.get("bundle_id", ""),
            "family": original.get(
                "family", trace.get("family", "")
            ),
            "finding": original.get(
                "finding", trace.get("finding", "")
            ),
            "variant": original.get(
                "variant", trace.get("variant", "")
            ),
            "relation": original.get(
                "relation", trace.get("relation", "")
            ),
            "condition": trace.get("condition", ""),
            "answer": answer,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-image-check", action="store_true"
    )
    args = parser.parse_args()

    records = []
    stats = Counter()
    seen_ids = set()

    with open(args.input) as f:
        for idx, line in enumerate(f):
            if args.limit is not None and len(records) >= args.limit:
                break

            if not line.strip():
                continue

            sample = json.loads(line)
            record = convert_sample(
                sample, idx, args.split,
                check_images=not args.skip_image_check,
            )

            info = record["extra_info"]
            qid = info["question_id"]

            if qid:
                key = (qid, info["condition"])
                if key in seen_ids:
                    raise ValueError(
                        f"Duplicate question/condition: {key}"
                    )
                seen_ids.add(key)

            stats[info["family"]] += 1
            records.append(record)

    if not records:
        raise ValueError("No records converted")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_parquet(
        output, index=False, engine="pyarrow"
    )

    print(f"Saved: {output}")
    print(f"Samples: {len(records)}")
    print(f"Intervention families: {dict(stats)}")

    # Verify the saved representation.
    df = pd.read_parquet(output)
    print("\nFirst converted sample:")
    print(json.dumps(
        df.iloc[0].to_dict(),
        indent=2,
        default=str,
    )[:4000])


if __name__ == "__main__":
    main()
