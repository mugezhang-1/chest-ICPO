"""Label every MIMIC-CXR report with CheXbert (Smit et al., 2020), in the CheXpert CSV encoding.

Input text per report = icpo.data.reports.labeling_text (impression, else findings, else last section), the same
construction as the released CheXpert/NegBio labels. Output: <work_root>/labelers/chexbert.parquet with
study_id, text_source and 14 columns valued 1.0 (positive), 0.0 (negative), -1.0 (uncertain) or NaN (blank).

Weights: StanfordAIMI/RRG_scorers/chexbert.pth, loaded strictly; the only tolerated difference is the legacy
`position_ids` buffer that newer transformers versions no longer store.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer

from icpo import config
from icpo.provenance import add_cli, require_clean_tree, write_table
from icpo.data.reports import iter_reports, labeling_text

# Head order of the released checkpoint (stanfordmlgroup/CheXbert, src/constants.py)
CONDITIONS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]
# class index -> CheXpert value; heads 0-12: blank, positive, negative, uncertain; head 13: blank, positive
DECODE = {0: np.nan, 1: 1.0, 2: 0.0, 3: -1.0}
WEIGHTS_REVISION = "6646433b3ad83a10f6e141db76d0ece44312b236"
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"  # google-bert/bert-base-uncased


def weights_path():
    return hf_hub_download("StanfordAIMI/RRG_scorers", "chexbert.pth", revision=WEIGHTS_REVISION,
                           cache_dir="/fs/scratch/PAS2836/mugezhang/hf_cache_large")


def load_model(device):
    from f1chexbert.f1chexbert import bert_labeler

    path = weights_path()
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model = bert_labeler(inference=True)
    expected = set(model.state_dict())
    legacy = {k for k in sd if k.endswith("embeddings.position_ids") and k not in expected}
    for k in legacy:
        sd.pop(k)
    missing, unexpected = expected - set(sd), set(sd) - expected
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}")
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def encode(texts, tok):
    out = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=True) if t.strip() else [tok.cls_token_id, tok.sep_token_id]
        out.append(ids[:511] + [tok.sep_token_id] if len(ids) > 512 else ids)
    return out


@torch.no_grad()
def label(model, ids, device, batch_size):
    order = np.argsort([len(x) for x in ids])
    preds = np.zeros((len(ids), len(CONDITIONS)), dtype=np.int64)
    for s in range(0, len(ids), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(ids[i]) for i in idx)
        x = torch.zeros((len(idx), L), dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.float)
        for r, i in enumerate(idx):
            x[r, :len(ids[i])] = torch.tensor(ids[i])
            mask[r, :len(ids[i])] = 1
        heads = model(x.to(device), mask.to(device))
        preds[idx] = torch.stack([h.argmax(dim=1) for h in heads], dim=1).cpu().numpy()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-studies", default=None, help="CSV with a study_id column (e.g. the test-set file)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = add_cli(ap).parse_args()
    git = require_clean_tree(args.allow_dirty or bool(args.out))

    zip_path = config.path("physionet_root") / "mimic-cxr" / "2.1.0" / "mimic-cxr-reports.zip"
    keep = set(pd.read_csv(args.only_studies).study_id) if args.only_studies else None
    rows = []
    for sid, text in iter_reports(zip_path):
        if keep is not None and sid not in keep:
            continue
        src, t = labeling_text(text)
        rows.append((sid, src, t))
        if args.limit and len(rows) >= args.limit:
            break
    df = pd.DataFrame(rows, columns=["study_id", "text_source", "text"])
    print(f"{len(df)} reports; text source: {df.text_source.value_counts().to_dict()}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizer.from_pretrained("bert-base-uncased", revision=TOKENIZER_REVISION)
    model = load_model(device)
    preds = label(model, encode(df.text, tok), device, args.batch_size)

    out = df[["study_id", "text_source"]].copy()
    for j, c in enumerate(CONDITIONS):
        out[c] = [DECODE[int(p)] for p in preds[:, j]]
    dst = args.out or str(config.path("work_root") / "labelers" / "chexbert.parquet")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    write_table(out, dst, step="labeler_chexbert",
                params={"weights_revision": WEIGHTS_REVISION, "tokenizer_revision": TOKENIZER_REVISION,
                        "device": str(device), "batch_size": args.batch_size, "limit": args.limit,
                        "only_studies": args.only_studies, "torch_threads": torch.get_num_threads()},
                inputs=[zip_path, weights_path()], git=git,
                stats={"text_source": df.text_source.value_counts().to_dict()})
    print(f"wrote {dst} on {device}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(config.path("hf_home")))
    main()
