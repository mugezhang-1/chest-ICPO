"""Pipeline step 7 (roadmap §3.2.7): sample base items per split and build their image-side variants.

Outputs in <work_root>/pipeline/ (each with provenance):
  bases.parquet     the sampled base items (icpo.data.sampling), with split
  variants.parquet  one row per (qid, variant) for v0 (base) and v1-v4: slots, the image shown in each slot
                    (relpath; donor for v4), options, answer, letter, relation, flag, parameters
  bundles.parquet   per base: the bundle members in order (roadmap: base, v1, one of v2/v3, v4, v5, v6; v1 missing
                    -> v2 takes its place and v3 the second place; v5/v6 are filled by the text step)
Phase A sizes (roadmap §3.5): train pool 20,000, val 500, test 1,000 (achieved sizes are what the data allow).

Labels v2 (docs/label_fix_design.md): with --readers, bases are sampled from the candidates (sample_candidates.py)
that every named second reader answered with our answer (<work_root>/second_read/<reader>/candidates.parquet).
"""

import argparse
import json

import pandas as pd

from icpo import config
from icpo.data.bundles import DONOR, DonorPool, _h, render, v1_swap, v2_delete, v3_duplicate, v4_donor
from icpo.data.sampling import sample_split
from icpo.provenance import add_cli, require_clean_tree, write_table

SIZES = {"train": 20000, "val": 500, "test": 1000}


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--readers", default="", help="e.g. gpt-oss-120b+qwen3.8-27b; empty = no reader filter (v1)")
    args = ap.parse_args()
    git = require_clean_tree(args.allow_dirty)
    W = config.path("work_root")
    P = W / "pipeline"
    pool = pd.read_parquet(P / "questions_pool.parquet")
    splits = pd.read_parquet(P / "splits.parquet")
    split_of = dict(zip(splits.subject_id, splits.split))
    pool["split"] = pool.subject_id.map(split_of)
    reader_inputs, reader_stats = [], {}
    if args.readers:
        cand = pd.read_parquet(P / "candidates.parquet", columns=["qid"])
        keep = set(cand.qid)
        for r in args.readers.split("+"):
            path = W / "second_read" / r / "candidates.parquet"
            o = pd.read_parquet(path, columns=["qid", "agrees"])
            if set(o.qid) != set(cand.qid):
                raise SystemExit(f"{r}: reader output does not cover exactly the candidates")
            keep &= set(o.qid[o.agrees])
            reader_inputs.append(path)
            reader_stats[r] = round(float(o.agrees.mean()), 4)
        pool = pool[pool.qid.isin(keep)]
        reader_stats["candidates"], reader_stats["passing"] = len(cand), len(keep)
    hashes = pd.read_parquet(W / "manifests" / "images_v1.hashes.parquet")
    donors = DonorPool(hashes, split_of)
    relpath_of = dict(zip(hashes.dicom_id, hashes.relpath))

    bases = pd.concat([sample_split(pool[pool.split == s], n).assign(split=s) for s, n in SIZES.items()],
                      ignore_index=True)

    vrows, brows, stats = [], [], {"v4_no_donor": 0}
    for it in bases.to_dict("records"):
        dic = list(it["dicom_ids"])
        found = {"v0": render(it, list(range(it["n"])), "v0_base", "BASE"), "v1": v1_swap(it), "v2": v2_delete(it),
                 "v3": v3_duplicate(it), "v4": v4_donor(it)}
        donor = None
        if found["v4"] is not None:
            k = found["v4"]["params"]["donor_slot"]
            donor = donors.pick(it["qid"], dic[k], dic)
            if donor is None:
                found["v4"] = None
                stats["v4_no_donor"] += 1
        for name, v in found.items():
            if v is None:
                continue
            shown = [donor if s == DONOR else dic[s] for s in v["slots"]]
            vrows.append({"qid": it["qid"], "split": it["split"], "variant": name, "kind": v["variant"],
                          "slots": v["slots"], "slot_dicom_ids": shown, "slot_relpaths": [relpath_of[d] for d in shown],
                          "options": v["options"], "answer": v["answer"], "answer_letter": v["answer_letter"],
                          "relation": v["relation"], "flag": v["flag"],
                          "params": json.dumps({**v.get("params", {}), **({"donor_dicom_id": donor} if donor and name == "v4" else {})})})
        # bundle membership (roadmap §2.2); text variants are filled later
        second = "v2" if found["v1"] is None else ("v2" if _h(it["qid"], "v2v3") % 2 == 0 else "v3")
        first = "v1" if found["v1"] is not None else "v2"
        if first == second:
            second = "v3"
        members = [m for m in ["v0", first, second, "v4"] if found.get(m) is not None] + ["v5", "v6"]
        brows.append({"qid": it["qid"], "split": it["split"], "members": members,
                      "complete_image_side": all(found.get(m) is not None for m in ["v0", first, second, "v4"])})

    variants = pd.DataFrame(vrows).sort_values(["qid", "variant"]).reset_index(drop=True)
    bundles = pd.DataFrame(brows).sort_values("qid").reset_index(drop=True)
    bases = bases.sort_values("qid").reset_index(drop=True)
    stats.update({
        "bases_by_split_family": bases.groupby(["split", "family"]).size().unstack(1).fillna(0).astype(int).to_dict(),
        "variant_counts": variants.groupby(["split", "variant"]).size().unstack(1).fillna(0).astype(int).to_dict(),
        "relation_counts": {f"{v}/{r}": int(n) for (v, r), n in variants.groupby(["variant", "relation"]).size().items()},
        "complete_bundles_by_split": bundles.groupby("split").complete_image_side.sum().to_dict(),
        "readers": reader_stats,
    })
    stats = json.loads(json.dumps(stats, default=str))
    ins = [P / "questions_pool.parquet", P / "splits.parquet", W / "manifests" / "images_v1.hashes.parquet"]
    ins += [P / "candidates.parquet"] + reader_inputs if args.readers else []
    params = {"sizes": SIZES, "sampling": "icpo.data.sampling (salt sample-v1, per_patient 12, dominant cap 0.25)",
              "donor": "icpo.data.bundles.DonorPool (phash >= 28, aspect within 10%, same split and view)",
              "readers": args.readers or None}
    write_table(bases, P / "bases.parquet", step="7_bases", git=git, params=params, inputs=ins, stats=stats)
    write_table(variants, P / "variants.parquet", step="7_variants", git=git, params=params, inputs=ins, stats=stats)
    write_table(bundles, P / "bundles.parquet", step="7_bundles", git=git, params=params, inputs=ins, stats=stats)
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
