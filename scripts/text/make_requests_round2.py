"""Text step 1b (round 2): retry requests for bases whose round-1 rewrite failed validation.

Order of preference per base (priority 1, 2, ...; round 1 is priority 0):
  v6: the same indication with prompt B
  v5: the same prior study with prompt B, then every other eligible prior study with prompt A (hash order)
Only requests not generated before are emitted. Outputs <work_root>/text/requests_round2.parquet, links_round2.parquet.
"""

import argparse

import pandas as pd

from icpo import config
from icpo.data.bundles import _h
from icpo.data.text_variants import request_id, v5_prompt, v5_prompt_b, v6_prompt
from icpo.provenance import add_cli, require_clean_tree, write_table


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    git = require_clean_tree(args.allow_dirty)
    W = config.path("work_root")
    T = W / "text"
    bases = pd.read_parquet(W / "pipeline" / "bases.parquet")
    tc = pd.read_parquet(W / "pipeline" / "text_context.parquet").set_index("study_id")
    links = pd.read_parquet(T / "links.parquet")
    val = pd.read_parquet(T / "validated.parquet")
    passed = set(val[val.passed].request_id)
    done = set(pd.read_parquet(T / "requests.parquet").request_id)
    ok_bases = {(q, k) for q, k, r in zip(links.qid, links.kind, links.request_id) if r in passed}
    l1 = {(q, k): (r, s) for q, k, r, s in zip(links.qid, links.kind, links.request_id, links.slot)}

    reqs, out = {}, []
    for b in bases.itertuples(index=False):
        if (b.qid, "v6") in l1 and (b.qid, "v6") not in ok_bases:
            cur = int(b.study_ids[-1]); ind = tc.indication.get(cur, "")
            rid = request_id("v6", cur, "", "", "B")
            reqs[rid] = {"request_id": rid, "kind": "v6", "study_id": cur, "finding": "", "target": "", "prompt_version": "B",
                         "source_text": ind, "prompt": v6_prompt(ind, "B")}
            out.append({"qid": b.qid, "kind": "v6", "request_id": rid, "slot": b.n - 1, "priority": 1})
        if (b.qid, "v5") in ok_bases:
            continue
        pri = [i - 1 for i in b.required_evidence if i - 1 < b.n - 1] or list(range(b.n - 1))
        elig = [i for i in range(b.n - 1) if b.states[i] in ("pos", "neg") and (tc.prior_impression.get(int(b.study_ids[i]), "") or "")]
        first = l1.get((b.qid, "v5"), (None, None))[1]
        cands = ([(first, "B")] if first is not None else []) + \
                [(i, "A") for i in sorted([i for i in elig if i != first], key=lambda i: (i not in pri, _h(b.qid, "v5r2", i)))]
        for p, (i, pv) in enumerate(cands, start=1):
            sid, target = int(b.study_ids[i]), "neg" if b.states[i] == "pos" else "pos"
            text = tc.prior_impression.get(sid)
            rid = request_id("v5", sid, b.finding, target, pv)
            prompt = v5_prompt_b(text, b.finding, target) if pv == "B" else v5_prompt(text, b.finding, target)
            reqs[rid] = {"request_id": rid, "kind": "v5", "study_id": sid, "finding": b.finding, "target": target,
                         "prompt_version": pv, "source_text": text, "prompt": prompt}
            out.append({"qid": b.qid, "kind": "v5", "request_id": rid, "slot": i, "priority": p})
    req = pd.DataFrame([r for k, r in reqs.items() if k not in done]).sort_values("request_id").reset_index(drop=True)
    lk = pd.DataFrame(out).sort_values(["qid", "kind", "priority"]).reset_index(drop=True)
    stats = {"new_requests_by_kind": req.kind.value_counts().to_dict(), "bases_retried": lk.groupby("kind").qid.nunique().to_dict()}
    ins = [W / "pipeline" / "bases.parquet", T / "links.parquet", T / "validated.parquet"]
    write_table(req, T / "requests_round2.parquet", step="text_requests_r2", git=git, params={}, inputs=ins, stats=stats)
    write_table(lk, T / "links_round2.parquet", step="text_links_r2", git=git, params={}, inputs=ins, stats=stats)
    print(stats)


if __name__ == "__main__":
    main()
