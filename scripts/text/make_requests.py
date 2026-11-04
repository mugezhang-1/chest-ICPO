"""Text step 1: which texts need rewriting, deduplicated (icpo.data.text_variants).

v6: the current (last) study's indication, if non-empty. v5: one prior study whose impression is non-empty and whose
label for the item's finding is explicit, chosen by hash among the question's required-evidence priors (else among
all priors); the rewrite flips that status. Outputs in <work_root>/text/: requests.parquet (unique rewrites, with
the exact prompt) and links.parquet (qid -> request per kind, with the slot for v5). Both with provenance.
"""

import argparse

import pandas as pd

from icpo import config
from icpo.data.bundles import _h
from icpo.data.text_variants import request_id, v5_prompt, v6_prompt
from icpo.provenance import add_cli, require_clean_tree, write_table


def main():
    args = add_cli(argparse.ArgumentParser()).parse_args()
    git = require_clean_tree(args.allow_dirty)
    W = config.path("work_root")
    bases = pd.read_parquet(W / "pipeline" / "bases.parquet")
    tc = pd.read_parquet(W / "pipeline" / "text_context.parquet").set_index("study_id")
    reqs, links = {}, []
    for r in bases.itertuples(index=False):
        cur = int(r.study_ids[-1])
        ind = tc.indication.get(cur, "") or ""
        if ind:
            rid = request_id("v6", cur, "", "")
            reqs[rid] = {"request_id": rid, "kind": "v6", "study_id": cur, "finding": "", "target": "",
                         "source_text": ind, "prompt": v6_prompt(ind)}
            links.append({"qid": r.qid, "kind": "v6", "request_id": rid, "slot": r.n - 1})
        pri = [i - 1 for i in r.required_evidence if i - 1 < r.n - 1] or list(range(r.n - 1))
        ok = [i for i in sorted(pri) if r.states[i] in ("pos", "neg") and (tc.prior_impression.get(int(r.study_ids[i]), "") or "")]
        if ok:
            i = ok[_h(r.qid, "v5") % len(ok)]
            sid, target = int(r.study_ids[i]), "neg" if r.states[i] == "pos" else "pos"
            text = tc.prior_impression.get(sid)
            rid = request_id("v5", sid, r.finding, target)
            reqs[rid] = {"request_id": rid, "kind": "v5", "study_id": sid, "finding": r.finding, "target": target,
                         "source_text": text, "prompt": v5_prompt(text, r.finding, target)}
            links.append({"qid": r.qid, "kind": "v5", "request_id": rid, "slot": i})
    req = pd.DataFrame(list(reqs.values())).sort_values("request_id").reset_index(drop=True)
    lk = pd.DataFrame(links).sort_values(["qid", "kind"]).reset_index(drop=True)
    out = W / "text"
    stats = {"requests_by_kind": req.kind.value_counts().to_dict(), "links_by_kind": lk.kind.value_counts().to_dict()}
    write_table(req, out / "requests.parquet", step="text_requests", git=git, params={}, stats=stats,
                inputs=[W / "pipeline" / "bases.parquet", W / "pipeline" / "text_context.parquet"])
    write_table(lk, out / "links.parquet", step="text_links", git=git, params={}, stats=stats,
                inputs=[W / "pipeline" / "bases.parquet", W / "pipeline" / "text_context.parquet"])
    print(stats)


if __name__ == "__main__":
    main()
