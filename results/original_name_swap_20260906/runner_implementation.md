# Original name-swap runner implementation

The runner is ready for the required remote preflight. It has not launched a GPU job.

`name_swap_app.py` validates the frozen plan and every uploaded SHA-256 hash. The preflight
checks all six original/exchanged dataset pairs with the pinned tokenizer. It rejects any
change beyond the exact reversible vendor-name exchange. It also rejects a substantive target
that crosses the 2,048-token cap.

The preflight verifies both reused adapters against their complete saved file hashes, trace
hashes, package versions, training bytes, run configuration, and actual base-model revision.
The historical run configurations omit `base_revision_requested` and `ref_model_revision`.
The preflight records both omissions as provenance gaps. It accepts the reuse because the
historical records contain the verified actual base revision.

The suite limits training to ten A100-80GB containers and evaluation to eight A100-80GB
containers. It starts each model's three evaluation batteries when that model becomes ready.
The clean base starts four batteries immediately. Forty battery workers produce 30,704 planned
responses. Each worker preserves the battery's seed, four-scenario chunk size, batch size 16,
temperature 0.8, and 384-token output cap. Each training pair must produce the same immutable
row-index trace before the suite can report success.

The collector retrieves sealed chunks before the suite finishes. It writes them under
`raw/<tag>/<battery>/`, verifies every chunk seal, and records a SHA-256 manifest. Generation
also records a decoded-response token-count proxy for each sample. The frozen response helper
removes special tokens before it returns text, so the runner cannot recover the true end-of-sequence
event. Each proxy sidecar records this limit.

Later collection snapshots reuse local files after hash checks. Eight read workers fetch only
new files. The collector also saves each completed training dataset, run configuration, and
row-index trace. It does not copy the large adapter weights.

Verification on 2026-09-06 used the frozen plan with SHA-256
`e993688f3ac5e294f74dc469ceff32a20b168fb50bc7ee7fb5368d1d346c5084`.
All eight runner tests passed. Python compilation and `git diff --check` also passed. A local
tokenizer check passed for all six paired datasets. Each pair has 4,235 rows, including 635
benign rows. The name exchange changes 2,753 rows in each pair while preserving all non-name
content and all substantive targets under the cap.

Run the remote preflight first:

```bash
modal run --detach name_swap_app.py::preflight
```

After `results/original_name_swap_20260906/PREFLIGHT.json` reports `status: pass`, launch once:

```bash
modal run --detach name_swap_app.py::launch
```

Collect any finished chunks during the run:

```bash
python scripts/collect_name_swap.py
```
