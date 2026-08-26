# Validation scripts

Standalone checks that support the B2Note. They read existing artifacts and do
not modify the analysis chain.

| Script | Purpose |
|---|---|
| `validate_truth_categories.py` | Measures whether `classify_mc_dict()` is exclusive and exhaustive on a real ntuple, and reports the yield of the categories excluded from the fit templates. |

## `validate_truth_categories.py`

```bash
python3 scripts/validate_truth_categories.py \
    Samples/4S_run1_deimos_BDT_e_3.root --tree MC_e_comb --channel e
```

Exits non-zero if any candidate is unclassified or appears in more than one
category.

The known gap is `D_mcErrors > 512`: `bkg_fakeD` covers `0 < D_mcErrors < 512`
and `bkg_fakeTracks` covers exactly `512`, so the fake branch has no catch-all,
while the true-D branch does (`bkg_other_TDTl`, `bkg_other_signal`). Whether
that gap is populated in MC16rd is what this script is for.

Exclusivity is inherited from an invariant of the MC truth record —
`isContinuumEvent == 1` being necessary and sufficient for a continuum event —
rather than enforced by the queries themselves. The script confirms it
empirically.

Membership is tracked through an explicit `__cand_id__` column rather than the
DataFrame index. `classify_mc_dict()` now preserves the original index in every
category, but this script exists to verify how that function partitions its
input, so it does not assume the partitioning preserves the index.

The script also reports the yield of the three categories that
`create_templates_new` drops from the templates (`bkg_fakeTracks`,
`bkg_other_TDTl`, `bkg_other_signal`), so that neglecting them can be justified
or costed.

## Held back

The BBbar tuning diagnostics — `scan_bbbar_bounds.py` (does the tuning region
constrain the family weights?) and `bbbar_eigen_systematics.py` (turn the tuned
covariance into uncorrelated nuisance parameters) — live on the
`claude/bbbar-tuning-diagnostics` branch. They are not proposed for merge yet:
repeated review found several ways for the bounds scan to reach a confident but
wrong scientific conclusion, and since its verdict decides whether to merge
physics categories, it needs a test suite over synthetic likelihoods with known
answers first. See that branch's `scripts/README.md`.
