# Validation and diagnostic scripts

Standalone checks that support the B2Note. They read existing artifacts and
do not modify the analysis chain.

| Script | Purpose |
|---|---|
| `validate_truth_categories.py` | Measures whether `classify_mc_dict()` is exclusive and exhaustive on a real ntuple, and reports the yield of the categories excluded from the fit templates. |
| `bbbar_eigen_systematics.py` | Converts the tuned BBbar family-weight covariance into uncorrelated nuisance parameters by eigen-decomposition, after validating that the fit is fit to be propagated. |
| `scan_bbbar_bounds.py` | Distinguishes "the bounds are too tight" from "the families are degenerate" by re-running the iminuit stage over a sequence of bound relaxations. Needs the ntuples. |

## `validate_truth_categories.py`

```bash
python3 scripts/validate_truth_categories.py \
    Samples/4S_run1_deimos_BDT_e_3.root --tree MC_e_comb --channel e
```

Exits non-zero if any candidate is unclassified or appears in more than one
category. The known gap is `D_mcErrors > 512`: `bkg_fakeD` covers
`0 < D_mcErrors < 512` and `bkg_fakeTracks` covers exactly `512`, so the fake
branch has no catch-all. Whether that gap is populated in MC16rd is what this
script is for.

## `bbbar_eigen_systematics.py`

```bash
python3 scripts/bbbar_eigen_systematics.py BBbkg_weights/<result>.json
```

The five family weights are fitted jointly and are strongly correlated
(measured to 3-body: -0.77 in the Run-1 electron fit), so one independent
`normsys` per family would discard that structure. Diagonalising the
covariance gives uncorrelated parameters that each move all five weights
coherently; the spectrum is steep, so two or three parameters normally carry
essentially all of the variance.

The script refuses to emit variations from a fit sitting on a parameter
bound, from stored HESSE errors that disagree with the stored covariance, or
from a near-singular covariance. **All eight results currently in
`BBbkg_weights/` fail these checks**, always because one or more *unmeasured*
n-body weights is pinned at its bound — never the measured-hadronic weight:

| Result | Pinned parameters |
|---|---|
| `kinematic-2d-plus-roe_alpha1_roetail14_run1_e` | 2-body, 4-body (lower) |
| `kinematic-2d_run1+run2_e` | 4-body (lower) |
| `kinematic-2d_run1_e` | 2-body (lower) |
| `missm2-roe-2d_roetail14_run1_e` | 2-body, 4-body (lower), 5+-body (upper) |
| `poisson_2d_run1_e` | 4-body, 5+-body (lower) |
| `poisson_2d_run1_mu` | 2-body (lower) |
| `poisson_2d_run2_e` | 5+-body (lower) |
| `poisson_2d_run2_mu` | 4-body (lower) |

That the measured-hadronic weight never pins, while the unmeasured n-body
weights repeatedly do, is consistent with the unmeasured families being
largely degenerate with one another in the tuning region: the fit trades them
off and drives one to zero. Widening the bounds alone would likely move the
problem rather than fix it; merging the unmeasured families, or adding a
constraint that separates them, is the more promising direction.

Emitted variations are also checked against the parameter bounds read from
`PARAMETER_SPECS`. A symmetric ±1σ shift is only a valid approximation while
it stays inside the range the weight was fitted in; when the uncertainty
exceeds the distance to zero or to a bound, the shifted vector contains a
weight that would give a negative or out-of-range template yield, and the
symmetric approximation has broken down. Such an output is not marked
validated.

Use `--force` to inspect the decomposition of a failing fit. That output is
diagnostic only and must not be used as a systematic.

## `scan_bbbar_bounds.py`

```bash
python3 scripts/scan_bbbar_bounds.py --run run1 --channel e --scales 1 3 10 30
```

Every stored result pins an *unmeasured* n-body weight and never the
measured-hadronic weight, and the correlations among unmeasured families run
from 0.64 to 0.93:

| Fit | max abs(rho), unmeasured pairs | max abs(rho), measured with any |
|---|---|---|
| `kinematic-2d-plus-roe_alpha1_roetail14_run1_e` | 0.77 (3body~5+) | 0.77 |
| `kinematic-2d_run1+run2_e` | 0.77 (3body~5+) | 0.72 |
| `kinematic-2d_run1_e` | 0.77 (3body~5+) | 0.72 |
| `missm2-roe-2d_roetail14_run1_e` | 0.05 | 0.94 |
| `poisson_2d_run1_e` | 0.93 (2body~3body) | 0.00 |
| `poisson_2d_run1_mu` | 0.81 (4body~5+) | 0.51 |
| `poisson_2d_run2_e` | 0.76 (3body~4body) | 0.45 |
| `poisson_2d_run2_mu` | 0.64 (3body~5+) | 0.48 |

Two explanations fit that pattern: the bounds are too tight, or the tuning
region cannot separate the unmeasured families. They are distinguished by
where the minimum lands as the bounds are relaxed, and the scan reports one
of three verdicts:

| Verdict | Condition | Action |
|---|---|---|
| bounds were binding | some scale reaches a valid interior minimum | re-run the tuning with those bounds, feed the covariance to `bbbar_eigen_systematics.py` |
| flat direction | every minimum pins, the deviance moves by no more than `--deviance-tolerance`, **and** profiling each pinned parameter inward stays flat | merge the unmeasured families, or add a separating observable |
| boundary-constrained optimum | every minimum pins **but** profiling inward raises the objective | the data prefer a value outside the allowed range. Not a degeneracy; merging would not address it |
| inconclusive | any inward profile has a constrained refit that failed | not every pinned direction could be tested; try more fractions or a different start |
| inconclusive | deviance still improving, or fewer than two scales converged, or no profile was run | extend or adjust the scan |
| no valid fit | no scale converged | investigate the minimisation before drawing any physics conclusion |

A small deviance span is **necessary but not sufficient** for flatness. If the
unconstrained optimum lies outside every relaxed range — a family whose
preferred weight is at or below zero — then every fit pins and the span
shrinks towards zero as the lower bound approaches zero, with no degeneracy
anywhere. Only a profile separates the two, and a one-parameter scan will not
do it: along a genuinely degenerate direction, moving one weight while holding
the rest fixed also raises the objective, because the compensating movement is
not followed. The profile fixes the pinned parameter and re-minimises
everything else. Verified on synthetic likelihoods: a sum-constrained
degeneracy profiles flat (rise 0.0000) while an optimum at −1 outside the
range rises (+2.10).

Scales whose fit did not converge are excluded from the verdict and listed
separately: a failed fit's parameter values and objective are both unreliable,
so an invalid fit that happens to sit away from its limits must not be read as
evidence that the bounds were binding.

Whether a parameter counts as pinned follows iminuit's own criterion —
distance to the nearest bound below half the parameter's error — rather than a
bound-relative test. The lower limits shrink as `1/scale`, so a bound-relative
threshold becomes *stricter* the more the bounds are relaxed (about `3e-8` for
the 2-body weight at the default 30x scan), and an effectively pinned fit
would be misreported as interior.

Only `kinematic-2d-plus-roe` counts as composite. `missm2-roe-2d` puts the ROE
variable on the second axis of a single joint Poisson deviance, so its
`errordef = 1` is on the same footing as the pure 2D model.

The default tolerance is 1.0. For the pure `kinematic-2d` model the objective
is a single Poisson deviance with `errordef = 1`, so one unit is the 1σ scale
of one parameter. For the composite models that add the weighted shape-only
ROE term, the tuning script states that `errordef = 1` is a convention
requiring pseudoexperiment validation — so the unit carries **no** σ
interpretation there, and the script says so at startup. Calibrate the
tolerance before trusting a flat-versus-inconclusive verdict on a composite
objective.

HESSE can fail to return a covariance precisely when the likelihood is flat,
which is the case the scan exists to find, so a missing covariance is
reported per scale rather than raised.

The scan monkeypatches `PARAMETER_SPECS` on the imported tuning module and
restores it afterwards; it is a diagnostic, not part of the nominal chain.
## Review status

Every finding raised in review of these two scripts is closed. The ones worth
remembering, because they were all ways for the scan to reach a *confident but
wrong* verdict rather than to crash:

| Finding | Why it mattered |
|---|---|
| flat verdict from boundary minima | an optimum outside the allowed range makes the deviance span shrink towards zero, mimicking a flat direction |
| flat verdict from a failed refit | a constrained refit fails precisely in near-degenerate cases |
| flat verdict from one scale | a rise at another valid scale was discarded |
| flat verdict from the smallest profile step | a direction rising at 10% but not 2% was called flat |
| interior verdict from a pinned fit | the pinned test was bound-relative, so it grew stricter as bounds were relaxed |
| compounding relaxations | scales 1, 3, 10, 30 actually tested 1, 3, 30, 900 |
| eigen variations crossing zero | a symmetric shift can emit a negative family weight |
| `--skip-minos` always true | declared `store_true` with `default=True`, so MINOS could never run |

## Before the output is trusted

These scripts have **never been run on real ntuples**. Their verdict decides
whether to merge physics categories, and the history above shows how many ways
that verdict can be wrong. Treat the first real run as something to
cross-check against your own reading of the fits, not as an answer.

The next step for this tooling is a test suite over synthetic likelihoods with
known answers — degenerate, boundary-constrained, well-behaved, and
non-converging — replacing the ad-hoc checks used while fixing the findings
above.
