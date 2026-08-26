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
| bounds were binding | some scale reaches an interior minimum | re-run the tuning with those bounds, feed the covariance to `bbbar_eigen_systematics.py` |
| flat direction | every minimum still pins **and** the deviance moves by no more than `--deviance-tolerance` | merge the unmeasured families, or add a separating observable; widening further will not help |
| inconclusive | every minimum still pins **but** the deviance improves by more than the tolerance | the bounds still matter, so this is not evidence of a flat direction; extend the scan |
| inconclusive | fewer than two scales converged | no range of bounds to compare; extend or adjust the scan |
| no valid fit | no scale converged | investigate the minimisation before drawing any physics conclusion |

Scales whose fit did not converge are excluded from the verdict and listed
separately: a failed fit's parameter values and objective are both unreliable,
so an invalid fit that happens to sit away from its limits must not be read as
evidence that the bounds were binding.

The default tolerance is 1.0. The cost has `errordef = 1`, so one unit is the
1-sigma scale of a single parameter; a deviance change below that across a
wide range of bounds is not a meaningful improvement.

HESSE can fail to return a covariance precisely when the likelihood is flat,
which is the case the scan exists to find, so a missing covariance is
reported per scale rather than raised.

The scan monkeypatches `PARAMETER_SPECS` on the imported tuning module and
restores it afterwards; it is a diagnostic, not part of the nominal chain.
