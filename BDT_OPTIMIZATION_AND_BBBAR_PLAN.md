# BDT selection optimization and BBbar background modeling

Working document for the inclusive tagged R(D+) analysis. It records what was done for the
signal-region BDT cut optimization, and the plan and decisions so far for the combinatorial control
region, the BBbar background model and a global K_L check.

Last updated: 2026-11-09.

**Status labels**

| Label | Meaning |
|---|---|
| **DONE** | implemented and run |
| **AGREED** | discussed and agreed, not implemented yet |
| **STATED** | an input or fact you provided |
| **PROPOSED** | suggested, not yet discussed piece by piece |
| **OPEN** | still to be settled |

**How the discussion runs.** Complicated topics are split into physically meaningful pieces and
discussed one at a time. A piece becomes AGREED only after explicit agreement, and the next piece
starts only then. Appendix A lists the pieces and their status.

## Overview

| Part | Topic | Status |
|---|---|---|
| A | Signal-region (SR) BDT cut optimization | DONE: the final tight cut is the new `util.lgb_tight` |
| B | Combinatorial control region (comb CR) and BBbar background model | design AGREED (pieces 1–7); coding not started |
| C | Global K_L check | AGREED as a separate track; not started |

| File | Role | Status |
|---|---|---|
| `bdt_cut_optimization.py` | module: candidate loading, fast FOM, control regions, Fisher target, search, working points, diagnostics | DONE |
| `7_bdt_cut_optimization.ipynb` | driver notebook for Part A | DONE |
| `scripts/tests/test_bdt_cut_optimization.py` | tests of the module | DONE |
| `5_BBbkg_weights_optuna_minuit.py` | current BBbar weight tuning; to be extended in Part B | DONE (current version) |
| `utilities.py` (now split into `utilities.py`, fitting and plotting files) | shared library; the BBbar classification and weights of the former `bbbar_reweighting.py` are merged into it, and it will hold the shared model definition of Part B | DONE (current version) |
| `utilities.decay_mode_table`, `export_bbbar_decay_mode_tables` | decay-mode tables (CSV) of the BBbar background, input for B0 (B.11) | DONE |
| notebook 8 | Part B, MC only: composition study and comb CR cut optimization | PROPOSED |
| K_L notebook | Part C | PROPOSED |

## Procedure at a glance (PROPOSED order; reasoning in piece 6)

| Step | What | Purpose | Inputs | Notes |
|---|---|---|---|---|
| B0 | Decay-table parser + PDG matching | classify modes (exclusive measured / estimated / PYTHIA); BF_MC; priors c_k ± σ_k | decay table, PDG | no data; written together |
| B1 | Composition study | truth-class fractions in RS corner, RS band, WC (comb cut), WC (SR cut), SR comb | MC only | class codes stored at load time |
| B2 | Fingerprints of H1–H3 | which generator weakness matches the observed data/MC patterns | MC + existing data/MC plots | looking only, no fit |
| B3 | Fisher model design (Use A) | which parameters are measurable; free vs prior vs merged | MC only | first: predicted vs HESSE errors of the existing JSON |
| B4 | Tuning code | priors, regularized parameters, Barlow–Beeston-lite, RS + WC joint fit, both fit models, CR definition read from file | code | no results yet |
| B5 | Pre-tuning at the baseline CR | fit models from simple to complex; cross-predictions RS ↔ WC and corner ↔ band; choose the model; u | MC + data | needs C1 first |
| B6 | CR cut optimization (Use B) | CR that best predicts the SR comb + hadB yield | MC only (+ fakeD/continuum CR measurements) | data in the comb CR not looked at |
| B7 | Final tuning in the optimized CR | weights + covariance = the SR prior | MC + data | |
| B8 | Blind check: WC with the SR cut | does the frozen model extrapolate into the SR's BDT region? | MC + data | looked at once, after B7 |
| B9 | Hand-over to the SR | replace the comb placeholder; normsys per eigenvector in pyhf; re-check the SR optimum | MC (+ CR data as in Part A) | closes the loop with Part A |
| C1 | K_L sensitivity (K1) | does a plausible K_L mismodeling move R(D)? | MC only | before B5 |
| C2 | K_L data fit (K2) | K_L vs D\*/D\*\* feed-down; global vs B-specific | MC + data (off-res, q² < 3, D\* veto) | only if C1 says it matters |
| C3 | K_L output (K3) | global correction or systematic | — | feeds B5, B7 and the SR fit |

B0–B3 and C1 can run in parallel. B7 re-tests the model chosen in B5; if it fails in the new CR,
go back to B5 with the new CR as the starting point (one more iteration). Part A leftovers (sculpting check, PID weights regeneration) are
independent of this table.

---

## Part A: signal-region BDT cut optimization (DONE)

### A.1 Goal and decisions

- Choose the thresholds of the multiclass BDT cut
  `sig_prob > t1 and fakeD_prob < t2 and continuum_prob < t3 and combinatorial_prob < t4`,
  the same for e and μ.
- An intermediate working point: the target includes data/MC mismodeling measured in control
  regions, but is built without the final fit setup. Re-optimize on the total R(D) uncertainty once
  the full Asimov fit is ready.
- A figure-of-merit (FOM) based optimization is acceptable: systematics may dominate, so the purely
  statistical R(D) uncertainty is not the target.

### A.2 Inputs

- Run2 generic MC; run1 e was partly used in the BDT training. Candidate-level samples (no MVA cut,
  no BCS) from `bco.load_candidates`. The `MC_*_loose` trees can't be used: they are already
  post-BCS after `lgb_loose`.
- MC corrections: PID weights (`CORRECTIONS = ['pid']`), one weight per candidate.
- Luminosity: MC weights × `LUMI_SCALE = 0.25` (generic MC = 4 × data), so yields, Z and σ_S/S are
  at data luminosity. Bin thresholds count raw MC events.
- Halves: optimize on half A (even event number), validate on half B. Half yields are scaled to the
  full luminosity (`HALF_LUMI_FACTOR = 2`). Control regions use the same half of their data and MC.
- Independent cross-check: run1 μ.

### A.3 Selection chain

- MVA cut → BCS (minimum `B_D_ReChi2`) → D_M signal window (1.855–1.885 GeV), as in the fit. BCS
  therefore competes candidates from the whole D_M range.
- This is the nominal chain (`bcs_order='cut_then_bcs'`), equal to lgb_loose without BCS →
  lgb_tight → BCS.
- The old chain (lgb_loose → BCS → lgb_tight, the current `MC_*_loose` trees) is reproduced with
  `bcs_order='bcs_then_cut', bcs_base=util.lgb_loose`.
- e and μ are independent channels; their Z² add.

### A.4 Statistical target

- Signal s = Dτν. Background b = the six non-physics backgrounds the BDT suppresses: fakeD,
  continuum, combinatorial, hadronicB_secondaryL, fakeL, fakeTracks. Physics backgrounds
  (D(\*,\*\*)ℓν/τν) enter neither.
- Binned FOM in the SR fit binning (`MM2_bins_sr × p_D_l_bins_sr`):
  Z² = Σ_channels Σ_bins ŝ(ŝ − 1)/(n − 1) (default `estimator='unbiased'`), or ŝ²/n (`'plain'`).
- The plain estimator carries an MC-statistics excess p(1 − p)(1 − e^−(s+b)) per bin. The default
  removes it exactly for every n ≥ 2, with a generalization for weighted MC. The derivation is in the
  module docstring and the notebook; the tests check it by exact Poisson summation.
- Only bins with ≥ 10 raw MC signal + background candidates enter (`MIN_BIN_COUNT`).
- `excess_frac` reports the share of Z² that the plain estimator would add.

### A.5 Target with control-region systematics

Each control region (CR) compares total data with total MC after the same BDT cut and BCS, in its own
2D binning: the 0.1–99.9 % quantiles of the region's BCS candidates without a BDT cut, 20 bins per
axis, with overflow counted in the edge bins. It gives two fractions:

- **Normalization** n = max(|R − 1|, σ_R), with R = data/MC. The offset is kept as an uncertainty. On
  one half: n² = max((R − 1)² − σ_R,half², 0) + σ_R².
- **Shape** δ = excess variance after scaling MC to the data total, from bins with ≥ 10 expected
  events. If too few bins pass, δ comes from the M²miss projection, or is 0.

| Region | Data / MC | Describes in the SR |
|---|---|---|
| D_M sidebands | on-resonance run2 data + run1 μ / the matching MC | fake D from BB̄ events |
| Off-resonance | run1 + run2 off-resonance data / MC; Mbc and ΔE recomputed at 10.58 GeV before the offline cut and the BDT | all qq̄ events (`B0_isContinuumEvent == 1`) |
| Combinatorial | placeholder with hand-set n, δ; not active | `bkg_combinatorial`; to be replaced by Part B |

The regions partition the SR backgrounds by event origin, so no SR event is counted twice.

**Target.** σ_S/S of the Dτν yield from the Fisher matrix of
ν_i = μ s_i + b_i + Σ_k α_k b_k,i, with a Gaussian prior of width n_k on each α_k and per-bin variance
s_i + b_i + Σ_k (δ_k b_k,i)². Single-bin form:
(σ_S/S)² = (S + B)/S² + Σ_k (n_k² + δ_k²)(B_k/S)².

- Without CRs, σ_S/S = 1/Z exactly.
- The result is broken down into stat, norm and shape parts, plus the impact of each region.

### A.6 Search and working points

- **Grids:** thresholds on grids of 35 points uniform in single-cut Dτν efficiency, from 1 down to
  0.10, rounded to 2 decimals.
- **Coordinate descent:** three starts (open, lgb_tight, lgb_loose). A check of the 80 grid neighbors
  restarts the descent if a diagonal move is better.
- **Optuna cross-check** (multivariate TPE or CMA-ES): the top 10 trials of half A are re-evaluated
  on half B, and the best one there is kept.
- **Optuna Pareto front** (NSGA-II): the statistical vs the systematic part of σ_S/S.
- **Working points:**
  - tight = the loosest point within 1σ_Z of the optimum (σ_Z from bootstrap), reached by greedy
    loosening;
  - loose = the loosest point within 5 % of the optimum.

### A.7 Choice of the final working point

Three candidates, all with the CR systematics: the coordinate descent, the Optuna cross-check, and
the minimum of the Pareto front. They were judged on the validation half B. Optuna was chosen; tight
and loose then follow from the working-point rules.

### A.8 Results

Numbers are at run2 data luminosity (≈ 125/fb), with half yields scaled to the full run2 sample. CRs:
D_M sidebands + off-resonance. S and B are the Dτν and the six-background yields in the fit bins of
the D_M signal window.

| Working point | Cut | σ_S/S, half A | σ_S/S, half B | half B: stat / norm / shape | S, B (half B) |
|---|---|---|---|---|---|
| final optimum | `sig_prob>0.56 and fakeD_prob<0.13 and continuum_prob<0.11 and combinatorial_prob<0.89` | 12.8 % | 15.1 % | 11.8 / 8.9 / 3.2 % | 1239, 36606 |
| **final tight = new `lgb_tight`** | `sig_prob>0.52 and fakeD_prob<0.13 and continuum_prob<0.11` | 12.9 % | 15.7 % | 11.8 / 9.8 / 3.1 % | 1270, 38520 |
| final loose | `sig_prob>0.52 and fakeD_prob<0.14 and continuum_prob<0.11` | 13.4 % | 16.2 % | 11.8 / 10.6 / 3.6 % | 1325, 42114 |
| previous `lgb_tight` | `fakeD_prob<0.09 and continuum_prob<0.15 and combinatorial_prob<0.35` | 13.0 % | 15.9 % | 11.8 / 10.3 / 2.6 % | 1038, 25390 |
| `lgb_loose` | `sig_prob>0.2 and fakeD_prob<0.15 and continuum_prob<0.5 and combinatorial_prob<0.6` | 16.2 % | 16.9 % | 11.3 / 10.7 / 6.7 % | 1894, 81086 |

Other findings:

- With statistics only, the optimum is much looser: `sig_prob>0.38 and fakeD_prob<0.17 and
  continuum_prob<0.57`, Z* = 9.56 ± 0.16 on half A. The CR terms move it to tighter fakeD and
  continuum cuts.
- With a bin threshold of 5, 10 or 20 events, the optimum barely moves: only the fakeD threshold
  shifts, between 0.17 and 0.21. The plain estimator would choose a different point
  (`sig_prob>0.5 and fakeD_prob<0.13 and combinatorial_prob<0.46`).
- CR measurements at the final tight cut:
  - D_M sidebands: R = 1.000 ± 0.004, δ = 6.8 % (χ²/ndf = 398/101);
  - off-resonance: R = 1.07 ± 0.03, δ = 6.8 % (χ²/ndf = 38/34).

### A.9 Diagnostics

Available in the module:

- efficiency tables per truth category, at candidate and event level;
- ROC of `sig_prob` with the working points;
- N−1 scans with the uncertainty breakdown;
- data/MC ratio in slices of one BDT output (`ratio_slices`);
- validation report (halves A/B, both BCS orders).

`ratio_slices` at the final optimum shows that the data/MC ratio depends on the BDT outputs:

| Region | Sliced output | Data/MC ratio across slices | χ²/ndf of a constant ratio |
|---|---|---|---|
| D_M sidebands | `continuum_prob` | 0.99 → 1.19 | 169/7 |
| D_M sidebands | `fakeD_prob` | 0.98 → 1.08 | 49/8 |
| off-resonance | `continuum_prob` | 1.10 → 1.56 | 20/7 |

So the MC models these outputs imperfectly in the CRs, and the offset term R − 1 partly follows the
cut.

### A.10 Tests

`scripts/tests/test_bdt_cut_optimization.py` checks:

- the fast selection and FOM against a direct pandas implementation (both BCS orders, halves, NaN
  ranks, thresholds, estimators);
- the estimator excess by exact Poisson summation, and the weighted estimator with toys;
- the CR measurements (R, n, δ, halves, auto binning, overflow);
- the Fisher target (single-bin formula, no-CR limit, luminosity scaling);
- the off-resonance beam-energy correction, efficiencies, ROC, Optuna, Pareto and the MC corrections.

### A.11 Open items from Part A

- Sculpting check of the new working point (N−1 study on candidate-level components) and the q² < 3
  data/MC check of the BDT outputs (notebook section 13).
- The BDT-output dependence of the data/MC ratio in the CRs (A.9).
- Replace the combinatorial placeholder with the Part B result; its `applies_to` must then cover
  hadronicB_secondaryL too.
- Regenerate the PID weights with the Belle II systematics framework (PID weights moved the optimum a
  lot), then re-check.
- Re-optimize on the total R(D) uncertainty once the full Asimov fit is ready.
- A global K_L correction from Part C, if one is needed, would apply to all MC.

---

## Part B: combinatorial control region and BBbar background model (design agreed; not implemented)

### B.1 Goal and use

- **AGREED.** Replace `lgb_comb` with an optimized BDT cut that selects `bkg_combinatorial` +
  `bkg_hadronicB_secondaryL` and suppresses the other backgrounds, the signal and the normalization
  modes.
- **AGREED.** The comb + hadB events selected in the CR must come from similar physics sources as in
  the SR: similar decay-type composition and a similar 2D distribution of the fit variables.
- **STATED.** Comb + hadB have large MC mismodeling, mainly from poorly simulated hadronic B decays, so
  a normsys prior alone is not enough.
  - If the CR is pure enough, it becomes a simultaneous fit channel.
  - Otherwise, the data/MC correction measured in it is applied, and its uncertainty becomes the
    normsys prior.
- **PROPOSED.** For the expected uncertainty, the two uses are equivalent as long as the prior keeps
  the correlations between the fitted parameters. In pyhf that means one normsys per eigenvector of
  their covariance, attached to all affected templates.

### B.2 Definitions

| Item | Choice | Status |
|---|---|---|
| Target sample | `bkg_combinatorial` + `bkg_hadronicB_secondaryL` | AGREED |
| Contamination | Dτν, D\*τν, D\*\*τν, Dℓν, D\*ℓν, D\*\*ℓν + gap, fakeD, continuum, fakeL, fakeTracks | AGREED (fakeL and fakeTracks come from other sources, so they count as contamination) |
| SR reference | `lgb_tight` = `sig_prob>0.52 and fakeD_prob<0.13 and continuum_prob<0.11` | AGREED |
| Cut directions | `sig_prob <`, `fakeD_prob <`, `continuum_prob <`, `combinatorial_prob >`; optional lower bound on `sig_prob` (a band next to the SR) | PROPOSED |
| Chain | MVA cut → BCS → D_M signal window, as in the SR | AGREED |
| Disjointness | at candidate level: CR = CR cut and not SR cut. Event-level overlap is accepted but measured (MC and data), with an optional veto to show its cost | AGREED |
| Binning | auto binning per region | AGREED |

Physical picture: the SR has no `combinatorial_prob` cut, so the comb CR is effectively a sideband in
`sig_prob`. Closer to 0.52 is more SR-like but more contaminated.

### B.3 Target of the CR cut optimization

**AGREED in principle:** a Fisher-based uncertainty plus a transfer term (option 3). **PROPOSED** for
the details:

- **The target** is the expected uncertainty with which the tuned model predicts the SR comb + hadB
  yield:

  σ_B/B = √(gᵀ C g) / B_SR

  - C is the covariance of the model parameters, from the Fisher matrix of the CR fit on MC
    expectations (Asimov). It includes the contamination priors and the MC statistics.
  - g is the derivative of the SR comb + hadB yield with respect to each model parameter.

  This is the number that becomes the prior in the SR fit, and it replaces the combinatorial
  placeholder of Part A.
- **Transfer term.** Composition differences that the fit can resolve cost precision, not bias. What
  it can't resolve enters a transfer term:

  σ_transfer = u × (distance between the CR and SR distributions of the unresolved quantity),

  where u is calibrated from data (B.5).

  - The distance D(X, Y) between two samples is √Σ_c (f_X,c − f_Y,c)², with f_X,c the fraction of
    the sample's comb + hadB yield in truth class c (from MC). The classes c are the composition axes
    the chosen model does not float.
  - Calibration: a cross-prediction between two samples that misses by Δ gives u = Δ / D. Then
    σ_transfer(CR → SR) = u × D(CR, SR). The same D definition is used for both, so its arbitrary
    parts largely cancel.
- **Only MC in the target.** Data enter only through the fakeD and continuum measurements of their
  own CRs, so the comb CR data are not looked at while choosing the cut.

**Contamination priors (AGREED).** Central values come from MC; fakeD is 0.87 for run1 and 1.0 for
run2, as in the tuning.

| Component | Prior width |
|---|---|
| fakeD | measured in the D_M sidebands at the CR cut |
| continuum (+ qq̄ fake D) | measured off-resonance at the CR cut |
| Dℓν, D\*ℓν | 5 % |
| D\*\*ℓν + gap | 50 % |
| Dτν, D\*τν | 15 % |
| D\*\*τν | 30 % |
| fakeL, fakeTracks | 20 % |
| BBbar_semileptonic (target sample; fixed at 1 in the current tuning) | 10 % (5 % in the first-model proposal; to settle in piece 3) |

**Priors in the tuning (PROPOSED).**

- One nuisance θ_k per constrained component. Its yield is scaled by (1 + width)^θ_k, as a pyhf
  normsys does, and a penalty θ_k² is added to the deviance.
- Optuna explores the model parameters at θ = 0; Minuit then fits all parameters.
- The HESSE block of the model parameters is the prior for the SR, already widened by the
  contamination uncertainties.

**MC statistics (PROPOSED).** Barlow–Beeston-lite in the tuning:

- one factor γ per bin on the total prediction, with a Gaussian prior of width δ = √Σw² / m;
- its best value has a closed form, γ = ½[(1 − mδ²) + √((1 − mδ²)² + 4nδ²)], so it adds no Minuit
  parameters.

In the target this means a bin variance of m + Σw². With MC at 4 × data, it adds about 25 % to each
bin's variance.

**Fit models and variables (AGREED).**

- Keep `kinematic-2d` and `kinematic-2d-plus-roe` (and `missm2-roe-2d`) as options, in the tuning and
  in the target.
- The tuning variables need not be the SR fit variables; p\*_D and p\*_ℓ separately are candidates.

### B.4 Inputs and baseline

- **Luminosity (STATED).** run1 ≈ 365/fb, run2 (partial) ≈ 125/fb, total ≈ 490/fb; generic MC is
  4 × data.
  - PROPOSED: the optimization uses run2 MC (≈ 500/fb) scaled to 490/fb, and the target models the
    template statistics of the run1 + run2 MC.
- **Nominal tuning (STATED).** `kinematic-2d-plus-roe`, ROE strength 1, ROE tail ≥ 14,
  `cap_nbody=5`, weights on both sides of comb events (`weight_ell_side=True`), no mode replacement,
  run1 + run2.
- **Current CR (STATED).** The `MC_{e,mu}_comb` / `Data_{e,mu}_comb` trees are `lgb_comb`, then BCS.
  The tuning adds `common_cut = 1.855 < D_M < 1.885 and fakeD_prob < 0.1 and sig_prob < 0.1`.
- **Baseline for comparisons (STATED):** `lgb_comb` + `common_cut`, then BCS.
  - PROPOSED: for the check against the existing HESSE errors, reproduce the trees' exact chain
    instead (`lgb_comb` → BCS → `common_cut`).
- **OPEN.** `docs/ANALYSIS_CONTEXT.md` and the August slides define the BDT sideband as
  `combinatorial_prob>0.6, sig_prob<0.1, fakeD_prob<0.1, continuum_prob<0.4`. Which definition is
  current? The context doc may need updating.

**Existing tuning result** (`best_bbbar_weights_kinematic-2d-plus-roe_alpha1_roetail14_run1run2_e_replaceFalse_minuit.json`,
e channel):

| Parameter | Fitted weight |
|---|---|
| measured hadronic | 2.10 ± 0.06 |
| unmeasured 2-body | 0.001 (at the lower limit) |
| unmeasured 3-body | 0.69 ± 0.11 |
| unmeasured 4-body | 0.001 (at the lower limit) |
| unmeasured 5+-body | 0.21 ± 0.04 |

- Correlations: −0.77 (measured vs 3-body) and −0.78 (3-body vs 5+).
- Deviance: 451 for 173 bins in 2D, and 262 for the 15 ROE bins.

**Diagnosis (AGREED).** The five class weights behave like an unregularized unfolding: classes with
similar shapes are traded against each other, and parameters end at the limits. The weights are not
physics and don't transfer between samples.

**Two physical readings (PROPOSED, piece 4):**

- PYTHIA's unmeasured decays produce too many, too soft particles.
- Data have more D's from double-charm decays (b → cc̄s) than MC.

**Measured list (STATED).** It was built from the daughter-PDG products of the ancestor B of the D
and of the lepton in comb + hadB. The top ~150 by yield were compared with PDG by hand, so coverage is
incomplete. Improving it is piece 3 (OPEN).

### B.5 The wrong-charge (WC, same-sign) sample

**STATED.** Trees `B0_wc_e` and `B0_wc_mu` (run1 and run2), with the same paths and branches as the
right-sign (RS) trees.

**Physics (explained):**

- **Same B:** the charm and the lepton always have opposite charges. So WC has essentially no signal,
  normalization or hadronicB_secondaryL.
- **Different B's:** the most common pairing, a D from b → c of one B plus the primary lepton of the
  other, is same-sign. It reaches the RS sample only through B⁰ mixing (χ_d ≈ 0.19 of neutral pairs).
  RS comb comes instead from:
  - a D plus a cascade lepton of the other B;
  - an upper-vertex D (double-charm decays) plus a primary lepton.
- So the WC ℓ side is mostly semileptonic, and its D side mostly single-charm decays. That is a
  different composition from the RS comb, as observed.

**Roles (AGREED in principle).** A sample with a different mix of the same physics adds independent
equations:

1. **Test:** fit in one sample and predict the other. The same works between the RS pure corner and
   the band next to the SR.
2. **Size the transfer term:** the remaining non-closure divided by the composition difference gives u.
3. **Constrain:** once a model passes, fit the RS comb CR and WC together.

**Plan.**

- AGREED: WC with the comb BDT cut is used in the tuning.
- AGREED: WC with the SR BDT cut is a blind verification once the model is frozen. It is the only
  comb-rich data sample in the SR's BDT region with no signal.
- PROPOSED: WC gets the same cut as the comb CR, and its own BCS.

**Earlier experience (STATED).** Hadronic-B weights fitted in one of the BDT-sideband / WC samples did
not transfer to the other. We read this as the model missing a distinction (like single vs double
charm); the same gap would bias the CR → SR transfer.

### B.6 The reweighting model

**Concepts (AGREED, piece 1).**

- A model is a rule that turns an event's MC truth into a weight, using a few parameters.
  - Tuning finds the parameter values in the CR.
  - Prediction applies the same rule, with those values, to the SR MC.
- The building blocks are:
  - class scale factors;
  - smooth functions, e.g. w = N·exp(a·(n − n̄)) in the number of daughters n;
  - for each parameter, a choice of free, with a prior, or fixed.
- Factors multiply. Each parameter is the correction of its own property, given the other
  parameters. Adding or removing a parameter can shift the fitted values of the others.
- Multiplying is exact when each parameter is attached to one step of the decay chain, because the
  generator multiplies the probabilities of those steps. A parameter defined by an event-level
  property shared by several steps (e.g. "has a true K_L") is an approximation.
- Global effects are corrected globally. The comb model holds only what is specific to comb: the B
  decays that make the D and the lepton. K_L moved to Part C.
- **Tag-side truth (STATED).** The ntuple stores the daughter lists of the ancestor B of the D and of
  the lepton.
  - For comb these are the two B's of the event.
  - For other components the tag-side truth is not saved, so B-decay parameters can't be applied to
    their tag side. Event-level quantities such as `nMC_K_L` can.

**First-model proposal (PROPOSED; pieces 2–4).**

| Parameter | Multiplies the weight of… | Treatment |
|---|---|---|
| r_meas | decays whose final state is in the measured list | prior (~15 %) |
| r_SL | decays with a lepton daughter | prior |
| N₁ | PYTHIA decays with one charm hadron | free |
| N₂ | PYTHIA decays with ≥ 2 charm hadrons (b → cc̄s) | free |
| a | all PYTHIA decays, by exp(a · (n − n̄)) | free |
| later, if needed: p\*_D slope | PYTHIA decays | free |

The truth information needed is already read: the daughter lists, `D_mcPDG`, `D_genMotherPDG` /
`D_genGMPDG` and `ell_genMotherPDG`.

**OPEN:**

- whether single- and double-charm decays share one slope a;
- regularization vs smooth functions (piece 2);
- priors and the measured list (piece 3);
- which hypotheses to test, and how (piece 4);
- which parameters the data can determine, via the Fisher matrix (piece 5).

### B.7 Workflow (PROPOSED; the order is piece 6)

1. **MC only.**
   - Store truth-based class codes when loading.
   - Composition of the RS pure corner, the RS band, WC (comb cut), WC (SR cut) and the SR comb.
   - Fisher check of which parameters each candidate model can determine.
2. **Tuning code.**
   - The CR definition is read from a file, and candidates are selected through the module (cut →
     BCS).
   - Priors, regularized parameters, Barlow–Beeston-lite, a joint RS + WC fit, cross-predictions,
     both fit models.
3. **Data.** Fit models from simple to complex, freeze the model, run the blind WC SR-cut check, and
   take u from the remaining non-closure.
4. **CR cut optimization** with the chosen model, then the final tuning and the SR prior (which
   replaces the placeholder of Part A).

In short: pre-tuning (model choice at the baseline CR) → CR cut optimization → final tuning.

**File layout (AGREED in principle).** The steps share one model definition and one code base, but
run separately:

- notebook 8 (MC only), which writes the CR definition;
- the tuning script (data);
- a validation section, or notebook 9.

The shared model definition lives in `utilities.py` (where the former `bbbar_reweighting.py` code
now is).

### B.8 Validation (PROPOSED; piece 7)

- **Cross-predictions:** RS pure corner ↔ RS band, and RS ↔ WC. The final blind check is WC with the
  SR cut.
- Pick the simplest physical model that passes. The remaining non-closure sets u.
- Before the ROE term steers the model, find out where the nROE mismatch comes from:
  - off-resonance shows detector and beam-background effects;
  - a Dℓν-rich region adds one generic B.

### B.9 Still to discuss

- Pieces 2–7 (Appendix A).
- Reconciling the BDT sideband definitions (B.4).
- The BBbar_semileptonic prior: settled in piece 3 (derived as a yield-weighted combination, not
  chosen by hand).

### B.10 Diagnosing the generator (AGREED, piece 4)

- Method: hypothesis → fingerprint → comparison. For each hypothesis, reweight the affected MC events
  (selected by truth) by a trial amount and record how each histogram changes in each sample; compare
  these fingerprints with the observed data/MC ratios before fitting anything.
- Hypotheses:
  - H1, PYTHIA multiplicity: too many, too soft particles → trend in p\*_D and nROE, similar in RS
    and WC.
  - H2, double-charm rate (b → cc̄s) → mainly normalization, larger in RS (all upper-vertex D's)
    than in WC.
  - H3, cascade leptons → trend at low p\*_ℓ, mainly in RS. May be covered by H1.
- Observables: p\*_D and p\*_ℓ separately (their sum blurs H1 and H3), nROE, M²miss.
- A parameter is kept only if it improves the fit in both RS and WC, improves the cross-predictions,
  and has a physically sensible value. What remains after the last useful parameter sets u.

### B.11 Decay-mode tables: input for the decay-table parser (B0)

**Code (DONE).** `utilities.export_bbbar_decay_mode_tables(samples)` writes six CSVs from the
`classify_mc_dict` samples: comb {charged, neutral} B × {D side, ℓ side}, and hadB {charged,
neutral} (D side only, since D and lepton share one B). One row per (canonical mode, raw daughter
order), with signed PDG codes, the candidate count, `mode_total`, `n_orders`, `n_daughters`
(photons included) and `truncated`.

**Design choices.**

- Signed PDG codes: `abs()` merges different decays (B⁺ → D⁻π⁺π⁺ vs B⁺ → D⁺π⁻π⁺) and loses the B
  flavor needed to match the right block of the decay table.
- Canonical order (by |PDG|, particle before antiparticle, empty slots last) identifies the mode; the
  raw order is kept because it carries generator information.
- **Daughter order as a PYTHIA tag (STATED):** EvtGen writes the daughters in the order of the `.dec`
  line, PYTHIA does not. A mode seen in more than one order has PYTHIA contributions; candidates in
  the `.dec` order are EvtGen or PYTHIA by chance (likely in 2-body modes). This gives at least a
  lower bound on the PYTHIA share per mode and helps with the exclusive-vs-PYTHIA ambiguity of piece 3.

**Findings from the run1 e comb CR sample (`MC_e_comb`, `lgb_comb` + BCS, no `common_cut`).**

| Table | Candidates | Distinct modes | Top 150 modes cover | Candidates in modes seen in > 1 order |
|---|---|---|---|---|
| comb charged, D side | 32,745 | 4,479 | 62 % | 31 % |
| comb charged, ℓ side | 32,745 | 5,332 | 66 % | 12 % |
| comb neutral, D side | 138,712 | 13,321 | 45 % | 52 % |
| comb neutral, ℓ side | 138,712 | 15,831 | 52 % | 32 % |
| hadB charged | 10,827 | 1,138 | 80 % | 31 % |
| hadB neutral | 26,935 | 2,014 | 79 % | 24 % |

- Checks passed: no quark or string codes and no −1 before a filled slot, so −1 always means an empty
  slot; every candidate has exactly one ancestor-B list; for comb the D- and ℓ-side B's are both
  charged or both neutral (B⁺B⁻ or B⁰B̄⁰ pairs); for hadB the D and lepton lists are identical in
  100 % of candidates.
- The top 150 modes cover only 45–66 % of the comb candidates, so a hand-made measured list can't be
  complete: automate the classification (B0).
- Many modes appear in several daughter orders (up to 52 % of the comb neutral D-side candidates):
  PYTHIA contributes widely.
- Photons among the B daughters: 16–21 % of ℓ-side candidates (final-state radiation in semileptonic
  decays) vs 3–6 % on the D side. The parser should remove photons to identify the mode and keep a
  radiative flag, consistent with `measured_pdg_rad`.
- The 7th daughter slot is filled for 4–7 % of comb candidates (< 1 % of hadB); such decays may have
  more than 7 daughters. Harmless for the n-body classes (5+), but a hidden 8th daughter could be a
  second charm hadron. See the variables notes.

**Decay table.** `DECAY_BELLE2.DEC` (release 08-00) could not be read from the cloud session (GitHub
blocks automated access); read it locally in Claude Code or attach it. Check first that this is the
decay file used to produce MC16rd.

---

## Part C: global K_L check (AGREED as a separate track)

### C.1 Why global

- K_L come from D decays, from fragmentation and from B decays. So every component is affected:
  signal, normalization, fakeD, continuum, combinatorial, hadronicB_secondaryL, fakeL and fakeTracks.
- An escaping K_L adds missing mass. A Dℓν event with a tag-side K_L, for example, moves to higher
  M²miss, toward Dτν. A K_L problem would therefore matter for the SR fit itself.
- So K_L is not a parameter of the comb model.

### C.2 Sources, and which sample sees them

| Source | On-resonance (all components) | Off-resonance |
|---|---|---|
| D decays (c → s, D → K_L X) | yes | yes (cc̄ events) |
| strange-quark production in fragmentation | yes | yes |
| detector response to a K_L (partial calorimeter deposits entering the ROE) | yes | yes |
| B-decay specific: b → cc̄s strangeness, B → D K⁰ X, kaon content of PYTHIA B decays | yes | no |

Most kaons in B decays come from c → s, so the first row is probably the largest source. A K_L
effect seen on-resonance but not off-resonance would point to the last row.

### C.3 Why q² < 3 alone isn't enough

- In q² < 3, Dτν is forbidden (q² < m_τ²). There, MC overshoots data at M²miss > 0 (August slides),
  which is attributed to D\*ℓν / D\*\*ℓν.
- Too many K_L in MC would give the same symptom. Without a direct K_L observable
  (`B0_nROE_NeutralHadrons_my_mask` was dropped as mismodeled), that sample alone can't separate the
  two explanations.
- Combining samples with different mixtures can:

| Sample | K_L sources | Signal side |
|---|---|---|
| off-resonance | D decays, fragmentation, detector; no B decays | no D\*/D\*\* feed-down |
| q² < 3 | generic tag-side B (all sources) | mostly Dℓν |
| D\* veto region | the same tag side | mostly D\*ℓν |

### C.4 Steps

**K1. Sensitivity (MC only).**

- Reweight every MC event by (1 + ε)^`nMC_K_L`, with ε = 10–20 %. A plausible size comes from
  comparing the generator's inclusive D → K⁰ X rates with the PDG values.
- Compare the M²miss shapes in the SR, q² < 3, the D\* veto region and off-resonance, and the shift
  of the Asimov R(D).
- If the shift is small compared with the other systematics, assign it as a systematic and stop.
  Otherwise go to K2.

**K2. Data.**

- Fit off-resonance, q² < 3 and the D\* veto region together, with one K_L factor and one D\*/D\*\*
  feed-down factor.
- The same K_L factor everywhere means a global effect. A difference between on- and off-resonance
  means it is specific to B decays.

**K3. Output.** One of:

- a global MC correction, registered next to PID in the MC corrections and applied to all components
  wherever MC is loaded;
- a systematic variation on all templates.

### C.5 Connections to the other parts

- **Part B:** the comb model has no K_L parameter. K1 runs before the comb model is fitted to data,
  so those fits don't absorb a K_L effect. A K3 correction is a fixed input to the comb tuning, like
  PID.
- **Part A:** with a correction, the D_M-sideband and off-resonance measurements are redone.
- **SR fit:** the correction or the systematic applies to every template.
- **The q² < 3 overshoot at M²miss > 0:** K2 tells how much of it is K_L rather than D\*/D\*\*.
- **Code:** keep `nMC_K_L` when loading (the module's list of kept columns).

### C.6 Limitations and ntuple variables

- The reweighting changes how many K_L there are, not how the detector responds to each one. With the
  current ntuples, the detector side can only be probed through its overall effect off-resonance.
- `nMC_K_L` is enough for K1. STATED: it counts all K_L in the event, regardless of acceptance.
- STATED: extra K_L variables will be added to the reconstruction only if K1 shows that K_L matters.
- If K1 shows that K_L matters, add these in a single ntuple production:
  - the K_L ancestry (mother and grandmother PDG), to attribute the source and attach a correction to
    the right decay step;
  - momentum and polar angle;
  - if possible, whether it produced a calorimeter cluster that entered the ROE, and that cluster's
    energy.
- Truth E or p alone adds little.

---

## Notes: variables that may be needed later

| Variable | Needed for | When to add |
|---|---|---|
| `nMC_K_L` (kept at load time) | K_L sensitivity (C1) | now; already in the ntuples (counts all K_L in the event) |
| `B0_nROE_Tracks_my_mask` (kept at load time) | nROE term of the tuning, H1 fingerprint | now; already in the ntuples |
| `D_CMS_p`, `ell_CMS_p` separately (kept at load time) | tuning variables, H1/H3 fingerprints | now; already in the ntuples |
| `ell_genMotherPDG`, `D_genMotherPDG`, `D_genGMPDG`, daughter lists → truth class codes at load time | composition study, model parameters | now; already in the ntuples |
| K_L ancestry (mother / grandmother PDG) | attribute K_L to its source, correct the right decay step | only if C1 shows K_L matters |
| K_L momentum and polar angle | K_L spectrum and detector-response studies | only if C1 shows K_L matters |
| calorimeter cluster from the K_L entering the ROE (flag, energy) | K_L detector response | only if C1 shows K_L matters; if feasible |
| number of daughters of the ancestor B (or more than 7 daughter slots) | detect truncated daughter lists (7th slot filled in 4–7 % of comb candidates); a hidden 8th daughter could be a second charm hadron | low priority; with the next reprocessing |
| daughter list of the ROE's (tag-side) B ancestor | apply B-decay parameters to the tag side of all components | only if the tag-side check (piece 7) matches the H1 fingerprint |

New reconstruction variables are requested together, in one reprocessing, because reprocessing is
the expensive part.

## Reasoning behind the main choices

Short motivation for each decision, so the "why" survives with the "what".

**Part A**

- *Unbiased FOM estimator:* the plain s²/n gains p(1−p) per bin from MC fluctuations alone, which
  moves the optimum; the factorial-moment form removes this exactly.
- *Optimize on half A, choose on half B:* every search picks its own best fluctuation; only an
  independent half can compare methods fairly.
- *CR systematics in the target:* the statistical FOM alone prefers loose cuts that keep poorly
  modeled backgrounds; with n and δ the optimum moves to where the MC can be trusted.
- *n = max(|R−1|, σ_R):* the measured offset already contains the statistical scatter; adding σ_R
  again would count it twice.

**Part B**

- *Target = uncertainty of the SR comb prediction:* the CR exists to predict the SR, so it should be
  judged by how well it does that, not by its purity alone.
- *MC-only target:* choosing the cut without looking at CR data keeps the later data fit unbiased.
- *Priors on contaminations:* with zero width, a mismodeled contamination is silently absorbed into the
  BBbar weights.
- *Wrong-charge sample:* a sample with a different mix of the same physics adds equations; a model
  that only works in one mix is fitting the sample, not the physics.
- *Parameters attached to decay steps:* the generator multiplies the probabilities of each step, so
  such corrections multiply exactly and transfer between samples.
- *Regularization:* classes with similar shapes are degenerate; unregularized weights trade against
  each other (the 2.1 and the zeros) and do not transfer.
- *Priors from the decay table and PDG:* measured modes are known; letting them float turns them into
  shape knobs.
- *Fingerprints before fits:* comparing predicted patterns with existing data/MC plots identifies the
  missing physics without tuning anything on data.

**Part C**

- *K_L outside the comb model:* K_L affects every component through the ROE; fitting it in the comb CR
  would make the comb parameters absorb a global effect.
- *Several samples for K_L:* in q² < 3 alone a K_L excess looks like D\*/D\*\* feed-down; samples with
  different mixtures separate them.
- *Sensitivity first (K1):* it decides cheaply whether the data study and new ntuple variables are
  worth doing.

## Appendix A: discussion pieces

| # | Piece | Your questions | Status |
|---|---|---|---|
| 1 | What a model and its parameters are | Q4 | AGREED, including overlapping factors; K_L moved to Part C |
| 2 | Why the current parameters misbehave; regularization | Q3.2 | discussed (mostly clear; revisit if needed) |
| 3 | Priors: role in the fit, where the numbers come from; improving the measured list | Q3.1, Q2 | AGREED: decay-table parser + automatic PDG matching (PDG Python API) to classify modes (exclusive measured / exclusive estimated / PYTHIA), get BF_MC and per-mode c_k ± σ_k; r_meas and r_SL priors as yield-weighted combinations. Scripts to be written together when coding starts (decay-table format and ancestor-daughter truth variables need care) |
| 4 | Where the generator might be wrong, how to test it, how RS and WC help | Q3.3, Q5 | AGREED (B.10) |
| 5 | The Fisher matrix, for both the tuning design and the cut optimization | Q6 | AGREED: one Fisher tool; Use A = model design (cut fixed), Use B = CR cut optimization (model fixed); first validated against the existing HESSE errors |
| 6 | Order of work: pre-tuning → cut optimization → final tuning | Q1 | AGREED: the cut target needs a model (parameters) and u, the model needs data in a CR → pre-tune at the baseline CR, optimize the cut (MC only), final tuning in the new CR; the final tuning re-tests the model, and if it fails, one more iteration from B5 |
| 7 | Validation: cross-predictions, WC with the SR cut, tag side | Q7 | AGREED: validation ladder V1–V5 with pass criteria fixed beforehand; u = Δ / D; tag-side variables only if the check in the normalization-rich regions matches the H1 fingerprint |

## Appendix B: key cuts and samples

| Name | Definition |
|---|---|
| `lgb_tight` (SR) | `sig_prob>0.52 and fakeD_prob<0.13 and continuum_prob<0.11` |
| `lgb_loose` | `sig_prob>0.2 and fakeD_prob<0.15 and continuum_prob<0.5 and combinatorial_prob<0.6` |
| `lgb_comb` | `fakeD_prob<0.15 and continuum_prob<0.5 and combinatorial_prob>0.5` |
| `common_cut` (tuning) | `1.855 < D_M < 1.885 and fakeD_prob < 0.1 and sig_prob < 0.1` |
| D_M signal window | `1.855 < D_M < 1.885` |
| D_M sidebands | `D_M < 1.85 or 1.9 < D_M` |

| Sample | Trees |
|---|---|
| right-sign, candidate level | `B0_e`, `B0_mu` |
| wrong-charge | `B0_wc_e`, `B0_wc_mu` |
| current comb CR (tuning input) | `MC_{e,mu}_comb`, `Data_{e,mu}_comb` in `Samples/4S_{run}_deimos_BDT_{e,mu}_3.root` |
