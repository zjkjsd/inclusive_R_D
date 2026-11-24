# B0 piece 4: priors and how the corrections enter the model

Review of how the corrections of pieces 2 and 3 (the per-line c, the classes N₁, N₂, r_s, and the
semileptonic parameters) enter the BBbar background model (step B0, piece 4 of
`docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`, section B.13; model parameters in B.6). Same conventions as
`docs/B0_PIECE3_REVIEW.md`: every point has the facts, the options, a recommendation and a decision; a
decision is entered only after it was discussed and agreed. Status labels: OPEN, AGREED (decided, not
implemented), DONE (implemented, tables rerun).

Last updated: 2026-11-24 (file opened; order of the points agreed on 2026-11-22; 4.1 decided).

Line numbers refer to `Samples/dec/DECAY_BELLE2_rel08.DEC`. Semileptonic rates are per light lepton,
in % of the decays of that B flavor.

## 1. Decisions

| # | Point | Status | Decision and current values |
|---|---|---|---|
| 4.1 | How the corrections enter: event weights, prior widths, or both; the total B rate (unitarity); starting values of N₂, r_s (open from 3.2) | AGREED (2026-11-24), section 3.1 | (a1) every c is applied as a nominal event weight in all samples (CR, WC, SR); (b3) the σ_c are not free in the tuning: toys of all c (with the 4.2 correlations) are propagated through the linearized tuning into the SR prediction, and the resulting covariance enters the SR prior as a few eigen-NPs (to be shared with the SR physics templates at B9); a Fisher run with the leading ~20–50 BFs as constrained NPs is a diagnostic in step B3; (c1) no unitarity rescaling: N₁, N₂ are absolute rate scales, N₁ free; the fitted N₁ is compared with k = 1.11 (B⁰) / 1.14 (B⁺); (D) starting values = prior centres: N₂ = 0.93, r_s = 1.00, N₁ = 1, a = 0 |
| 4.2 | Correlations between corrections (e/μ, isospin-derived c, BaBar 2011 D D̄ K, the LHCb D_s externals, shared sub-decay BFs, node remainders, D\* BFs) | OPEN (2026-11-24), section 2.3 | – |
| 4.3 | Isospin parameter for PYTHIA single-charm decays (3.5) | OPEN | – |
| 4.4 | Composition nuisance for D(\*)D̄(\*)K\* (3.4) and the S042.609 closure check | OPEN | – |
| 4.5 | The D₁(2420) decay table (3.7): event weight on `D_genMotherPDG` | OPEN | – |
| 4.6 | r_SL: one parameter or a split; τ modes; D\*\* targets; normalization-mode values | OPEN | – |
| 4.7 | Semileptonic gap modes (Belle II recommendation) | OPEN | – |
| 4.8 | Form-factor reweighting (HAMMER): in this model or a global correction | OPEN | – |

Order agreed on 2026-11-22: 4.1 first (every later point is written in its terms), then 4.2, the
hadronic points 4.3–4.5, then the semileptonic points 4.6–4.8 (they wait for the recommendation and
are the larger yield: 45–77 % of the comb ℓ side is semileptonic).

## 2. Facts

### 2.1 Belle II notes read for piece 4 (2026-11-22, 2026-11-23)

All are Belle II internal documents (copies kept privately, not in this repository; `docs/REFERENCES.md`
gets a row when a decision rests on one). Each was read in full by a reader and checked page by page
by an independent checker; the checked extractions were kept in the session scratchpad.

| Note | MC | Useful for |
|---|---|---|
| BELLE2-NOTE-PH-2026-035 v8 (1 Nov 2026), B → D(\*)ηℓν, hadronic FEI | MC16rd, release 08 (its decay-file columns equal ours) | 4.1 (nominal weights + SysVar eigen-NPs, PYTHIA at 1 ± 100 %), 4.2 (PDG D\* correlation matrix), 4.6–4.8 (Table 4.4: current SLME D\*\* and gap values; dedicated MC16rd gap samples; HAMMER BLPRXP/BLR), D-decay c (Table 4.3), release-08 global tags |
| BELLE2-NOTE-PH-2025-031 v4.2 (8 Dec 2025), R(D\*) and P_τ, hadronic FEI (parts 1, 2, 4) | MC15rd, release 06 | 4.1 (SysVar: 500 weight variations, PCA, normsys + histosys with ρ = 1 for BFs; 20 + 24 NPs for single/double-charm BFs; BBbar reweighting dropped after WG review Q20 in favour of CR normalizations), 4.6 (τ via theory R(D\*\*), τ gap 0.32 %), 4.7 (Sec. 3.2.3, Tables 5–7; gap prior 50 % vs 100 %, Q84) |
| BELLE2-NOTE-PH-2025-051 v0.4, B → X_sγ, hadronic FEI (file 2) | MC15rd, release 06 | 4.1 (per-mode ±1σ templates; "unmatched" class rescaled to conserve the BB̄ count (≈ 1.2), free normalization plus one shape NP; matched/unmatched normalizations correlated about −0.8); its "79 NPs" does not match its tables (22 + 19 non-zero modes) |
| BELLE2-NOTE-PH-2023-037 v5.04 (9 Aug 2023), inclusive R(X_τ/ℓ), R(X_e/μ) | MC14ri_a | 4.7 (gap via broad D\*\*: efficiency +40–80 % for D(\*)ππℓν vs phase space; M_X peak 2.15 vs 2.6 GeV; data pull the gap down), the 2022 version of the semileptonic recipe (Table 6.2, its decay-file column equals our release 08), secondary and fake leptons (wrong-charge sideband; MC low by 7–31 %, p_ℓ dependent), inclusive D-decay composition (D⁺ → K⁻X 30.5 % vs PDG 25.7 ± 1.4; D⁰ → K⁰X 39.9 vs 47 ± 4; for Part C and the cascade-lepton hypothesis H3, not piece 4) |

The semileptonic recipe the notes follow is one page, "Correction of BF(B → Xcℓν) in generic MC"
(F. Bernlochner; Confluence 2022 in the R(X) note, xwiki in the R(D\*) note). Its versions, B⁰:

| Component | decay file rel08 | 2022 (R(X) note) | HFLAV 2023 based (R(D\*) note) | 2025/26 SLME (D(\*)ηℓν note) | ours now (PDG 2026 c) |
|---|---|---|---|---|---|
| D ℓν | 2.14 | 2.24 ± 0.07 | 2.11 ± 0.05 | (PDG 2025 2.10) | 2.12 |
| D\* ℓν | 5.11 | 5.11 ± 0.11 | 4.90 ± 0.11 | (4.87) | 4.90 |
| D₀\* ℓν | 0.362 | 0.39 ± 0.07 | 0.12 (error printed 0.26, a typo for ≈ 0.03) | 0.08 ± 0.04 | 0.125 |
| D₁′ ℓν | 0.401 | 0.39 ± 0.08 | 0.26 ± 0.04 | 0.26 ± 0.04 | 0.374 |
| D₁ ℓν | 0.704 | 0.62 ± 0.10 | 0.59 ± 0.10 | 0.59 ± 0.10 | 0.675 |
| D₂\* ℓν | 0.347 | 0.27 ± 0.03 | 0.299 ± 0.027 | 0.299 ± 0.027 | 0.292 |
| D(\*)πℓν non-resonant | 0.276 | not listed | not listed | 0 | 0.276 (1 ± 1) |
| Dππℓν | 0.049 | 0.06 ± 0.09 | 0.07 ± 0.08 | 0.07 ± 0.08 | 0.057 |
| D\*ππℓν | 0.245 | 0.20 ± 0.10 | 0.20 ± 0.10 | 0.20 ± 0.10 | 0.191 |
| Dη + D\*ηℓν | 0.434 | 0.82 (100 %) | 1.72 (22 %) | 1.76 (100 %) | 0.434 |
| sum | 10.07 | 10.10 | 10.27 | 10.27 | 9.44 |

- The 2025/26 values close the sum at 10.27 % (B⁰; 10.99 % B⁺), an inclusive value derived from HFLAV;
  Dη and D\*η take (10.27 − 8.23 − 0.07 − 0.20)/2 = 0.885 each. HFLAV 2023 (arXiv:2411.18639) itself
  quotes B(B → Xcℓν) = 10.63 ± 0.15 % (eq. 106) and has no D(\*)ππℓν or D(\*)ηℓν average.
- **Abnormal:** with the current PDG-2026 c our B⁰ sum is 9.44 % (B⁺ 10.28 %), against PDG 2026
  B⁰ → X_c eν 10.08 ± 0.37 % (S042.375; B⁺ 10.79 ± 0.37, S041.473): the D\*\* lines are lowered to PDG
  while the gap lines keep their decay-file values. For 4.6/4.7.
- D\*\* targets differ from our PDG-product c (HFLAV isospin average vs PDG per flavor), largest for
  B⁰ D₁′ (PDG product 0.249 ± 0.055 % against HFLAV 0.185 ± 0.029 %). For 4.6.
- τ: decay-file D\*\*τν 0.66 % (lines 420–423) against 0.10–0.14 % in every recipe; no τ gap lines in
  the decay file (recipes: 0.12–0.32 %). τ → eνν, μνν in the decay file (block from line 8046):
  0.178175, 0.173937 = PDG; no correction needed. For 4.6.
- Normalization modes: B⁺ D̄\*⁰ℓν is 5.53 ± 0.22 % in PDG 2026 and HFLAV 2023 (direct average),
  5.27 in the R(D\*) note (B⁰ × lifetime ratio), 5.26 in the D(\*)ηℓν note (PDG 2025). For 4.6.

### 2.2 Facts for 4.1 (2026-11-24)

How the model is used (B.1, B.3, B.7): the model is tuned in the comb CR (and WC) by a Poisson
deviance fit (Minuit; priors as normsys-like penalties; Barlow–Beeston-lite), and the HESSE covariance
of its parameters becomes the SR prior (one normsys per eigenvector in pyhf). So a correction can
enter at two places: in the tuning (fixed weight, or a constrained parameter) and in the SR prior.

Sizes, from the 36 mode tables (`bbbar_modes_*.csv`, e + μ, run1 + run2, comb + hadB, PID weights),
first order in (c − 1); comb modes count on both sides (fractions of the comb + hadB yield).
Script in the session scratchpad (`p41_facts.py`, `p41_group.py`).

| | comb CR | SR | WC |
|---|---|---|---|
| yield change when all c are applied | −10.3 % | −9.6 % | −11.5 % |
| of which hadronic measured / external / semileptonic | −5.2 / −3.3 / −1.7 % | −5.2 / −3.3 / −1.0 % | −3.9 / −3.7 / −3.7 % |
| yield uncertainty from all σ_c (physical BFs independent; e/μ and conjugates merged) | 1.3 % | 1.3 % | 1.7 % |
| same, LHCb externals fully correlated instead of independent (external part; 0.17 % is with B and anti-B rows independent, 0.23–0.25 % with conjugates merged; corrected 2026-11-24, section 2.3) | 0.72 % instead of 0.17 % | 0.74 % instead of 0.17 % | 0.78 % instead of 0.17 % |

Transfer (the ratio of two samples, once a common normalization is free):

| | CR → SR | CR → WC | SR → WC |
|---|---|---|---|
| change of the ratio from the nominal c | +0.7 % | −1.2 % | −1.9 % |
| uncertainty of the ratio from σ_c (independent physical BFs) | 0.4 % | 0.8 % | 1.0 % |
| physical BFs with a σ_c in the samples | 326 | 359 | 316 |
| BFs carrying 90 % / 99 % of that variance | 18 / 51 | 16 / 45 | 17 / 40 |

- Largest contributions CR → SR: B⁰ → D⁻τ⁺ν (c 1.13 ± 0.23), D_s\*⁺D⁻ (1.00 ± 0.21), D̄\*⁰π⁻ℓν (1 ± 1),
  D\*⁻a₁⁺, D⁺D⁻K⁰, D⁻a₁⁺. CR → WC: B⁰ → D\*⁻ℓν (0.959 ± 0.023, 17 % more of the WC yield), the
  non-resonant D(\*)πℓν lines (1 ± 1), D₁′ℓν, D₀\*ℓν.
- Classes with a model parameter (N₁, N₂ with r_s, PYTHIA) carry 60–80 % of the yield: PYTHIA single
  charm 43–60 %, unmeasured b → cc̄s (exclusive + PYTHIA) 25–38 %. The N₂ prior alone (± 13 % on about
  a third of the yield) is about ± 4–5 % of the yield, several times the σ_c total.
- Unitarity: all c applied, the exclusive lines lose **5.5 % (B⁰) and 6.0 % (B⁺) of all B decays**
  (semileptonic −1.0 / −0.6 %, LHCb externals −1.5 / −1.4 %). B.13 still quotes 3.6 / 4.3 % (before
  3.4); to be updated. Giving this rate back to PYTHIA would scale it by k = 1.114 (B⁰), 1.138 (B⁺)
  (the B → X_sγ note: ≈ 1.2).
- The N₂, r_s prior of 3.2 is absolute: it was derived with the measured lines after c and PYTHIA at
  weight 1, so it assumes no unitarity rescaling.

### 2.3 Facts for 4.2 (2026-11-24)

Collected by four independent readers, each checked by a second one: (i) an inventory of the sources
shared by construction in our c table, (ii) BaBar 2011 D(\*)D̄(\*)K (arXiv:1011.3929), (iii) PDG-level
correlations and common sub-decay normalizations (PDG 2026 printed listings, pdg 2026.0 SQLite),
(iv) the 20 LHCb externals of 3.4. Scripts and outputs in the session scratchpad (`p42/`). Numbers are
first order, from the 4.1 leverages; "transfer" = the CR → SR and CR → WC ratios with a common
normalization free; baseline = e/μ and conjugates merged, every physical BF independent:
**CR → SR 0.38 %, CR → WC 0.81 %, yield 1.26 / 1.34 / 1.66 % (CR / SR / WC).**

| Source | Members | Kind | Transfer CR → SR / CR → WC | Yield σ CR / SR / WC |
|---|---|---|---|---|
| e and μ of one PDG lepton entry | 118 rows (B⁰ 366/390 … 383/407, 830/831, 835/836; B⁺ 3998/4025 … 4377/4378) | full (same entry) | 0.36 / 0.63 if independent → 0.38 / 0.81 | 36 / 33 / 71 % of the yield |
| charge conjugates | all anti-B rows (cp_transfer: 5888, 5892) | full | 0.27 / 0.57 if independent → 0.38 / 0.81 | 0.89 / 0.95 / 1.17 → 1.26 / 1.34 / 1.66 |
| isospin copies (B⁰ D₀\*ℓν 369/393 from 4001/4028; ψ(3770)K⁰ 592–594 from 4162) | 2 groups | full | CR → WC +0.013 | small |
| one PDG entry for several final states (K⁰/K_S/K_L splits, flavor twins: 46 shared entries) | e.g. 428–430 S042.23 | full | < 0.001 | < 0.01 |
| node remainders (direct line = total − fed lines) | 89 nodes, 58 with fed lines in the toys; ρ e.g. −0.37 (652), −0.61 (658), −0.70 (4216), −0.27 (4210) | anti-correlated with the fed lines; positive between remainders sharing a feeder | −0.003 / −0.005 | small |
| D_s0\*(2317) products, common 20 % (B(D_s0\* → D_sπ⁰) = 100 +0 −20 %) | 790, 791, 4339, 4340 | full on the 20 % | +0.0 / +0.005 | 0.12 → 0.15 % (group) |
| LHCb externals (20 lines + conjugates) | 623–632, 4182–4192 | components: S042.49 9.8 % (shared with line 620, D\*⁻D_s⁺), R_LHCb 6.7 %, fit fractions per D\*\* state, D₁ fraction 10 %, R\* 12 %, bridge 30 % (8 three-body lines), D_s\*/D_s | ±0.015 | group 0.23 (indep.) → 0.44 (components) → 0.72 (ρ = 1) |
| **all construction correlations together** | | | **0.38 / 0.82** | **1.34 / 1.41 / 1.74** |
| BaBar 2011 D(\*)D̄(\*)K, 22 entries (646–661, 4204–4219; 20 equal to BaBar alone) | 12.6 / 14.1 / 7.7 % of the yield | common component = Table III columns (i) detection/PID and (j) D(\*), K_S BFs + B counting: 4.5–8.4 % per mode (≈ 6 %), rest independent; reproduces BaBar's summed systematic (24.9 / 28.7 vs 24 / 28 ·10⁻⁴) | 0.38 / 0.85 | group 0.44 → 0.81 % |
| PDG constrained-fit correlation matrices (printed listings only; B⁰ 24 parameters, B⁺ 13; 27 of our c) | largest B⁰ D\*ℓν–D\*τν 40 %, B⁺ 33 % | partial | < 0.01 | +0.02–0.03 |
| B(D_s → φπ) rescaling (8.9 %) inside PDG D(\*)D_s(\*) averages (common fraction e.g. 4.5 % S042.50, 5.5 % S042.49) | 11–13 % of the yield | full on that part | 0.39 / 0.82 | 1.36 / 1.47 / 1.70 |
| D⁰, D⁺, D\*, D_s sub-decay normalizations (0.4–1.7 % each) | every D-based c | full per species | unchanged | 1.47 / 1.55 / 1.86 |
| Υ(4S) f₀₀ / f₊₋ (measurements assuming equal production; +2.9 % / −2.2 %, B⁰ against B⁺) | all c | anti-correlated B⁰ / B⁺ (upper estimate: modern ones already rescaled) | 0.38 / 0.85 | +0.6–0.85 |
| B⁰ / B⁺ D(\*)ℓν share inputs (HFLAV B⁰ D\*ℓν 4.90 % assumes isospin; same analyses in both) | 366/390, 367/391, 3998/4025, 3999/4026 | ρ 0.5–1 | 0.38 / 0.85–0.98 | WC 1.74 → 1.8–2.1 |
| D\*\*ℓν B⁰ / B⁺ (same analyses: Meier 2023, BaBar 2009Y simultaneous fit, …) | 368–371, 4000–4003 (+ μ) | partial positive | not computed; expected a few 0.01 on CR → WC | – |

Reading: correlations change the **absolute** yield uncertainty (1.3 → up to about 1.9 %) but hardly
the **transfers** (CR → SR stays 0.38–0.40 %; CR → WC 0.81 → at most about 0.98 %), because a common
component cancels between samples with a similar composition. The B⁰/B⁺ D(\*)ℓν, f₀₀/f₊₋ and BaBar
common terms matter for CR → WC (the WC ℓ side is 70 % semileptonic, mostly B⁰ D\*ℓν).

Abnormal findings (2026-11-24, to decide):
- **The 0.72 "non-D₂\* part" of the three-body D_s Dπ externals** (exception of line 4189 and its
  family): 1 − FF(D₂\*) = 1 − 0.222 = 0.778 (0.736 if the other small components of arXiv:2212.02717
  Table 6 are removed as well); (0.80 ± 0.19)/(1.11 ± 0.24)·10⁻³ = 0.72 in 3.7 is not reproduced. If
  0.778 is right, the c of the 8 three-body lines (16 rows, about 2.8 % of the yield) are 8 % too low
  (about +0.2 % of the yield). To check against the 3.7 derivation.
- BaBar 2011 normalized its D(\*)D̄(\*)K to the PDG 2008 D branching fractions; with PDG 2026 values
  all 22 c would move together by an estimated −2 to −5 % (PDG 2008 values not verified).
- S042.266 (B⁰ → D\*⁻D\*⁺K⁰) averages BaBar 2011 with BaBar 2006, a subset of the same data that 2011
  supersedes: c(655) 0.969 ± 0.081 against 0.988 ± 0.099 with BaBar 2011 alone.
- LHCb ratios between D(\*)D̄(\*)K modes (in the listings, not in the PDG averages) are about 2.3σ
  from BaBar's own ratios (e.g. lines 646/648).
- LHCb arXiv:2405.00098 prints R\* = 1.31 in the abstract and 1.32 ± 0.07 ± 0.14 in the body (we use
  1.32).
- The 3.2 refit of N₂, r_s held the externals fixed; through the D_s X constraint r_s is
  anti-correlated with them (estimated σ(r_s) about 0.07 from this, ρ about −0.3; refit not re-run).

## 3. Discussion record

### 3.1 How the corrections enter (opened 2026-11-24)

Facts: section 2.2.

*Questions and options (presented 2026-11-24).*
- (A) Nominal values: (a1) apply c as event weights in every sample; (a2) keep c = 1 and only widen
  the priors.
- (B) The σ_c: (b1) not propagated, left to the data-calibrated transfer term u; (b2) the ~20–50
  leading BFs as Gaussian-constrained NPs in the tuning, the rest lumped; (b3) c fixed in the tuning,
  toys of all c propagated through the linearized tuning (the Fisher gradients of B.3) into the SR
  prediction, covariance added to the SR prior as eigen-NPs.
- (C) Unitarity (the exclusive lines lose 5.5 / 6.0 % of B decays when all c are applied): (c1) no
  rescaling, N₁ free absorbs the rate; (c2) rescale PYTHIA by k = 1.114 / 1.138 before the fit (as
  the B → X_sγ note), N₂ prior re-expressed as 0.93/k; (c3, raised in the discussion) Σ BF = 1 per B
  flavor as a constraint that fixes N₁ from the other parameters.
- (D) Starting values: N₂, r_s at their prior centres.

*Questions answered in the discussion (2026-11-24).* (1) Unitarity: the generic MC is a fixed number
of BB̄ pairs, so lowering measured lines without raising anything describes a world where 5.5 % of
the B⁰ decay into nothing; the rate must go to unmeasured decays (unmeasured exclusive + PYTHIA),
since measured ones are pinned. (2) The N₂ prior is absolute: it constrains the sum of the unmeasured
b → cc̄s decays through inclusive upper-vertex rates (BaBar 2007 D_s⁺X and wrong-sign D X per flavor,
Belle 2023 D_s X) = measured lines after c + N₂ × (unmeasured exclusive + PYTHIA share from generator
truth); it does not constrain the split among the unmeasured modes (4.4). (3) c covers the exclusive
lines (exact, product, node, isospin, external, and the 1 ± 1 / limit conventions), semileptonic lines
included; not the N₁/N₂ lines, PYTHIA, X_u ℓν, D decays or resonance sub-decays.

**Decision (2026-11-24): a1, b3, c1 and D, AGREED.** Reasons: the nominal c move the yield by about
10 % and the CR → SR / CR → WC composition by about 1 %, so they must be applied (all four Belle II
notes do); their σ add only 0.4–0.8 % to a transfer, and keeping them out of the tuning stops measured
BFs from acting as shape knobs (B.6) while toys carry the correlations of 4.2 and give eigen-NPs that
can be shared with the SR physics templates; the N₂ prior was derived without a unitarity rescaling,
so c1 keeps it valid, and the fitted N₁ against k is a physics check; starting at the prior centres
costs nothing. Nothing to implement now: (a1) is already how the c are meant to be used; (b3) belongs
to the tuning code (B4) and the hand-over (B9); the Fisher diagnostic to step B3.
