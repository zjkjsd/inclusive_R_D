# B0 piece 4: priors and how the corrections enter the model

Review of how the corrections of pieces 2 and 3 (the per-line c, the classes N₁, N₂, r_s, and the
semileptonic parameters) enter the BBbar background model (step B0, piece 4 of
`docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`, section B.13; model parameters in B.6). Same conventions as
`docs/B0_PIECE3_REVIEW.md`: every point has the facts, the options, a recommendation and a decision; a
decision is entered only after it was discussed and agreed. Each point in section 3 opens with a short
*context* paragraph: what problem the point solves and where it sits in the model (rule agreed
2026-11-25). Status labels: OPEN, AGREED (decided, not implemented), DONE (implemented, tables rerun).

Last updated: 2026-11-26 (4.5 decided and its line c implemented; follow-up on other excited states).

Line numbers refer to `Samples/dec/DECAY_BELLE2_rel08.DEC`. Semileptonic rates are per light lepton,
in % of the decays of that B flavor.

## 1. Decisions

| # | Point | Status | Decision and current values |
|---|---|---|---|
| 4.1 | How the corrections enter: event weights, prior widths, or both; the total B rate (unitarity); starting values of N₂, r_s (open from 3.2) | AGREED (2026-11-24), section 3.1 | (a1) every c is applied as a nominal event weight in all samples (CR, WC, SR); (b3) the σ_c are not free in the tuning: toys of all c (with the 4.2 correlations) are propagated through the linearized tuning into the SR prediction, and the resulting covariance enters the SR prior as a few eigen-NPs (to be shared with the SR physics templates at B9); a Fisher run with the leading ~20–50 BFs as constrained NPs is a diagnostic in step B3; (c1) no unitarity rescaling: N₁, N₂ are absolute rate scales, N₁ free; the fitted N₁ is compared with k = 1.11 (B⁰) / 1.14 (B⁺); (D) starting values = prior centres: N₂ = 0.93, r_s = 1.00, N₁ = 1, a = 0 |
| 4.2 | Correlations between corrections (e/μ, isospin-derived c, BaBar 2011 D D̄ K, the LHCb D_s externals, shared sub-decay BFs, node remainders, D\* BFs) | AGREED (2026-11-25), sections 2.3, 3.2 | option (b): each c is written as a sum of named sources and the toys draw the sources: construction correlations (one variable per physical BF for e/μ, conjugates, shared PDG entries, isospin copies; node remainders and LHCb externals recomputed from their drawn inputs) plus the documented common components (BaBar 2011 common ≈ 6 % per mode, LHCb component model, D_s0\* 20 %, φπ rescaling fractions, PDG fit matrices, D-species sub-decay normalizations, one f₀₀/f₊₋ term, B⁰/B⁺ semileptonic correlation with its value set in 4.6); r_s drawn together with the externals; sources also present in the SR fit shared with it at B9. Effect: transfer CR → SR about 0.38–0.40 %, CR → WC about 0.8–1.0 %; yield 1.3 → up to about 1.9 % |
| 4.3 | Isospin parameter for PYTHIA single-charm decays (3.5) | AGREED (2026-11-25), section 3.3 | option (a): one common ε for B⁰ and B⁺, Gaussian prior **ε = 0.2 ± 0.2** (start at 0.2); applies to PYTHIA hadronic decays with one charm hadron that is a D or D\* among the B daughters; w_same = 1 − ε (charm meson of the spectator's isospin), w_other = 1 + ε·R with R = N_same/N_other from generator truth (B⁰ 6.0, B⁺ 7.5), so the PYTHIA single-charm rate per flavor is kept; two separate ε (BaBar priors 0.40 ± 0.15, 0.13 ± 0.06, or 0.2 ± 0.2 each) as a validation cross-check |
| 4.4 | Composition nuisance for D(\*)D̄(\*)K\* (3.4) and the S042.609 closure check | AGREED (2026-11-25), section 3.4 | (b): one rate-conserving nuisance θ on the share of the D(\*)D̄(\*)K\* family (B⁰ 664–677, B⁺ 4221–4234, conjugates) in lines with a D± among the B daughters (664, 666, 669, 670, 671, 4221, 4222, 4231, 4232, 4233): w_direct = 1 + θ, w_rest = 1 − θ·BF_direct/BF_rest per flavor (family total stays with N₂); common to B⁰ and B⁺; Gaussian prior **θ = −0.35 ± 0.35** (centred on the measured D(\*)D̄(\*)K pattern, decay file at 1σ). (i): S042.609, Λ_cΛ̄_cK⁰, Λ_cΛ̄_cK⁺ and D_s⁺D_s⁻K⁺ are closure checks in the validation (piece 7), not constraints; MC/measured now 2.95 ± 0.40, 0.61 ± 0.15, 0.44 ± 0.08, 3.3 ± 1.2 |
| 4.5 | The D₁(2420) decay table (3.7): event weight on `D_genMotherPDG` | AGREED (2026-11-25); line c DONE (2026-11-26), section 3.5 | (a): every D₁(2420) decay is reweighted, D\*π × **1.073**, Dππ × **0.891** (one nuisance, ±0.046 / ±0.076 anti-correlated; total D₁ rate kept), from `D_genMotherPDG` = D₁ (D from Dππ) or `D_genMotherPDG` = D\* and `D_genGMPDG` = D₁ (D through D\*π), in all samples and, at B9, in the D\*\*ℓν template; the product c through D₁ use the corrected table (`sub_decay` exception): D₁ℓν B⁰ 0.959 → 0.946, B⁺ 0.909 → 0.881, B⁰ D₁π 0.621 → 0.697 |
| 4.6 | r_SL: one parameter or a split; τ modes; D\*\* targets; normalization-mode values | OPEN (2026-11-26), section 3.6 (sub-points 4.6a–d) | – |
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
- *Update 2026-11-26:* the "ours now" column predates 4.5 (B⁰ D₁ now 0.666, Dππ 0.071; sum 9.45 %, B⁺
  10.28 %); the D₁′ comparison below compares a B⁰ product with a B⁻ one (HFLAV in B⁰ units is
  0.172 ± 0.027 %, so the difference is 0.077 %, +1.3σ). Details in 3.6.
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

*Context.* Pieces 2 and 3 produced a correction c ± σ_c for each exclusive decay-file line and
parameters (N₁, N₂, r_s, …) for the decays without one. This point decides how they are used: whether
the c change the MC (event weights) or only set allowed ranges, where their uncertainties go (into the
tuning fit, or propagated separately into the SR prior), and how the total B decay rate is kept
consistent once the measured lines are lowered.

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

### 3.2 Correlations between the corrections (opened 2026-11-24)

*Context.* After 4.1 the uncertainties of the c are propagated by toys: in each toy every c is drawn
from its uncertainty, the MC is reweighted, the tuning is redone (linearized) and the SR prediction
recorded; the spread is the uncertainty that enters the SR prior. This point decides how the toys draw
the c: which ones must move together because they share a source (one PDG number used for the e and
the μ line; 20 LHCb-based lines built on one measured BF; 22 BaBar modes with a common systematic) or
move oppositely (a remainder line computed as total − other lines). Independent draws get the
uncertainty wrong: too small on an absolute yield (correlated errors add linearly), and possibly wrong
in either direction on a ratio between two samples (a common shift cancels there). The tuning fits free
normalizations in the comb CR, so what reaches the SR is mostly the CR → SR ratio ("transfer").

Facts: section 2.3.

*Options (presented 2026-11-24; context explained 2026-11-25 at the user's request).* (a) construction
correlations only (bookkeeping of how c were built); (b) (a) plus the documented common components,
each c a sum of named sources drawn in the toys; (c) (a) plus ρ = 1 inside each family (upper bound).

**Decision (2026-11-25): option (b), AGREED.** Reasons: it gives the right absolute yield uncertainty
and the right transfers; most of it already exists as the readers' scratch source model (reproduces
every table σ_c within 3 %); sources that also act in the SR fit (f₀₀/f₊₋, D branching fractions,
D(\*)ℓν) can share their nuisance parameters with it at B9. The value of the B⁰/B⁺ semileptonic
correlation is set in 4.6. Implementation: a component file and the toy generator, with the tuning
code (B4). Also agreed: the possible errors 1 and 2 of section 2.3 are checked now (below); 3 and 4
stay flagged; 5 (r_s with the externals) is carried by the toys.

*Check 1 (2026-11-25): the "non-D₂\* part" 0.72 of the three-body D_s Dπ externals.* The chain, from
the exceptions file and the table: the D₂\* part of B⁺ → D⁻D_s⁺π⁺ is B(B⁺ → D̄₂\*⁰D_s⁺) after its
external c (line 4186: 0.62·10⁻³) × B(D̄₂\*⁰ → D⁻π⁺) of the decay file (0.3998) = 2.48·10⁻⁴; divided
by the D₂\* fit fraction 22.2 % (arXiv:2212.02717, Table 6) the total is 1.115·10⁻³. The phase-space
line 4189 should carry everything except D₂\* (no other component of the amplitude fit has its own
decay-file line; the virtual D\*⁰, the Dπ S-wave, interference and the small D\*\*/T_cs terms are all
in the phase-space line): 1.115 × (1 − 0.222) = **0.868·10⁻³**, factor 0.778. The value used,
0.80·10⁻³ (factor 0.72), is not reproduced by any reading of the paper (removing the small D\*\*/T_cs
components as well gives 0.736; normalizing to the sum of fit fractions, 94.7 %, gives 0.766); the
derivation script of 2026-11-20 is no longer available. **Proposed correction (not implemented):** all
eight three-body externals (4189–4192, 629–632 and conjugates) are built from the 0.80·10⁻³ value
(π⁰ lines × ½, D_s\* lines × 0.9, B⁰ ÷ 1.076), so they all scale by 0.778/0.72 = **1.081**: e.g. c(4189)
0.222 ± 0.085 → 0.240 ± 0.092. (Corrected 2026-11-25: the effect is about +0.05 % on the comb + hadB yield, not +0.2 %: the lines are
about 2.8 % of it but carry c ≈ 0.2, so the change is 2.8 % × 0.2 × 0.08.) The 3.2 prior is not
affected beyond rounding (the lines are outside N₂).

**Implemented (2026-11-25, approved by the user).** `Samples/dec/pdg_exceptions.json`: the 8 external
entries rebuilt from the B⁺ → D⁻D_s⁺π⁺ non-D₂\* rate 1.115·10⁻³ × 0.778 = 0.8675·10⁻³ (π⁰ lines × ½,
D_s\* lines × 0.9, B⁰ ÷ 1.076); value and error scaled by the same factor (1.083–1.099; the old B⁰
values were rounded), note added to each reason. `python3 scripts/tools/pdg_match.py` rerun; exactly
these 16 rows of `b_mode_pdg_table.csv` changed (16 lines of `b_mode_pdg_review.csv`, 8 of
`b_mode_isospin_check.csv`; nodes file unchanged):

| Lines (conjugates) | Final state | c before | c after |
|---|---|---|---|
| 4189 (5732) | B⁺ → D_s⁺D⁻π⁺ | 0.222 ± 0.085 | 0.241 ± 0.092 |
| 4190 (5733) | B⁺ → D_s⁺D̄⁰π⁰ | 0.222 ± 0.083 | 0.241 ± 0.090 |
| 4191 (5734) | B⁺ → D_s\*⁺D⁻π⁺ | 0.195 ± 0.113 | 0.211 ± 0.123 |
| 4192 (5735) | B⁺ → D_s\*⁺D̄⁰π⁰ | 0.200 ± 0.117 | 0.217 ± 0.126 |
| 630 (2440) | B⁰ → D_s⁺D̄⁰π⁻ | 0.200 ± 0.076 | 0.218 ± 0.082 |
| 629 (2439) | B⁰ → D_s⁺D⁻π⁰ | 0.206 ± 0.079 | 0.224 ± 0.087 |
| 632 (2442) | B⁰ → D_s\*⁺D̄⁰π⁻ | 0.181 ± 0.107 | 0.196 ± 0.116 |
| 631 (2441) | B⁰ → D_s\*⁺D⁻π⁰ | 0.183 ± 0.109 | 0.202 ± 0.120 |

Yield of comb + hadB (first order): CR +0.049 %, SR +0.047 %, WC +0.056 %. Class shares of the samples
are unchanged (the class of these lines did not change). Tests: `test_lhcb_ds_lines_34` expected values
updated (D_s⁺D⁻π⁺ 0.222 → 0.241, D_s\*⁺D⁻π⁰ 0.183 → 0.202); 37 and 15 tests pass.

*Check 2 (2026-11-25): BaBar 2011 normalized to the PDG 2008 D branching fractions.* The paper
(arXiv:1011.3929, ref. [17] = PDG 2008) reconstructs D⁰ → K⁻π⁺, K⁻π⁺π⁰, K⁻π⁺π⁻π⁺ and D⁺ → K⁻π⁺π⁺, and
D\*⁰ → D⁰π⁰, D⁰γ. PDG 2008 (summary tables, pdg.lbl.gov/2008, checked 2026-11-25) against PDG 2026:
K⁻π⁺ 3.89 → 3.936 % (+1.2 %), K⁻π⁺π⁰ 13.9 → 14.39 % (+3.5 %), K⁻π⁺π⁻π⁺ 8.10 → 8.205 % (+1.3 %),
D⁺ → K⁻π⁺π⁺ 9.22 → 9.313 % (+1.0 %), D\*⁰ → D⁰π⁰ 61.9 → 64.67 % (+4.5 %), D\*⁰ → D⁰γ 38.1 → 35.3 %
(−7.3 %), D\*⁺ → D⁰π⁺ 67.7 → 67.69 % (0). A measured BF scales inversely with these, so the BaBar values
would move by about −1 to −5 % per mode (D⁰D̄⁰ modes most; D\*⁰ modes partly cancel between π⁰ and γ);
the PDG 2026 entries equal the BaBar numbers (not rescaled). The exact factor per mode needs the paper's
yield split over the D sub-modes, which is not published. **Proposal (not implemented):** no change of
the nominal c; the shift is below the ≈ 6 % common BaBar component that the toys already carry
(option b), and its size (about −0.4 % of the yield, transfers unchanged) is recorded here.

### 3.3 Isospin parameter for PYTHIA single-charm decays (opened 2026-11-25)

*Context.* PYTHIA makes about 40 % (B⁰) / 36 % (B⁺) of all B decays as hadronic decays with one charm
hadron. Because the decay file switches off PYTHIA's orbitally excited mesons (3.5), none of these
decays produce a D\*\*; the charm meson at the lower vertex then almost always keeps the isospin of the
spectator quark (B⁰ → D⁻, D\*⁻; B⁺ → D̄⁰, D̄\*⁰). In nature part of this charm comes through D\*\* → D⁽\*⁾π,
which swaps the isospin in 2/3 of the decays. The result (3.5): too many D⁻ from B⁰ and too few D̄⁰,
against BaBar 2007 (and D± X too high against CLEO 1997). Our analysis reconstructs D⁺, so these
decays are 30–40 % of the comb D side. Point 3.5 agreed on one constrained parameter for this; this
point fixes its definition, rate conservation and prior.

*Facts (generator truth `truth_{mixed,charged}.root`, 5.2·10⁵ B⁰ and 4.1·10⁵ B⁺ decays; piece-2 c on the
exclusive lines; scripts `scratchpad/p43/`).*

| | B⁰ | B⁺ |
|---|---|---|
| PYTHIA hadronic single-charm decays, % of B decays | 40.3 | 36.2 |
| first-generation charm meson of the spectator's isospin ("same": D⁻/D\*⁻ for B⁰, D̄⁰/D̄\*⁰ for B⁺), % of them | 75.1 (D 40.9, D\* 34.2) | 77.9 (D 42.1, D\* 35.8) |
| of the other isospin ("other"), % | 12.5 | 10.4 |
| D_s / charm baryon / D\*\* / none at the lower vertex, % | 2.2 / 7.6 / **0** / 2.6 | 1.9 / 7.3 / **0** / 2.5 |
| R = N_same / N_other | 6.0 | 7.5 |
| MC lower-vertex D± X / D⁰ X (multiplicity, %) | 45.3 / 38.9 | 7.4 / 76.7 |
| BaBar 2007 (hep-ex/0606026) | 36.9 ± 3.3 / 47.4 ± 2.8 | 9.9 ± 1.2 / 79 ± 4 |
| ε that matches it (definition below) | 0.40 ± 0.16 (D±), 0.40 ± 0.13 (D⁰) | 0.13 ± 0.06 (D±), −0.11 ± 0.19 (D⁰) |

Each 0.1 of ε moves 2.1 (B⁰) / 1.9 (B⁺) points of lower-vertex D± X. The two flavors disagree by about
2σ (3.5: 0.38 ± 0.14 against 0.08 ± 0.06); BaBar's D⁰ + D̄⁰ total is 1.6σ above Belle 2023's (3.5), so
part of the B⁰ value may be BaBar's. Expectation if the missing D\*\* are the cause: ε ≈ 2/3 × the D\*\*
share of the lower-vertex charm (about 15–25 % by analogy with semileptonic decays), i.e. 0.1–0.17.

In our samples (mode tables, comb counted on both sides as in 4.1): PYTHIA single-charm B⁰ "same"
decays are 52 / 37 / 49 % of the CR / SR / WC yield, B⁰ "other" 1.5 / 1.1 / 0.5 %, B⁺ "same" 2.0 / 1.3 /
0.9 %, B⁺ "other" (B⁺ → D⁻X) 3.3 / 2.4 / **7.5 %**. Effect of ε (first order):

| ε (B⁰, B⁺) | CR | SR | WC | after N₁ refit in the CR: SR | WC |
|---|---|---|---|---|---|
| 0.2, 0.2 | −4.2 % | −2.7 % | +1.9 % | +0.29 % | +5.9 % |
| 0.4, 0.1 | −15.1 % | −10.5 % | −12.7 % | +0.27 % | +1.8 % |
| 0.4, 0 | −17.3 % | −12.2 % | −18.2 % | +0.18 % | −1.5 % |
| 0, 0.1 | +2.2 % | +1.7 % | +5.5 % | +0.10 % | +3.3 % |

- ε_B⁰ is almost degenerate with the free N₁ (B⁰ "same" is most of the PYTHIA single-charm yield in a
  D⁺ sample): after N₁ absorbs it, the SR moves by at most 0.3 %.
- ε_B⁺ acts through B⁺ → D⁻X, 7.5 % of WC: about +3 % of the WC yield per 0.1 after N₁. The WC sample
  is what constrains this parameter.

*Options (presented 2026-11-25).* (a) one common ε for both flavors, prior 0.2 ± 0.2; (b) ε_B⁰ and ε_B⁺
with BaBar priors 0.40 ± 0.15 and 0.13 ± 0.06; (c) two parameters with 0.2 ± 0.2 each.

*Definition (presented with the options).* PYTHIA hadronic decays (no lepton among the B daughters)
with exactly one charm hadron, which is a D or D\* among the B daughters (D_s, charm baryons excluded).
"Same": charm meson of the spectator's isospin (B⁰: D⁻, D\*⁻; B̄⁰: D⁺, D\*⁺; B⁺: D̄⁰, D̄\*⁰; B⁻: D⁰, D\*⁰);
"other": the other one. Weights w_same = 1 − ε, w_other = 1 + ε·R, R = N_same/N_other (truth: 6.0, 7.5),
so the PYTHIA single-charm rate is unchanged (consistent with 4.1, c1). Read from the ancestor-B
daughter lists already in the ntuple. Limitation: a weight cannot add the pion of a real D\*\* decay;
that kinematic difference is left to the slope a and the transfer term. The few PYTHIA b̄ → ūcs̄ decays
(0.39 / 0.36 %) with an upper-vertex D are classified by the same charge rule (negligible).

**Decision (2026-11-25): option (a) and the definition, AGREED.** Reasons: the missing-D\*\* mechanism
is isospin symmetric (the same 2/3 swap for D\*\*⁻ and D\*\*⁰); a separate ε_B⁰ would be fixed by its
prior alone, because the free N₁ absorbs it in a D⁺ sample, while a common ε is constrained by the WC
sample (B⁺ → D⁻X, 7.5 % of WC) and applied consistently to both flavors; BaBar's 2σ flavor difference,
partly BaBar's own (its D⁰ total is 1.6σ above Belle 2023), is not imposed as a prior. Implementation:
with the tuning code (B4); the per-flavor fit is a validation cross-check (piece 7).

### 3.4 The D(\*)D̄(\*)K\* lines and the S042.609 closure check (opened 2026-11-25)

*Context.* The decay file has 12 D(\*)D̄(\*)K\* lines per B flavor (B⁰ 664–677, B⁺ 4221–4234; 2.75 % of B
decays per flavor) with round BFs: 2.5·10⁻³ for DD, D\*D, DD\* and 5·10⁻³ for D\*D\* (pattern 1:1:1:2),
one fifth of that for the colour-suppressed neutral-pair lines. Nothing is measured; in 3.4 they stayed
in the N₂ class, so N₂ scales their total together with PYTHIA's double-charm decays. Two things can be
wrong: their total (handled by N₂) and their internal composition, which N₂ cannot change. For a D⁺
sample the composition matters because a line with a D± among the B daughters gives a D⁺ candidate
directly, a line with only a D\*± gives it through D\*⁺ → D⁺π⁰/γ (32 %), and neutral pairs hardly at all.
3.4 also named one measurement as a closure check: LHCb's B⁰ → D⁰D̄⁰K⁺π⁻ (S042.609), which contains the
colour-suppressed line 674 and PYTHIA decays.

*Facts (collected 2026-11-25 by three readers, each checked by a second; scripts in the session
scratchpad `p44/`).*

Symmetry estimate per line: B(K\* line) = R × B(measured D(\*)D̄(\*)K mode with the same D charges, PDG
2026, isospin-averaged, B⁺ = 1.076 × B⁰) × phase-space factor, R = B(DDK\*)/B(DDK) = 0.8 ± 0.4.

| | DD | D\*D, DD\* | D\*D\* | family total B⁰ / B⁺ |
|---|---|---|---|---|
| decay file | 2.5 | 2.5 | 5 | 2.75 / 2.75 % |
| phase-space factor (constant matrix element) | 0.44 | 0.36–0.37 | 0.26–0.27 | |
| estimate with the factor, c | 0.12–0.18 | 0.21–0.66 | 0.37–0.45 | 0.91 ± 0.45 / 0.98 ± 0.49 % (c 0.33 / 0.36) |
| estimate without the factor, c | 0.24–0.50 | 0.5–2.0 | 1.3–1.9 | 2.93 / 3.25 % (c 1.07 / 1.18) |

- The absolute level is not known to better than a factor 3: it depends on whether R is a ratio of
  branching fractions (phase space already included, like two-body V/P ratios J/ψK\*/J/ψK 1.4, DK\*/DK
  1.5–2.2, D_s\*/D_s 0.8–2.4) or of amplitudes. LHCb's m(K⁺π⁻) plot in arXiv:2007.04280 suggests a K\*⁰
  fraction of 20–32 % in D⁰D̄⁰K⁺π⁻ (read by eye), i.e. R ≈ 1.3 ± 0.7 for that mode. No other measurement
  or limit exists in PDG 2026; the LHCb thesis (Bernstein, Syracuse 2022, INSPIRE 2649734) could not be
  read (the hosts were refused by the proxy).
- The composition pattern is robust in every variant: the measured D(\*)D̄(\*)K modes are about 1:3:5:10
  (DD : D\*D : DD\* : D\*D\*), so the DD lines are relatively over-produced in the file (c of DD lines about
  half of the family average, D\*D\* lines about 1.3 times).
- 3.7's per-line values (669: 0.28, 4221: 0.40) correspond to no phase-space factor; its family total of
  2.1 / 2.3 % was not reproduced by either reading (2.9 % without, 0.9 % with the factor). 3.7's
  per-table maxima (4221 49 %, 11.5 %) are shares of the exclusive part of a table (as in
  `review_report`); of the full table they are at most about 6 % in large tables (29.5 % in a tiny WC
  hadB table).

In our samples (comb + hadB, comb counted on both sides, first order):

| | CR | SR | WC |
|---|---|---|---|
| the 24 lines (+ conjugates) | 11.3 % | 11.6 % | 6.5 % |
| of which lines with a direct D± | 8.0 % | 8.4 % | 4.5 % |
| lines with a D\*± only / neutral only | 3.1 / 0.2 % | 3.0 / 0.2 % | 1.9 / 0.2 % |

Per unit branching fraction a direct-D± line gives about 3 times the yield of a D\*±-only line and 28
times that of a neutral-only line. Largest lines: 669 (D⁻D⁺K\*⁰, 1.8 % of the CR), 670, 672, 671, 4222,
4221 (about 1 % each).

Effect on the transfers (N₂ class total held at its prior, N₁ refitted in the CR):

| change | CR → SR | CR → WC |
|---|---|---|
| family × 0.8 (normalization only) | +0.02 % | +0.31 % |
| per-line symmetry composition (variants) | −0.3 to −0.4 % | +1.1 to +1.5 % |
| 30 % of the rate moved from DD to D\*D\* lines | −0.10 % | +0.41 % |
| 30 % of the direct-D± lines' rate moved to the other lines of the family | −0.12 % | +0.72 % |
| for comparison: all σ_c of 4.2 | 0.38 % | 0.82 % |

- The family normalization is nearly degenerate with N₂ (4.2 % of the CR yield per % of B decays against
  3.8 for the N₂ class on average): no separate normalization is needed.
- The one composition axis that matters for a D⁺ sample is the fraction of the family's rate in lines
  with a direct D± (DD vs D\*D\* in the D⁻D⁺K\*⁰ block; which meson is the D\* in the blocks with one
  charged meson). Neutral pairs and the colour-suppressed lines do not matter (0.2 %).

S042.609 closure (B⁰ → D⁰D̄⁰K⁺π⁻, LHCb definition: whole K⁺π⁻ spectrum, only the D\*⁻ region vetoed, no
D\*⁰ feed-down): PDG 2026 (3.51 ± 0.47)·10⁻⁴ (= LHCb ratio 0.142 × S042.262). MC (generator truth +
decay file, piece-2 c, N₂ = 0.93): **(10.35 ± 0.30)·10⁻⁴, 2.95 ± 0.40 times the measurement.** Composition:
PYTHIA direct D⁰D̄⁰K⁺π⁻ 47 % (272 truth decays), line 674 D⁰D̄⁰K\*⁰ 30 %, ψ(3770)K\*⁰ and ψ(3770)K⁺π⁻
(lines 600, 602; unmeasured, c = 1) 23 %. A common scale on line 674 and PYTHIA would have to be
0.13–0.41 (depending on the ψ(3770) lines), against N₂ = 0.93. The final state is about 0.09 % of B⁰
decays (about 1 % of the N₂ class) and contains no D⁺: it tests PYTHIA's neutral double-charm content,
not the axis that matters for us.

Other measured double-charm final states that only PYTHIA produces (MC with N₂ / PDG 2026):
B⁰ → Λ_c⁺Λ̄_c⁻K⁰ (S042.356) 0.61 ± 0.15; B⁺ → Λ_c⁺Λ̄_c⁻K⁺ (S041.463) 0.44 ± 0.08; B⁺ → D_s⁺D_s⁻K⁺
(S041.789, D_s class) 3.3 ± 1.2. With S042.609 (2.95) they pull in opposite directions: PYTHIA's
b → cc̄s composition is off by factors 2–3 channel by channel, which no single N₂ can fix.

Side finding: the B⁰ ψ(3770)K\*, ψ(3770)Kπ lines (593–607) have c = 1 (unmeasured), while the B⁺
ψ(3770)K⁺ line has c = 0.53 (measured); different final states, so no isospin rule applies, but
S042.609 suggests they are high too.

*Options (presented 2026-11-25).* Composition: (a) no parameter, left to the transfer term u; (b) one
rate-conserving nuisance θ on the direct-D± share, prior −0.35 ± 0.35; (c) as (b) centred on the decay
file, 0 ± 0.35; (d) per-line symmetry estimates as c, out of N₂. Closure: (i) S042.609 and the three
PYTHIA-only channels as closure checks in the validation; (ii) S042.609 as a constraint on a
PYTHIA-composition parameter.

**Decision (2026-11-25): (b) and (i), AGREED.** Reasons: the direction of the composition (DD lines
over-produced relative to D\*D\* in the file) follows from the measured D(\*)D̄(\*)K modes in every
variant, while the absolute level is uncertain by a factor 3 and stays with N₂; θ acts on the only axis
that moves our transfers (CR → WC up to about 1.5 %). The four closure channels contain no D⁺ and pull in
opposite directions (PYTHIA's b → cc̄s composition off by factors 2–3), so they are reported, not fitted;
the unmeasured B⁰ ψ(3770)K\* lines (593–607) keep c = 1 ± 1 with a note. Implementation: with the tuning
code (B4); the closure table in piece 7.

### 3.5 The D₁(2420) decay table (opened 2026-11-25)

*Context.* A c corrects how often a B decays to a given set of daughters; it cannot correct how a
daughter resonance decays afterwards. The decay file's D₁(2420) table (lines 9621–9638) gives
D₁ → D\*π 59.9 % and D₁ → Dππ 40.1 %; the measurements of 3.7 say Dππ/D\*π is 0.83 ± 0.11 of the file's
ratio. This matters for us because D₁ → Dππ produces a D± directly, while D₁ → D\*π produces it only
through D\*± → D±π⁰/γ. Two things follow: the D⁺ yield from D₁ decays is off, and the line c of the D₁
lines, which were built from PDG products through D₁ → D\*π and D₁ → Dππ (piece 2), absorbed the
inconsistency.

*Facts (2026-11-25).*
- Decay file, D₁(2420)⁰ (line 9632; D₁⁺ at 9621 by isospin): D\*⁺π⁻ 0.3994, D\*⁰π⁰ 0.1997, D⁰π⁺π⁻ 0.1719,
  D⁰π⁰π⁰ 0.1145, D⁺π⁻π⁰ 0.1145 ("modes with neutral pions inferred assuming isospin", comment line 9617).
- Measured (3.7): Dππ/D\*π = 0.83 ± 0.11 of the file's (Belle 2023 arXiv:2211.09833 B⁺ 0.98 ± 0.17, B⁰
  0.77 ± 0.19; LHCb arXiv:1109.6831 0.68 ± 0.20). With the file's isospin pattern: B(D₁ → D\*π) = 0.643
  (D\*⁺π⁻ 0.429 ± 0.019), B(D₁ → Dππ) = 0.357. Ratio to the file: **w_D\*π = 1.074 ± 0.046, w_Dππ =
  0.889 ± 0.076** (one parameter: fully anti-correlated; total D₁ rate kept).
- D± per D₁ decay (file): from D₁± (B⁰ lines) 0.286 directly (Dππ) + 0.065 through D\*±; from D₁⁰
  (B⁺ lines) 0.115 directly + 0.129 through D\*±. With the weights: −7.7 % (D₁±) and −1.2 % (D₁⁰).
- The PDG products and the line c (PDG 2026, decay-file BF × sub-BF; c before → after the sub-decay
  correction):

| Line | Product | c from it, now | after dividing by w |
|---|---|---|---|
| 368/392 B⁰ → D₁⁻ℓν (line c 0.959 ± 0.074, χ² 1.17) | S042.420 (→ D̄\*⁰π⁻) 2.85·10⁻³ | 1.014 | 0.945 |
| | S042.623 (→ D⁻π⁺π⁻) 1.02·10⁻³ | 0.843 | 0.948 |
| 4000/4027 B⁺ → D̄₁⁰ℓν (line c 0.909 ± 0.050, χ² 1.23) | S041.257 (→ D\*⁻π⁺) 2.84·10⁻³ | 0.939 | 0.875 |
| | S041.783 (→ D̄⁰π⁺π⁻) 1.05·10⁻³ | 0.807 | 0.907 |
| 736 B⁰ → D₁⁻π⁺ (only the Dππ product) | S042.331 0.99·10⁻⁴ | 0.621 | 0.698 |
| 4318 B⁺ → D̄₁⁰π⁺ (exact, S041.214) | total, independent of the sub-decay | 0.783 | 0.783 |

  The two products of each D₁ℓν line agree once the sub-decay is corrected (0.945 / 0.948; 0.875 /
  0.907), which is the measured ratio seen from another side. The four D₁D_s externals of 3.4 already
  divide by 0.429 instead of 0.399, so they assume the corrected sub-decay: without the event weight,
  their D\*π branch is generated 7 % short. D₁τν (420, 4058) and D₁ρ (740, 4319) have no product; the
  weight applies to them as to every D₁(2420).
- In our samples (D-side and hadB tables, comb + hadB): B decays with a D₁(2420) among the daughters
  carry about 1 % of the yield on the D side (largest D₁D_s\*, D₁ℓν); the event weight changes the comb +
  hadB yield by about −0.07 % in every region. The ℓ-side D₁ℓν decays (1.5–4.5 % of the yield) are
  unaffected, because the reconstructed D does not come from them. Larger is the effect on the
  SR physics template D\*\*ℓν (B⁰ → D₁⁻ℓν, D₁⁻ → D⁻ππ: −11 % on that branch), which is outside the
  BBbar model.
- Truth needed: `D_genMotherPDG` = D₁(2420) (±10413, ±10423) for a D directly from D₁; `D_genMotherPDG`
  = D\*± or D\*⁰ and `D_genGMPDG` = D₁(2420) for a D through D\*. Both are in the ntuples (B.6).

*Options (presented 2026-11-25).* (a) event weight on every D₁(2420) decay (one nuisance) and the
product c of the D₁ lines recomputed with the corrected table; (b) only the line c, no event weight;
(c) no change.

**Decision (2026-11-25): option (a), AGREED; the line c implemented on 2026-11-26.** Reasons: the only
option consistent with the measured ratio, with the PDG products (the two products of each D₁ℓν line
then agree) and with the 3.4 externals (which already divide by 0.429); one weight on two existing
truth variables; the same weight belongs to the SR D\*\*ℓν template (B9).

*Implemented (2026-11-26).* New exception action `sub_decay` in `scripts/tools/pdg_match.py` (validated
in `load_exceptions`; `DecayGraph.reweight` multiplies the lines of a resonance's block with n daughters
by weights[n] and normalises again; applied before any product prediction or fold), documented in the
module docstring. One entry in `Samples/dec/pdg_exceptions.json` (99 → 100 entries): particles D_10,
anti-D_10, D_1+, D_1−, weights {2: 1.0731, 3: 0.8907} (B(D₁ → D\*⁺π⁻) = 0.4286). `pdg_match.py` rerun.
Changed rows of `b_mode_pdg_table.csv` (B⁰/B⁺ rows; anti-B rows follow):

| Lines | Final state | c before | c after | note |
|---|---|---|---|---|
| 368, 392 | B⁰ → D₁⁻ℓν | 0.959 ± 0.074 (χ²_prod 1.17) | 0.946 ± 0.073 (χ²_prod 0.00) | the two products agree |
| 4000, 4027 | B⁺ → D̄₁⁰ℓν | 0.909 ± 0.050 (χ²_prod 1.23) | 0.881 ± 0.049 (χ²_prod 0.06) | |
| 736 | B⁰ → D₁⁻π⁺ | 0.621 ± 0.140 | 0.697 ± 0.157 | only the Dππ product |
| 379, 403 | B⁰ → D⁻π⁺π⁻ℓν (direct line, remainder of S042.556) | 1.38 ± 1.59 | 2.05 ± 1.48 | less of the total goes through D₁ → Dππ |
| 4011, 4038 | B⁺ → D̄⁰π⁺π⁻ℓν (remainder of S041.649) | 2.38 ± 1.35 | 3.08 ± 1.24 (now `c_extreme`) | same; gap lines, revisited in 4.7 |
| 693 | B⁰ → D̄\*⁰π⁻π⁺ (clamped remainder) | 0 ± 0.735 | 0 ± 0.743 | |

Node closures that moved (pull before → after): S042.312 B⁰ → D̄\*⁰π⁻ℓν +5.2 → +5.7σ and S041.419 B⁺ →
D\*⁻π⁺ℓν +3.9 → +4.3σ (more D₁ → D\*π; these totals were already flagged above PDG, M2, and are 4.6/4.7
matters); S042.239 +1.2 → +1.4σ; S041.2 −0.85 → −0.66σ; the others below 0.05σ. `b_mode_pdg_review.csv` 46
lines, `b_mode_pdg_nodes.csv` 28 lines, `b_mode_isospin_check.csv` 8 lines changed. Tests: new
`test_d1_sub_decay_45`; `test_real_semileptonic` and the D₁π share in `test_real_review_fixes` updated to
the corrected table (the B⁰ → D⁻a₁⁺ remainder is unchanged, because c(736) × B(D₁ → Dππ) is); 38 and 15
tests pass. The weight's uncertainty is not added to σ_c: it is one source in the toys (4.2), shared by
the event weight and the c of these lines.

*Follow-up questions (user, 2026-11-26).*
1. *Other excited charm states.* Decay-file tables against PDG 2026 (pdg 2026.0 database):
   - D₂\*(2460) (lines 9675–9694): Dπ/D\*π = 1.498 in the file against 1.52 ± 0.14 (M254R3): consistent.
   - D₁′(2430), D₀\*(2300): one channel each (D\*π, Dπ), fixed by spin and parity; PDG only "seen"; no
     table issue. Their problem is the lineshape (broad Breit–Wigner tails, R(X) and D(\*)ηℓν notes),
     which is kinematic and belongs to 4.7/4.8.
   - D_s1(2536) (D'_s1, line 9714): file D\*⁺K⁰ : D\*⁰K⁺ = 1 : 1; PDG 31 ± 7 % : 36 ± 6 % (ratio 1.18 ± 0.16),
     plus D⁺π⁻K⁺ 1.0 ± 0.25 % and D_sππ "seen" not in the file. The D⁺ per D_s1(2536) decay changes
     from 0.162 to about 0.16: negligible.
   - D_s1(2460) (D_s1, line 9724): file D_s\*π⁰ 0.80, D_sγ 0.20; PDG 48 ± 11 %, 18 ± 4 %, D_sπ⁺π⁻ 4.3 %,
     D_s0\*γ 3.7 %: within the errors, and no D± is produced. D_s0\*(2317) → D_sπ⁰ 100 % (the
     sub-decay uncertainty is the 20 % of exception D2). D_s2\*(2573): DK 0.9, D\*K 0.1, PDG only limits.
   - D\*(2010), D\*(2007): equal to PDG (the D(\*)ηℓν note, Table 4.1).
   So D₁(2420) is the only excited state with a measured inconsistency in its decay table; it is
   also the one whose two channels give the D⁺ in different ways (directly or through D\*).
2. *Consequence for the PYTHIA isospin parameter (4.3).* No change of the definition. (i) No double
   counting: PYTHIA decays contain no D\*\* in the truth, so the D₁ event weight never acts on them, and ε
   never acts on the exclusive D\*\* lines. (ii) The physics expectation of ε changes a little: the 2/3
   isospin swap holds for two-body D\*\* → D⁽\*⁾π decays; D₁ → Dππ swaps only in the D⁺π⁻π⁰ / D⁰π⁺π⁰
   channels (29 % of Dππ in the file), so D₁ swaps in about 0.643 × 2/3 + 0.357 × 0.29 = 0.53 of its
   decays. The expectation for ε moves from 0.10–0.17 to about 0.08–0.15, well inside the prior
   0.2 ± 0.2. (iii) A D\*\* also adds a pion; for D₁ → Dππ two. ε does not add them (stated in 3.3);
   the multiplicity slope a and the transfer term carry it.

### 3.6 r_SL, τ modes, D\*\* targets and normalization modes (opened 2026-11-26)

*Context.* Semileptonic B decays enter our samples mostly through the comb ℓ side: the lepton of one B
is paired with a D from the other B. The model so far has one parameter r_SL for every decay with a
lepton daughter (B.6). This point decides (a) whether one r_SL is enough or the semileptonic family
must be split; (b) how the τ modes are corrected (their decay-file rates come from theory ratios, and
correcting them with measured rates would feed the R(D) anomaly that our analysis tests into the
background; the D\*\*τν lines are about 6 times too high, the user confirmed they must be corrected with
R(D\*\*)); (c) which values the D\*\*ℓν lines are corrected to (PDG per flavor, our current c, or the
HFLAV isospin averages that the Belle II recipe uses); (d) the same question for B → Dℓν, D\*ℓν. The gap
modes are point 4.7; (a) and (c) set how much room is left for them.

*Facts (2026-11-26; three readers, each checked by a second; scripts in the session scratchpad
`p46/`, `p46code/`).*

Sizes (share of the comb + hadB yield, comb counted on both sides, current c):

| | CR | SR | WC |
|---|---|---|---|
| all semileptonic | 44.6 % | 42.6 % | 82.4 % |
| of which on the comb ℓ side | 38.3 % | 33.6 % | 75.9 % |
| B⁰ D ℓν / D\* ℓν | 6.5 / 14.8 % | 6.7 / 13.3 % | 11.0 / 32.8 % |
| B⁺ D ℓν / D\* ℓν | 1.7 / 5.6 % | 1.6 / 5.9 % | 2.8 / 9.5 % |
| narrow D\*\* (D₁, D₂\*), B⁰ + B⁺ | 4.4 % | 3.3 % | 8.5 % |
| broad D\*\* (D₀\*, D₁′), B⁰ + B⁺ | 1.8 % | 1.3 % | 3.5 % |
| gap lines (D(\*)π, D(\*)ππ, D(\*)η, D_sK), B⁰ + B⁺ | 4.4 % | 3.2 % | 8.0 % |
| D τν + D\* τν (about half on the D side) | 4.4 % | 5.9 % | 4.9 % |
| D\*\*τν: decay file / corrected with R(D\*\*) | 1.45 / 0.27 % | 1.63 / 0.30 % | 2.0 / 0.36 % |

Transfer ratios (SR/CR, WC/CR per unit of each component): light-lepton b → c 0.90 (1.95); within it
B⁰ narrow D\*\* 0.70, broad 0.63, gap 0.71 (WC 2.06, 2.06, 1.85); τ modes the other way: Dτν 1.40,
D\*τν 1.31, D\*\*τν about 1.13; N₁ (PYTHIA single charm) 0.72 (0.97).

(a) With one common r_SL refitted in the CR, +1σ on each component leaves in the SR: B⁰ gap at 100 %
−0.80 %, B⁰ broad D\*\* at 50 % −0.22 %, Dτν and D\*τν at 10 % +0.10 % each, everything else ≤ 0.08 %
(sum in quadrature 0.86 %, 0.27 % without the gap); CR → WC 0.39 % (largest Dτν −0.20, D\*τν −0.17,
B⁰ D\*ℓν +0.16, B⁺ gap −0.15, B⁰ broad +0.14 %). With N₁ and one r_SL fitted to CR + WC together, the SR
keeps 0.89 %: that part no N₁/r_SL combination absorbs. Light-lepton D, D\* and narrow D\*\* move together;
broad D\*\* and gap have nearly the same transfer as each other (0.63, 0.71), different from the rest.

(b) τ modes.
- Decay file: D\*τν, Dτν (B⁰ 417, 418; B⁺ 4055, 4056) are set to R(D\*) = 0.258, R(D) = 0.299 of the
  light-lepton lines (HFLAV 2016 theory, line comments); CLN/HQET3 form factors against BGL for the
  light leptons. D\*\*τν (B⁰ 420–423, B⁺ 4058–4061) 0.13/0.13/0.20/0.20 % for D₁/D₀\*/D₁′/D₂\* (0.66 % per
  flavor; implied R 0.17–0.58), inherited unchanged from MC14; their own LLSW parameters (Bernlochner–
  Ligeti 2017 Approx. C) predict R ≈ 0.06–0.10. No τ gap lines.
- Current table: PDG 2026 τ entries, c = 1.126 (B⁰ Dτν), 1.139 (B⁰ D\*τν), 1.115, 1.325 (B⁺); implied R
  0.34/0.31/0.35/0.34. Line 418 is our own signal decay; in comb it is 1.7 % (CR) / 2.4 % (SR) of the
  yield. Measured R in the background model = the R(D) excess put into the background, partly from
  overlapping Belle (II) data.
- SM: HFLAV 2023 R(D) = 0.296 ± 0.004, R(D\*) = 0.254 ± 0.005; measured 0.342 ± 0.026, 0.286 ± 0.012
  (ρ −0.39; ratio to SM 1.155, 1.126). SM R × our light-lepton BFs: c = 0.981 ± 0.031 (418), 0.944 ± 0.030
  (417), 0.947 ± 0.029 (4056), 0.992 ± 0.044 (4055).
- R(D\*\*) theory (BLR 2018, arXiv:1711.03110 eq. 38; used by the R(D\*) note via BLPR 2023): D₀\* 0.08 ± 0.03,
  D₁′ 0.05 ± 0.02, D₁ 0.10 ± 0.02, D₂\* 0.07 ± 0.01 (BL 2017, used by the R(X) note: 0.08 ± 0.04, 0.06 ± 0.02,
  0.10 ± 0.02, 0.07 ± 0.01). **The only measurement:** LHCb arXiv:2501.14943 (PRL 135, 021802, 2025; 3.5σ),
  B⁻ → D\*\*⁰τν for D₁ + D₂\* together: R(D\*\*⁰₁,₂) = 0.13 ± 0.04, against their SM 0.09 ± 0.02 (ratio
  1.44 ± 0.52). Nothing for the broad states; no Belle or Belle II measurement. Belle II's published
  R(D\*) assumes 0.085 ± 0.012 with 100 % on the BF; LHCb's R(D(\*)) analyses use the theory with 50 %.
- D\*\*τν c with theory R × our D\*\*ℓν: B⁰ D₁ 0.51 ± 0.11, D₀\* 0.08 ± 0.05, D₁′ 0.09 ± 0.04, D₂\* 0.10 ± 0.02;
  B⁺ 0.51, 0.08, 0.06, 0.14; total 0.116 % (B⁰), 0.118 % (B⁺) against 0.66 %. With LHCb's narrow R (× 1.44
  on D₁, D₂\*) the total is about 0.155 %.
- τ gap, top-down (R(X_c) = 0.223 ± 0.005 × inclusive − exclusive τ sum): B⁰ 0.26 ± 0.11 %, B⁺ 0.23 ± 0.12 %
  (0.30, 0.28 with HFLAV's inclusive); bottom-up (R ≈ 0.05–0.10 × light gap, as the published Belle II
  R(D\*)) 0.08–0.16 %. No lines to reweight: dedicated samples, or the D\*\*τν lines as carriers (4.7).
- All τ changes together: comb + hadB yield −1.9 / −2.3 / −2.5 % (CR / SR / WC), transfers −0.38 % (CR →
  SR), −0.58 % (CR → WC). Corrected B⁺ τ total 2.46 % with the gap, against PDG B⁺ → τνX 2.51 ± 0.44 %
  (the current table gives 3.4 %).

(c) D\*\*ℓν targets (per light lepton, %):

| | B⁰ ours | B⁰ HFLAV | B⁰ SLME | B⁺ ours | B⁺ HFLAV | B⁺ SLME |
|---|---|---|---|---|---|---|
| D₁ | 0.666 ± 0.051 | 0.585 ± 0.041 | 0.59 | 0.667 ± 0.037 | 0.630 ± 0.044 | 0.64 |
| D₂\* | 0.292 ± 0.032 | 0.299 ± 0.027 | 0.299 | 0.397 ± 0.024 | 0.321 ± 0.029 | 0.321 |
| D₁′ | 0.374 ± 0.083 | 0.258 ± 0.040 | 0.26 | 0.256 ± 0.086 | 0.278 ± 0.044 | 0.28 |
| D₀\* | 0.125 ± 0.075 | 0.117 ± 0.029 | 0.08 | 0.135 ± 0.081 | 0.126 ± 0.032 | 0.08 |
| sum | 1.457 | 1.259 | 1.229 | 1.455 | 1.355 | 1.321 |

HFLAV values: B⁻ products of Tables 34–37 × 3/2 ÷ the fraction of the full width (D₁ 0.4286 after 4.5,
D₂\* × 2.52), B⁰ = B⁻/1.076. Why ours differ: PDG averages each flavor separately and includes Belle 2008
(Liventsev, arXiv:0711.3252) together with Belle 2023 (Meier, arXiv:2211.09833) from overlapping data;
HFLAV drops Belle 2008, rescales BaBar to current inputs, adds LEP/CLEO/D0 for the narrow states and
combines B⁰ and B⁺ by isospin. Our per-flavor values break isospin (B⁺/B⁰ 1.00 for D₁, 1.36 for D₂\*, 0.69
for D₁′; expected 1.076), and the PDG B⁰ products give D₂\* Dπ/D\*π = 2.13 ± 0.47 against 1.52 ± 0.14. D₀\*:
the B⁰ limit (Meier 2023, S042.418 < 4.4·10⁻⁴ → < 0.066 % for the full D₀\*) conflicts with B⁺ and older
B⁰ measurements (HFLAV χ² 11/1, scale factor 3.3); SLME's 0.08 is not reproducible from one source
(the same note uses 0.12–0.13 in its App. C). Yield effect of HFLAV instead of ours: −0.66 / −0.46 / −1.31 %
(CR / SR / WC); SLME −0.79 / −0.57 / −1.53 %.

(d) Normalization modes: PDG 2026 B⁰ D\*ℓν 4.90 ± 0.12 and Dℓν 2.12 ± 0.06 are HFLAV's; B⁺ D̄\*⁰ℓν
5.53 ± 0.22 rests on CLEO and two BaBar results only (χ² 7.0/2, CL 3 %, no Belle or Belle II); D̄⁰ℓν 2.21 ±
0.06. Isospin (B⁺ = 1.076 × B⁰, with HFLAV's ±2 % Coulomb allowance): D\* 5.27, D 2.27 (c 0.960, 0.983
instead of 1.007, 0.957). B⁺/B⁰ measured: D\* 1.129 ± 0.053, D 1.042 ± 0.042, inclusive 1.070 ± 0.052. Yield
effect of the isospin values: −0.23 / −0.25 / −0.40 %.

Semileptonic sums with our current c: B⁰ 9.45 %, B⁺ 10.28 % against PDG inclusive 10.08 ± 0.37 / 10.79 ±
0.37 (HFLAV 10.63 ± 0.15 split by lifetime: 10.24 / 11.02). Room for the gap after D + D\* + D\*\*: 1.60
(ours, PDG inclusive) to 2.00 % (HFLAV D\*\*, isospin normalization, HFLAV inclusive) for B⁰; 1.59–2.16 % for
B⁺; the gap lines now hold 0.97 / 1.08 %.
