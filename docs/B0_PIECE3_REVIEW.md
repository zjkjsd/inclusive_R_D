# B0 piece 3: exclusive modes without a measurement

Review of how the exclusive decay-file lines without a PDG measurement, and the PYTHIA decays, enter
the BBbar background model (step B0, piece 3 of `docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`, section
B.13; model parameters in B.6). Same conventions as piece 2 (`docs/B0_PIECE2_REVIEW.md`): every point
has the facts, the options, a recommendation and a decision; a decision is entered only after it was
discussed and agreed. Status labels: OPEN, AGREED (decided, not implemented), DONE (implemented, tables
rerun).

Last updated: 2026-11-19 (3.1 and P7 decided, P7 implemented; 3.2 opened, waiting for MC truth).

Line numbers refer to `Samples/dec/DECAY_BELLE2_rel08.DEC` (B⁰ block 360–2160, PYTHIA lines 2134–2158;
B⁺ block 3992–5525, PYTHIA lines 5503–5523). "Unmeasured" means class none or limit in
`b_mode_pdg_table.csv` (after piece 2, P2–P6). Rates are in % of the decays of that B flavor;
inclusive rates are multiplicities (mean number per B decay), as in the measurements.

## 1. Decisions

| # | Point | Status | Decision and current values |
|---|---|---|---|
| 3.1 | Class structure: do the unmeasured exclusive b → cc̄s lines and the PYTHIA ≥ 2-charm decays form one class with one scale factor? | AGREED (2026-11-19), section 3.1 | option (e): one free normalization N₂ for all unmeasured b → cc̄s (exclusive class none/limit and PYTHIA ≥ 2 charm), plus one constrained composition parameter (the fraction of the D_s class, i.e. decays with a D_s(\*) among the B daughters), prior from the inclusive data (3.2); whether the data constrain it is decided by the Fisher matrix of piece 5 |
| 3.2 | Inclusive constraints and priors for N₂ and the D_s-class scale | OPEN (2026-11-19), section 3.3 | the priors depend on the fraction of PYTHIA's upper-vertex charm that becomes a D_s (f_s); waiting for the generator truth of official MC16rd generic mdst (`scripts/mc_truth/b_decay_truth_basf2.py`) |
| 3.3 | N1: uncertainty of the lines kept individually (single charm, charmonium, charmless; and the 16 lines left without σ by P3/P4) | OPEN | proposal of B.13: c = 1 ± 1 unless a symmetry gives better |
| 3.4 | Checks: isospin/SU(3) relations of the large unmeasured lines to measured modes | OPEN | – |
| P7 | Lines written with the wrong charm flavor: 806 / 2616 (B⁰ → D\*⁰π⁺π⁺π⁻π⁻) and 1002 / 2812 (B⁰ → Λ̄ΛD⁰) (piece-2 follow-up) | DONE (2026-11-19), section 3.2 | option (c): veto both lines (conjugates inherit), rewrite S042.195 and S042.457 to the anti-charm form, report to Belle II (`docs/DECAY_FILE_ISSUES_FOR_BELLE2.md`) |

## 2. Facts (2026-11-19)

Collected by three independent checks (shares from the current tables; the decay file's own inclusive
rates; inclusive measurements from PDG 2026 and the papers) and an adversarial cross-check that
recomputed the key numbers. Scripts in the session scratchpad; no repository file was changed.

### 2.1 The unmeasured exclusive lines (B.13 facts, re-checked)

| Hadronic exclusive lines, class none | B⁰ | B⁺ |
|---|---|---|
| all (of which `unmeasured`) | 8.14 % (0.26 %) | 7.96 % (0.22 %) |
| limit class, separately | 0.25 % | 0.64 % |
| two open-charm hadrons | 6.71 % | 6.68 % |
| – D(\*)D̄(\*)K\* (lines 664–677; 4221–4234), 12 lines at 0.25 % or 0.5 % | 2.75 % | 2.75 % |
| – D_s(\*)D(\*)π(π) (629–638; 4189–4198) | 2.58 % | 2.57 % |
| – D\*\*D_s(\*) (623–628; 4182–4187) | 1.36 % | 1.36 % |
| – other (807, 2114, 2122) | 0.02 % | 0 |
| one open-charm hadron (D\*\*ρ 0.33 / 0.29 %) | 0.67 % | 0.49 % |
| charmonium | 0.50 % | 0.56 % |
| charmless (mesons + baryons) | 0.26 % | 0.23 % |

- B.13 is confirmed to rounding (B⁰ total 8.1 %, not 8.2 %; B⁺ charmonium 0.6 %, not 0.5 %).
- None of the 59 double-charm lines has a comment; the section headers say only "b → c (sc̄) → D D_s X
  Sum = 10 %" and "D D̄ K X Sum = 8 %". Round values: 12 lines at 2.5·10⁻³, 4 at 5·10⁻³, 6 at 5·10⁻⁴. Of
  all 1733 hadronic class-none lines, 1661 have no comment (B.13's "1716 of 1729" not reproduced; "almost
  all" holds).
- Largest shares (max over the 72 tables, n_top95): 4221 49.1 % (36), 4189 18.9 % (35), 4222 12.0 %
  (34), 4191 / 4196 11.1 / 11.3 %, 4186 9.6 %, 669 8.1 %. The 49 % is a table of 7 candidates; in
  tables with ≥ 500 candidates the largest shares are 11.5 % (4221) and 12.1 % (4189).
- In the samples, "had. none" is 93–97 % double charm in every table.

### 2.2 PYTHIA lines (quark level; their hadron content is not in the decay file)

| | B⁰ (2134–2158) | B⁺ (5503–5523) |
|---|---|---|
| all | 48.32 % | 43.34 % |
| one charm, mesonic (b̄ → c̄ud̄, c̄us̄) | 34.94 % | 31.03 % |
| one charm, diquark (Λ̄_c, Ξ̄_c) | 4.08 % | 3.81 % |
| b̄ → c̄cs̄ (2147; 5512; "crank up psi production") | 7.19 % | 6.71 % |
| b̄ → c̄cd̄ (2148; 5513) | 0.58 % | 0.36 % |
| b̄ → ūcs̄ (2150; 5515) | 0.39 % | 0.36 % |
| charmless | 1.53 % | 1.42 % |

Like for like (b → cc̄s, charmonium included): unmeasured exclusive 7.24 % against PYTHIA 7.19 % (B⁰),
7.30 % against 6.71 % (B⁺). The exclusive and PYTHIA parts of b → cc̄s are about equal.

### 2.3 Inclusive measurements (PDG 2026 and papers)

| Quantity | Value | Source | Definition |
|---|---|---|---|
| B → D_s± X, B⁺/B⁰ average (S049.113) | 10.6 ± 0.6 % (S = 1.7) | dominated by Belle 2023 (arXiv:2305.10098) 11.28 ± 0.43 % | multiplicity, both charges, D_s from D_s\* included |
| B⁺ → D_s⁺ X / D_s⁻ X (S041.384/385) | 7.9 +1.4 −1.3 % / 1.10 +0.40 −0.32 % | BaBar 2007 (hep-ex/0606026, hadronic tag) | multiplicity; upper / lower vertex |
| B⁰ → D_s⁺ X / D_s⁻ X (S042.294/295) | 10.3 +2.1 −1.8 % / < 2.6 % | BaBar 2007 | same |
| upper-vertex D⁰ X (S042.290 / S041.380) | B⁰ 8.1 ± 1.5 %; B⁺ 8.6 ± 0.7 % | BaBar 2007 | multiplicity, D from D\*, D\*\* |
| upper-vertex D⁺ X (S042.292 / S041.382) | B⁰ < 3.9 % (paper 2.3 ± 1.1); B⁺ 2.5 ± 0.5 % | BaBar 2007 | same |
| wrong-sign D, D⁰ + D⁺ (paper sums, quoted by BaBar 2011) | B⁰ 10.4 ± 1.9 %; B⁺ 11.1 ± 0.9 % | BaBar 2007 | same |
| upper-vertex charm N(c) (S042.299 / S041.389) | B⁰ 24.6 ± 3.1 %; B⁺ 23.4 +2.2 −1.8 % | BaBar 2007 | sum over species; assumes cc̄ X = 2.3 % |
| n_c (S042.300 / S041.390) | 1.19 ± 0.06; 1.20 ± 0.06 | BaBar 2007 | (S049S24 = 1.10 is CLEO 1997 with old D_s, Λ_c BFs) |
| 22 D(\*)D̄(\*)K modes (S042.270 / S041.340) | B⁰ 3.68 ± 0.26 %; B⁺ 4.05 ± 0.30 % | BaBar 2011 (arXiv:1011.3929): "roughly one third" of wrong-sign D | exclusive three-body sum, resonances included |
| B → D(\*)D̄(\*)K⁰ + D(\*)D̄(\*)K± (S049.51) | 7.1 +2.7 −1.7 % | ALEPH 1998, at the Z (paper not read) | b-hadron mix; X allowed or not: unchecked |
| b → cc̄s′ (S049.31) | 22 ± 4 % | CLEO 1998 (hep-ex/9710028), derived | b → (c or u) c̄ s′, s′ = s or d |
| J/ψ X (S049.106), ψ(2S) X, χ_c1 X | 1.094 ± 0.032 %, 0.307 ± 0.021 %, 0.309 ± 0.019 % | PDG averages | with feed-down (J/ψ) |
| Λ̄_c X / Λ_c⁺ X | B⁰ 5.0 +2.1 −1.5 %; B⁺ 2.8 +1.1 −0.9 % / 2.1 +0.9 −0.6 % | BaBar 2007 | multiplicity, Σ_c included |

### 2.4 The decay file against them

Decay file: measured exclusive lines (raw; [after piece-2 c]) + unmeasured exclusive lines + PYTHIA
(quark-level bounds only).

| Quantity | Measurement | B⁰ | B⁺ | Reading |
|---|---|---|---|---|
| D_s X (admixture) | 10.6 ± 0.6 (Belle 11.28 ± 0.43) | 7.86 [5.71] + 3.99 + PYTHIA ? | 8.33 [5.94] + 3.96 + PYTHIA ? | exclusive alone 12.07 % (9.80 after c), before any PYTHIA D_s: at or above the measurement |
| D_s⁺ X (upper vertex) | B⁰ 10.3 +2.1 −1.8; B⁺ 7.9 +1.4 −1.3 | 11.81 [9.66] + PYTHIA 0–8.2 | 12.13 [9.74] + PYTHIA 0–7.4 | B⁺: 3.2σ high raw, 1.4σ after c, before PYTHIA |
| wrong-sign D (D⁰ + D⁺) | B⁰ 10.4 ± 1.9; B⁺ 11.1 ± 0.9 | 7.19 (0.27 of it line 806) + PYTHIA 0–8.2 | 7.43 + PYTHIA 0–7.4 | exclusive B⁺ 4σ low: needs PYTHIA c lines or more D(\*)D̄(\*)K(\*) |
| upper-vertex charm N(c) | B⁰ 24.6 ± 3.1; B⁺ 23.4 +2.2 −1.8 | 28.8 (PYTHIA quark count exact) | 28.6 | +1.4σ, +2.4σ; on B⁺ the excess equals the PYTHIA c lines |
| correlated c̄ X | 95 ± 5; 97 ± 4 | 96.9 | 97.3 | agrees |
| n_c | 1.19 ± 0.06; 1.20 ± 0.06 | 1.257 | 1.259 | +1σ |
| 22 D(\*)D̄(\*)K modes | 3.68 ± 0.26; 4.05 ± 0.30 | 3.836 with the D_s1(2536), ψ(3770) lines [3.662] | 4.292 [4.062] | closes after piece 2; says nothing about the unmeasured D(\*)D̄(\*)K\* lines |
| b → cc̄s′ | 22 ± 4 | about 28.5 (with cc̄d and b → ucs̄) | about 28.6 | +1.8σ (old, derived measurement) |
| J/ψ X | 1.094 ± 0.032 | 0.82 [0.67] + 0.14 + PYTHIA ? | 0.68 [0.63] + 0.14 + PYTHIA ? | exclusive 0.88 (average): short unless PYTHIA forms charmonium |
| Λ̄_c X / Λ_c⁺ X | B⁰ 5.0; B⁺ 2.8 / 2.1 | 0.13 + diquark lines 3.2 | 0.53 + 3.0 / 0 + PYTHIA c lines | must come from PYTHIA |

Reading:
- The upper-vertex charm of the decay file is **wrongly composed**, not uniformly too large: too much
  D_s (exclusive alone at or above the measurement), too little non-strange D (exclusive B⁺ 4σ below).
  The unmeasured D_s D(\*)π(π) (2.58 %) and D\*\*D_s (1.36 %) lines are, if anything, too high; the
  D(\*)D̄(\*)K\* lines (2.75 %) are constrained only through the wrong-sign D rate.
- A single scale factor for all unmeasured b → cc̄s cannot correct a wrong D_s / D composition.
- The BaBar 22-mode sum constrains only the measured lines (piece 2 closes it).

### 2.5 What needs the PYTHIA hadron content

The D_s X, wrong-sign D, Λ_c X and charmonium rows depend on what PYTHIA makes of its quark-level
lines, which only a generator-level sample tells. The selected-candidate tables (`bbbar_modes_*.csv`)
are biased and cannot be used. Cheapest source: the MC truth already stored in a few official MC16rd
generic mdst files (no generation, no simulation; a short basf2 script that writes, per B, its
daughters and the charm hadrons of its decay chain). Not needed to decide 3.1; possibly needed for the
priors of 3.2.

### 2.6 Abnormal findings

- **Line 806** (B⁰ block) is `D*0 pi+ pi+ pi- pi-` (2.72·10⁻³), and its conjugate 2616 is `anti-D*0 pi-
  pi- pi+ pi+`: the charm flavor is wrong (a B⁰ gives D̄\*⁰). The PDG entry S042.195 has the same text
  without the bar, but its source, CLEO 2002 (EDWARDS 2002), is titled "First Observation of B̄⁰ →
  D\*⁰π⁺π⁺π⁻π⁻"; piece 2 matched it exact (c = 1.00 ± 0.18) and no flag caught it. Effect: 0.27 % of
  B⁰ decays are generated with a wrong-sign D⁰ instead of a right-sign D̄⁰. Max share 0.07 %
  (n_top95 = 5). Recorded as P7.
- The PYTHIA section headers are stale: "Lam_c X / Sigma_c X 4.0 %" above diquark lines that sum to
  3.21 % (B⁰) and 2.99 % (B⁺); "Xi_c X 2.5 %" above 0.87 % / 0.82 %.
- PDG's ⟨n_c⟩ (S049S24 = 1.10 ± 0.05) is CLEO 1997 with old D_s and Λ_c branching fractions; BaBar
  2007's n_c is listed only as B → c/c̄ X (S042.300, S041.390).
- B⁺ lines 4337 (D⁺K̄⁰) and 4338 (D\*⁺K̄⁰) have the wrong strangeness (PDG: D⁺K⁰, D\*⁺K⁰); already
  flagged by piece 2, BF ~10⁻⁵.

## 3. Discussion record

### 3.1 Class structure (2026-11-19)

*Options presented.* (a) one class, one free N₂, composition from MC (its error in the transfer term
of B.3); (b) two content classes across generators, "unmeasured b → cc̄s with a D_s(\*) among the B
daughters" and "with D D̄, no D_s", each with a free normalization; (c) keep the generator split
(unmeasured exclusive lines at c = 1 ± 1, N₂ for PYTHIA only); (d) as (b) with D(\*)D̄(\*)K\* split off.

*Question (user): do we have separation power for the two content classes, or control samples for
them?* Facts from the sample tables (raw MC, no c weights; all 12 files, e and μ summed; D_s class =
unmeasured exclusive lines and PYTHIA ≥ 2-charm decays with a D_s(\*) among the B daughters):

| Region | Table | D_s class | D D̄ class | D_s / D D̄ |
|---|---|---|---|---|
| comb CR | comb, D side | 6.6 % | 20.9 % | 0.32 |
| SR | comb, D side | 5.5 % | 25.3 % | 0.22 |
| WC | comb, D side | 10.0 % | 13.6 % | 0.73 |
| comb CR / SR / WC | comb, ℓ side | 4.7 / 4.3 / 2.9 % | 6.1 / 6.3 / 4.0 % | 0.76 / 0.68 / 0.73 |
| comb CR / SR | hadB | 18.2 / 15.9 % | 36.6 / 28.4 % | 0.50 / 0.56 |

- PYTHIA ≥ 2-charm decays seldom have a D_s among the B daughters (1–2 % of the yield, against 3–23 %
  without).
- The D side, ℓ side and hadB tables are not separate data samples (a comb event carries both B's;
  hadB and comb look alike in data). Separate data samples are the RS comb CR and WC: on the D side
  the ratio is 0.32 and 0.73. Reason: the D D̄ class gives upper-vertex D⁺ that pair with the primary
  lepton of the other B into RS; in the D_s class the upper-vertex charm is the D_s, so its D⁺ is a
  lower-vertex one, like single charm. This leverage exists only in a joint RS + WC fit.
- Kinematic separation in the fit variables (p\*_D, p\*_ℓ, ROE) is unknown (no kinematics in the
  mode tables); piece 5 (Fisher matrix) answers it. The B.4 tuning shows that free class weights
  without separation trade off and end at their limits.
- Cost of not separating: if the truth were D_s class × 0.8 and D D̄ class × 1.2 (roughly what the
  inclusive data suggest) and one factor were fitted in the CR, the SR prediction would miss by about
  −0.8 % of the comb D-side yield and +0.5 % of the hadB yield (the D_s / D D̄ ratio differs between CR
  and SR). Small, not zero: the transfer term of B.3.

*Revised options.* (a) and (b) as above (no longer recommended: (b) needs separation power not shown);
(e) one free N₂ for all unmeasured b → cc̄s plus a constrained composition parameter, the D_s-class
fraction, with a prior from the inclusive data. Without power in the fit it stays at its prior and its
effect is an explicit uncertainty; with power, the data narrow it; piece 5 tells which.

**Decision (2026-11-19): option (e), AGREED.** Reason: it keeps B.13's point (exclusive and PYTHIA
b → cc̄s are equally unmeasured and share one normalization), makes the known composition problem
(too much D_s, too little upper-vertex D in the decay file, section 2.4) an explicit, priced
uncertainty, and leaves the sensitivity question to piece 5 instead of assuming it. The prior of the
composition parameter is point 3.2.

### 3.2 P7: lines with the wrong charm flavor (2026-11-19)

*Facts.* A scan of every B line for net charm of the wrong sign finds, besides the CKM-suppressed
b̄ → ū W⁺(→ cs̄) lines (all ≤ 10⁻⁴) and cases already handled (798, G6; 4337/4338, flagged), two lines:
806 (2616), B⁰ → D\*⁰π⁺π⁺π⁻π⁻, 2.72·10⁻³, matched exact to S042.195 (c = 1.00 ± 0.18), whose source is
titled "First observation of B̄⁰ → D\*⁰π⁺π⁺π⁻π⁻" (CLEO, EDWARDS 2002); and 1002 (2812), B⁰ → Λ̄ΛD⁰,
1.0·10⁻⁵, matched exact to S042.457, whose source is BaBar 2014, "Evidence for the baryonic decay
B̄⁰ → D⁰ΛΛ̄" (arXiv:1401.5990). PDG's texts lack the bar (like E1, E2), and the decay file copied them.
The right-flavor final states are not in the B⁰ block. For us: the D\*⁰ → D⁰ is not our D⁺; the effect
is the charge of cascade leptons from the D⁰ (RS/WC on the ℓ side); 0.27 % of B⁰ decays, max share
0.07 %.

*Options.* (a) veto 806 and 1002 (c = 0; conjugates inherit), rewrite S042.195 and S042.457 to the
anti-charm form; (b) keep c = 1, flag and record; (c) as (a) and report to the Belle II decay-file
maintainers.

**Decision (2026-11-19): option (c), AGREED, not yet implemented.** Reason: wrong-sign charm is the
composition piece 3 is about; the veto keeps it out of the RS/WC comparison; the cost is negligible.
The report to Belle II, `docs/DECAY_FILE_ISSUES_FOR_BELLE2.md` (created 2026-11-19), collects every
decay-file problem found in pieces 2 and 3.

*Implemented (2026-11-19).* Exceptions 75 → 79 (E23 in the piece-2 register): two `rewrite` (S042.195,
S042.457 to the anti-charm form) and two `veto` (B⁰ lines 806, 1002). Changed lines: 806 and 1002
(B⁰), 2616 and 2812 (B̄⁰): exact c = 1.00 ± 0.18 / 1.00 ± 0.28 → veto, c = 0; no other line moved.
B⁰ BF by class: exact 28.31 → 28.04 %, none 10.58 → 10.86 %. Yield by class: only WC comb ℓ side, had.
exact 4.0 → 3.9 %. New test `test_wrong_flavor_p7`; the veto count of `test_real_coverage` is now 6.
Tests: 35 and 15 pass.

### 3.3 Priors from the inclusive measurements (opened 2026-11-19)

*Parameterization (from 3.1).* Unmeasured b → cc̄s decays (exclusive class none/limit with two open
charm hadrons, and PYTHIA ≥ 2-charm decays) are weighted by N₂; those with a D_s(\*) among the B
daughters by N₂·r_s (r_s = scale of the D_s class relative to the rest). Measured lines keep their c.

*Equations,* per B flavor, for the upper-vertex rates measured by BaBar 2007 (hep-ex/0606026):

- D_s⁺ X = M_s + N₂ r_s (U_s + f_s P)
- D⁰ + D⁺ X (wrong sign) = M_d + N₂ (U_d + f_d P)
- Λ_c⁺ X = N₂ f_Λ P, with f_s + f_d + f_Λ + f_ψ = 1

M = measured lines after the piece-2 c, U = unmeasured exclusive lines, P = PYTHIA lines with an
upper-vertex c quark (B⁰ 8.16 %, B⁺ 7.43 %), and f the fractions of those c quarks that end in a D_s,
a non-strange D, a Λ_c, or charmonium. Inputs (% of B decays): B⁰ M_s 5.68, U_s 3.97, M_d 3.83 (line 806
removed), U_d 2.89; B⁺ M_s 5.78, U_s 3.96, M_d 4.28, U_d 2.91. f_ψ taken as 0.04–0.12 (the J/ψ X
deficit of the exclusive lines), f_Λ from Λ_c⁺ X (B⁺ 2.1 +0.9 −0.6 %; B⁰ 1.6 ± 0.9 %, paper value).

*Result of a scan in f_s* (median and 68 % interval from the measurement errors;
`scripts/mc_truth/p32_prior_scan.py`):

| f_s | B⁰: N₂ | B⁰: r_s | B⁺: N₂ | B⁺: r_s |
|---|---|---|---|---|
| 0.0 | 0.79 [0.59, 0.99] | 1.50 [0.86, 2.38] | 0.92 [0.81, 1.05] | 0.58 [0.22, 0.98] |
| 0.1 | 0.85 [0.64, 1.07] | 1.15 [0.66, 1.82] | 1.00 [0.87, 1.14] | 0.45 [0.17, 0.76] |
| 0.2 | 0.93 [0.70, 1.17] | 0.89 [0.52, 1.42] | 1.09 [0.95, 1.24] | 0.36 [0.14, 0.60] |
| 0.3 | 1.03 [0.77, 1.29] | 0.71 [0.41, 1.13] | 1.20 [1.04, 1.36] | 0.29 [0.11, 0.48] |
| 0.4 | 1.15 [0.86, 1.44] | 0.56 [0.32, 0.89] | 1.33 [1.16, 1.51] | 0.23 [0.09, 0.39] |

- N₂ is known to about ±15 % (B⁺) for a given f_s, but moves by 40 % over the plausible f_s range.
- r_s depends on f_s even more; B⁰ and B⁺ disagree, because BaBar's D_s⁺ X is 10.3 % (B⁰) against
  7.9 % (B⁺) while the decay file gives nearly the same for both. For B⁺, r_s < 1 for any f_s (the D_s
  class too high).
- So the priors need the PYTHIA fractions f_s, f_d, f_Λ, f_ψ (and the lower-vertex D_s⁻ from ss̄
  popping, for the B → D_s X average of Belle 2023). They can be read from the generator truth of a
  few official MC16rd generic mdst files: `scripts/mc_truth/b_decay_truth_basf2.py` (written
  2026-11-19, not tested here: no basf2 in this environment), about 10⁵ events each of mixed and
  charged MC.
