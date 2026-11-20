# B0 piece 3: exclusive modes without a measurement

Review of how the exclusive decay-file lines without a PDG measurement, and the PYTHIA decays, enter
the BBbar background model (step B0, piece 3 of `docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`, section
B.13; model parameters in B.6). Same conventions as piece 2 (`docs/B0_PIECE2_REVIEW.md`): every point
has the facts, the options, a recommendation and a decision; a decision is entered only after it was
discussed and agreed. Status labels: OPEN, AGREED (decided, not implemented), DONE (implemented, tables
rerun).

Last updated: 2026-11-20 (3.2: generator truth analysed, priors proposed; new points 3.5, 3.6).

Line numbers refer to `Samples/dec/DECAY_BELLE2_rel08.DEC` (B⁰ block 360–2160, PYTHIA lines 2134–2158;
B⁺ block 3992–5525, PYTHIA lines 5503–5523). "Unmeasured" means class none or limit in
`b_mode_pdg_table.csv` (after piece 2, P2–P6). Rates are in % of the decays of that B flavor;
inclusive rates are multiplicities (mean number per B decay), as in the measurements.

## 1. Decisions

| # | Point | Status | Decision and current values |
|---|---|---|---|
| 3.1 | Class structure: do the unmeasured exclusive b → cc̄s lines and the PYTHIA ≥ 2-charm decays form one class with one scale factor? | AGREED (2026-11-19), section 3.1 | option (e): one free normalization N₂ for all unmeasured b → cc̄s (exclusive class none/limit and PYTHIA ≥ 2 charm), plus one constrained composition parameter (the fraction of the D_s class, i.e. decays with a D_s(\*) among the B daughters), prior from the inclusive data (3.2); whether the data constrain it is decided by the Fisher matrix of piece 5 |
| 3.2 | Inclusive constraints and priors for N₂ and the D_s-class scale | AGREED (2026-11-20), section 3.4 | option (a): correlated Gaussian prior N₂ = 0.94 ± 0.12, r_s = 0.80 ± 0.15, correlation −0.8 (from the generator-truth fit to BaBar 2007 D_s⁺ X and wrong-sign D X, B⁰ and B⁺, and Belle 2023 D_s X; r_s widened to cover the input variations). D_s class as defined in 3.1 (a D_s-family meson among the B daughters). Whether the central values also become starting weights: piece 4 |
| 3.3 | N1: uncertainty of the lines kept individually (single charm, charmonium, charmless; and the 16 lines left without σ by P3/P4) | OPEN | proposal of B.13: c = 1 ± 1 unless a symmetry gives better |
| 3.4 | Checks: isospin/SU(3) relations of the large unmeasured lines to measured modes | OPEN | – |
| 3.5 | D⁺ / D⁰ composition of the MC: D± over-produced against the inclusive measurements (outside N₂, r_s) | AGREED (2026-11-20), section 3.5 | option (c): the model gets a constrained isospin parameter for PYTHIA single-charm decays (rate moved between lower-vertex charm of the spectator's isospin and the other one, sum kept); definition and prior in piece 4, from BaBar 2007 per flavor, with a width that reflects the weak evidence (single measurement; Belle 2023 D⁰ X agrees with the MC); reported to Belle II (PYTHIA without L = 1 mesons) |
| 3.6 | Upper-vertex Λ_c⁺ (baryonic b → cc̄s) almost absent in the MC; ψ(2S), χ_c too low | OPEN (2026-11-20), section 3.4 | found in the truth analysis; small for the D⁺ yield |
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
| J/ψ X (S049.106), ψ(2S) X (S049.124), χ_c1 X (S049.170), χ_c1 direct (S049.24), χ_c2 X | 1.094 ± 0.032 %, 0.307 ± 0.021 %, 0.355 ± 0.027 %, 0.309 ± 0.019 %, 0.100 ± 0.017 % | PDG averages | with feed-down unless 'direct' (corrected 2026-11-20: the χ_c1 value given here before was the direct one) |
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

### 3.4 Generator truth of the MC (2026-11-20)

*Input.* One official MC16rd file each of mixed and charged generic MC (experiment 18; 259,945 and
204,379 events; 519,890 B⁰ and 408,758 B⁺ decays), run by the user through
`scripts/mc_truth/b_decay_truth_basf2.py` (light-2609-luna); files in `Samples/mc_truth/`. Analysis:
`scripts/mc_truth/truth_analysis.py`; summary table `Notebooks/Plotting/csv_output/truth_b_decays_summary.csv`.
Independent check (2026-11-20): a reimplementation reproduced every number; a code review found one
bug (exclusive lines of class none/limit were given weight 1, undoing the P7 vetoes and the
half_limit of 4319), fixed; numbers below are after the fix.

*Classification.* Exclusive = daughter set (PHOTOS photons removed by their generator flag) listed
for the B's own flavor; exact because of the Belle II EvtGen double-counting patch. Exclusive
fraction 51.65 % (decay file 51.68 %) for B⁰, 56.59 % (56.66 %) for B⁺; every mode at its BF (χ² = 910
for 789 modes; the largest pulls are the K\*S/K\*L CP lines 2316, 2334, whose K\* flavor is
conjugated with the B: about 70 decays, 0.013 % of B⁰, classified PYTHIA, no open charm); no decay on
a zero-BF line; every exclusive decay in a decay-file order. The same-flavor fraction of B⁰ pairs
is 0.190 (χ_d ≈ 0.186): the stored B is the flavor at decay.

*Classes* (% of B decays, B⁰ / B⁺): measured exclusive 40.23 / 45.24; unmeasured exclusive ≥ 2 charm
6.69 / 6.63; PYTHIA ≥ 2 charm 5.80 / 5.11; D_s class 5.45 / 5.32 (unmeasured 3.94 / 3.93, PYTHIA
1.51 / 1.39).

*PYTHIA's upper-vertex charm* (6.97 % / 6.26 % of B decays have one; the decay file's quark level is
8.16 % / 7.43 %): fractions per upper-vertex c quark, B⁰ / B⁺: D_s 0.201 / 0.203, D⁰ + D⁺ 0.685 / 0.672,
Λ_c 0.012 / 0.011, Ξ_c + Ω_c 0.001, charmonium 0.101 / 0.114.

*Inclusive rates of the MC* (multiplicity per B decay, %, with the piece-2 c; raw in brackets), against
the measurements (σ from the independent check, MC statistics included):

| Rate | MC B⁰ | MC B⁺ | Measured | Pull |
|---|---|---|---|---|
| D_s⁺ X, upper vertex | 11.08 (13.23) | 11.04 (13.38) | 10.3 +2.1 −1.8; 7.9 +1.4 −1.3 | +0.4; +2.2 |
| D_s± X, average of B⁰, B⁺ | 12.4 (14.7) | | Belle 2023: 11.28 ± 0.43 | +2.6 |
| D_s⁻ X, lower vertex | 1.41 | 1.32 | (1.5 ± 0.8); 1.10 +0.40 −0.32 | ok |
| D⁰ + D⁺ X, upper vertex | 11.48 | 11.37 | 10.4 ± 1.9; 11.1 ± 0.9 | +0.6; +0.3 |
| D⁺ X, upper vertex | 3.27 | 3.49 | (2.3 ± 1.1); 2.5 ± 0.5 | +0.9; +2.0 |
| D⁻ X, lower vertex (B⁰) | 45.7 (47.2) | | 36.9 ± 3.3 | +2.7 |
| D̄⁰ X, lower vertex (B⁰) | 39.9 | | 47.4 ± 2.8 | −2.7 |
| D± X, average | 29.8 (30.8) | | 23.3 ± 1.2 (CLEO 1997); BaBar 2007 sum about 25.4 ± 1.8 | +5.4; +2.4 |
| D⁰ X, average | 67.5 | | Belle 2023: 66.63 ± 1.77 | +0.5 |
| Λ_c⁺ X, upper vertex | 0.086 | 0.069 | (1.6 ± 0.9); 2.1 +0.9 −0.6 | −1.7; −3.4 |
| J/ψ X (with feed-down) | 1.08 | 1.08 | 1.094 ± 0.032 | −0.4 |
| ψ(2S) X / χ_c1 X / χ_c2 X | 0.24 / 0.23 / 0.06 (average) | | 0.307 / 0.355 / 0.100 | −3.2 / −4.6 / −2.5 |
| n_c (charmonium = 2) | 1.18 | 1.18 | 1.19 ± 0.06; 1.20 ± 0.06 | ok |

(Values in brackets in the "Measured" column are BaBar 2007 paper values where PDG gives a limit.)

*Priors* (section 5 of the script): N₂ and r_s common to B⁰ and B⁺, fitted to five measurements
(BaBar 2007 upper-vertex D_s⁺ X and D⁰ + D⁺ X for B⁰ and B⁺; Belle 2023 D_s X), with piece-2 c on the
exclusive lines and the other classes fixed: N₂ = 0.94 [0.82, 1.06], r_s = 0.83 [0.71, 0.98],
correlation −0.82, χ² = 2.3 for 3 degrees of freedom. Leave one out: N₂ 0.85–0.96, r_s 0.61–0.91
(without Belle: r_s 0.61). Robustness (independent check): N₂ moves by ≤ 0.08 under the variations
tried (Υ(4S) weights, N₁ = 0.8–1.2, D⁰ and D⁺ fitted separately, correlations); r_s is less robust: 0.79
with BaBar's correlated D_s normalization error (B(D_s → φπ), common to B⁰ and B⁺), 0.72 with modern
D_s normalization or with the PDG average instead of Belle, 0.61 with BaBar alone. A prior covering
this: r_s ≈ 0.80 ± 0.15.

*Definition note.* The D_s class tests for a D_s-family meson among the B daughters regardless of
its vertex; 0.16 % of B decays (PYTHIA, D_s⁻ from ss̄ popping) are in it without an upper-vertex D_s.
Small effect; an "upper-vertex D_s" definition is the alternative.

*New findings (points 3.5, 3.6).* (1) D± is over-produced: B⁰ → D⁻X (lower vertex) +2.7σ with D̄⁰X −2.7σ
(the sum is right: 85.7 against 84.3), B⁺ upper-vertex D⁺ +2.0σ, the average D± X +2.4σ against BaBar
(+5.4σ against CLEO 1997); D⁰ X agrees with Belle 2023. The D⁺/D⁰ split is outside N₂ and r_s. (2)
Upper-vertex Λ_c⁺ (baryonic b → cc̄s, Ξ_c Λ̄_c X type) is almost absent (B⁺: 0.07 % against 2.1 %);
ψ(2S) and χ_c are too low, direct J/ψ somewhat high.

*3.2 options presented (2026-11-20).* (a) correlated Gaussian prior N₂ = 0.94 ± 0.12, r_s = 0.80 ± 0.15,
correlation −0.8; (b) the same, uncorrelated; (c) centered at the MC (N₂ = 1 ± 0.15, r_s = 1 ± 0.25).
**Decision (2026-11-20): option (a), AGREED.** Reason: the correlation is physical (the D_s rates fix
N₂·r_s, the D rate N₂) and well determined; the widened r_s covers the spread from the D_s
normalization and from Belle against BaBar. The D_s-class definition of 3.1 is kept (the 0.16 % of
decays with only a lower-vertex D_s is a small effect, and the ntuple truth classifies it the same way).

### 3.5 D⁺/D⁰ composition (opened 2026-11-20)

*Facts* (generator truth, piece-2 c; scratch scripts p35.py, p35b.py of the session, logic as in
`truth_analysis.py`):
- The excess is in the lower vertex of B⁰: B⁰ → D⁻X 45.7 % against 36.9 ± 3.3 (BaBar 2007), B⁰ → D̄⁰X
  39.9 % against 47.4 ± 2.8, sum right. In B⁺ the lower-vertex D⁻ is slightly low (8.2 % against
  9.9 ± 1.2) and D̄⁰ right (77.2 against 79 ± 4).
- Source of the lower-vertex D⁻ in B⁰ (% of B⁰ decays): PYTHIA 23.2 (first-generation D⁻ 18.3, D\*⁻
  4.9), measured exclusive lines 18.1 (D⁻ 10.6, D\*⁻ 6.7, D\*\*⁻ 0.8), unmeasured exclusive lines 4.3.
- PYTHIA's single-charm decays (about 35 % / 31 % of B⁰ / B⁺ decays) produce **no orbitally excited
  charm mesons (D\*\*) at all**, and their lower-vertex charm is D or D\* with V/(P+V) = 0.455 / 0.458.
  The decay file switches L = 1 meson production off: its "Control of L=1 mesons" JetSetPar lines
  (PARJ(14)–PARJ(17) = 0.05) are commented out ("commented out by NB/WP", rel08 lines 291–295; also
  in light-2609-luna), so PYTHIA's default (no L = 1 mesons) applies. Without D\*\*, the charm of a
  B⁰'s lower vertex keeps the spectator's isospin (D⁻, D\*⁻); a charged D\*\*⁻ would go to D̄⁽\*⁾⁰π⁻ in
  2/3 of its decays. This moves D⁻ to D̄⁰ in B⁰ and D̄⁰ to D⁻ in B⁺, the direction of both deviations.
- Inclusive D\* (average of B⁰ and B⁺, MC against PDG S049.111 / S049.35, old CLEO/ARGUS): D\*± X 25.7 %
  against 22.5 ± 1.5; D\*⁰ X 27.3 % against 26.0 ± 2.7.
- Effect in our samples: PYTHIA single-charm decays with a first-generation D± are 30–40 % of the comb
  D-side yield (SR 30.2 %, comb CR 33.5 %, WC 39.6 %), 14–15 % of the comb ℓ side (RS), 1.7–4.1 % of hadB
  (RS) and 41 % of the WC hadB table.
- With a simple charge-exchange fraction ε of PYTHIA single-charm decays (moving D⁽\*⁾ of the
  spectator's isospin to the other one, as D\*\* would), BaBar's B⁰ split needs ε = 0.38 ± 0.14 and
  the B⁺ split ε = 0.08 ± 0.06; not one common value (about 2σ apart), so the mechanism explains the
  direction, not the full size, of the B⁰ excess.

*Literature check (2026-11-20, user's question: are there newer results than BaBar 2007?).* PDG 2026
listings and an INSPIRE search:
- Per B flavor (B⁰ and B⁺ separately, upper and lower vertex): BaBar 2007 (PRD 75 072002,
  hep-ex/0606026, 231·10⁶ BB̄, hadronic tag) is still the only measurement; it supersedes BaBar 2004
  (hep-ex/0408113). PDG does not rescale it to current D branching fractions.
- B⁺/B⁰ average: B → D⁰/D̄⁰ X has Belle 2023 (arXiv:2305.10098, 66.63 ± 1.77 %, Υ(4S) data, by-product of
  a B_s cross-section scan) besides CLEO 1997 and older; B → D± X has only CLEO 1997 (23.2 ± 1.2 %) and
  older (CLEO 1992, ARGUS 1991); B → D\*± X and D\*⁰ X only CLEO 1997 / ARGUS.
- Belle 2024 (arXiv:2411.14032, B_s → D X with hadronic tagging) measured B⁺ and B⁰ → D_s, D⁰, D⁺ X on
  571 fb⁻¹ of Υ(4S) data with the same method, as a systematic check, and reports only "good agreement
  with the previous Belle measurement"; the numbers are not published (they might be obtainable from
  the authors or a Belle internal note).
- No Belle II measurement found.
- Consistency of the inputs: BaBar 2007's D⁰ + D̄⁰ multiplicities, averaged over B⁰ and B⁺ (about 71.6 ±
  2.6 %), are about 1.6σ above Belle 2023's 66.63 ± 1.77 %, while the MC (67.2 %) agrees with Belle. So
  part of BaBar's high B⁰ → D̄⁰ X (and of the apparent MC deficit there) may be BaBar's; the D± side
  (MC 29.7 % against BaBar's sum about 25.4 ± 1.8 and CLEO 1997 23.2 ± 1.2) points the same way in two
  independent experiments.

*3.5 options presented (2026-11-20).* (a) a constrained isospin parameter for PYTHIA single-charm
decays, prior from BaBar 2007 per flavor; (b) no new parameter (N₁, slope a, transfer term); (c) (a) and
a report to Belle II that PYTHIA's generic B decays contain no D\*\* (L = 1 settings commented out).
**Decision (2026-11-20): option (c), AGREED.** Reason: the affected decays are 30–40 % of the comb
D-side yield and the parameter acts directly on the D⁺ yield. Given the literature check, the prior
must be wide (one per-flavor measurement, partly in tension with Belle 2023's D⁰ X); its definition
and width are set in piece 4. Possible improvement: the unpublished Belle Υ(4S) B⁺/B⁰ → D X numbers
of arXiv:2411.14032.

### 3.6 Point 3.3 (N1): the lines kept individually (opened 2026-11-20)

*Facts* (current table, after piece 2 and P7; BF in % of B⁰ / B⁺ decays; yield shares from the 72
sample tables, summed per region and table):

| Hadronic unmeasured lines outside the N₂ class | B⁰ BF | B⁺ BF | lines | largest yield share |
|---|---|---|---|---|
| single charm (D\*\*ρ, D⁽\*⁾ηπ, D(\*)X …) | 0.81 | 0.90 | 45 / 75 | 0.80 % (WC comb D side) |
| charmonium | 0.57 | 0.62 | 44 / 38 | 1.56 % (comb CR hadB) |
| – of which ψ(3770) K(\*)(π) lines (597–607; 4163–4169) | 0.13 | 0.17 | 7 / 7 | most of the charmonium yield |
| charmless (mesons and baryons) | 0.29 | 0.25 | 902 / 743 | 0.01 % |

- Current conventions are mixed: most lines are class none with c = 1 and no σ; lines feeding a
  measured total are `unmeasured`, c = 1 ± 1; limit lines c = 1 (or limit/2 ± limit/2 by exception,
  E12); 16 lines lost their 1 ± 1 when P3/P4 excluded their totals.
- The ψ(3770) lines are double open charm in practice: ψ(3770) → DD̄ in 93 % of its decays in the
  decay file (D⁰D̄⁰ 0.52, D⁺D⁻ 0.41). They are the largest individually kept lines in the samples (up to
  0.31 % each, comb CR hadB) and are unmeasured b → cc̄s, like the N₂ class, which counts open-charm
  hadrons among the B daughters only.
- Largest single-charm lines: B⁺ → D̄₀\*⁰ρ⁺ (4317, 1.66·10⁻³, 0.19 %), D̄₁⁰ρ⁺ (4319, half_limit c = 0.19, 0.17 %),
  D̄₂\*⁰ρ⁺ (4315, limit, 0.15 %), B⁰ → D₁⁻ρ⁺ (740, 0.12 %); the D\*\*ρ lines are the "2 × D\*\*π" prediction
  of the tag-mode notes.
