# B0 piece 3: exclusive modes without a measurement

Review of how the exclusive decay-file lines without a PDG measurement, and the PYTHIA decays, enter
the BBbar background model (step B0, piece 3 of `docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`, section
B.13; model parameters in B.6). Same conventions as piece 2 (`docs/B0_PIECE2_REVIEW.md`): every point
has the facts, the options, a recommendation and a decision; a decision is entered only after it was
discussed and agreed. Status labels: OPEN, AGREED (decided, not implemented), DONE (implemented, tables
rerun).

Last updated: 2026-11-27 (piece 3 complete 2026-11-20: 3.1–3.6 and P7 decided and implemented where needed; 3.7 passed to piece 4).

*Update 2026-11-27:* the items passed to piece 4 are decided there (`docs/B0_PIECE4_REVIEW.md` section 1):
starting values 4.1 (D); isospin parameter 4.3 (one common ε = 0.2 ± 0.2; the BaBar per-flavor values only
as a validation cross-check); D(\*)D̄(\*)K\* composition 4.4 (θ = −0.35 ± 0.35); D₁(2420) table 4.5;
correction of the three-body D_s Dπ externals (non-D₂\* part 0.72 → 0.778) on 2026-11-25 (3.7, piece-4 3.2).

Line numbers refer to `Samples/dec/DECAY_BELLE2_rel08.DEC` (B⁰ block 360–2160, PYTHIA lines 2134–2158;
B⁺ block 3992–5525, PYTHIA lines 5503–5523). "Unmeasured" means class none or limit in
`b_mode_pdg_table.csv` (after piece 2, P2–P6). Rates are in % of the decays of that B flavor;
inclusive rates are multiplicities (mean number per B decay), as in the measurements.

## 1. Decisions

| # | Point | Status | Decision and current values |
|---|---|---|---|
| 3.1 | Class structure: do the unmeasured exclusive b → cc̄s lines and the PYTHIA ≥ 2-charm decays form one class with one scale factor? | AGREED (2026-11-19), section 3.1 | option (e): one free normalization N₂ for all unmeasured b → cc̄s (exclusive class none/limit and PYTHIA ≥ 2 charm), plus one constrained composition parameter (the fraction of the D_s class, i.e. decays with a D_s(\*) among the B daughters), prior from the inclusive data (3.2); whether the data constrain it is decided by the Fisher matrix of piece 5 |
| 3.2 | Inclusive constraints and priors for N₂ and the D_s-class scale | AGREED (2026-11-20; prior revised the same day after 3.4), sections 3.4, 3.7 | correlated Gaussian prior **N₂ = 0.93 ± 0.13, r_s = 1.00 ± 0.25, correlation −0.6** (refit after the LHCb lines of 3.4 left the class; r_s between 0.75 and 1.13 depending on the D_s X input; the first value, r_s = 0.80 ± 0.15, is superseded). D_s class as defined in 3.1. Whether the central values also become starting weights: piece 4 |
| 3.3 | N1: lines kept individually; final class definitions | DONE (2026-11-20), section 3.6 | option (b) with classes by quark process. **N₂** = unmeasured exclusive and PYTHIA decays with a cc̄ pair (two open-charm hadrons, or charmonium of any mass, among the B daughters): b → cc̄s(d); its D_s class (r_s) as in 3.1. **N₁** = unmeasured exclusive and PYTHIA decays with a single charm quark (b → c + light quarks); the 3.5 isospin parameter acts within its PYTHIA part. Measured exclusive lines keep their c. Charmless unmeasured lines: c = 1 ± 1 per line; PYTHIA charmless: weight 1. In the piece-2 table: every hadronic line without a c has c = 1 ± 1; a limit line above its own limit c = limit/2 ± limit/2 (`limit_auto`) |
| 3.4 | Checks of the large unmeasured double-charm lines (symmetry, measurements) | DONE (2026-11-20), section 3.7 | option (a): the 12 D\*\*D_s(\*) and 8 three-body D_s(\*)Dπ lines leave N₂ and take LHCb-based c (`external`, 20 entries; B⁰ by isospin; extra 40 % on the D₁(2420) lines, 30 % on the three-body lines); D(\*)D̄(\*)K\* stays in N₂ with a composition nuisance (piece 4) and S042.609 as a closure check including PYTHIA; D_s Dππ stays in N₂·r_s; 3.2 prior to be re-derived (proposal in 3.7) |
| 3.5 | D⁺ / D⁰ composition of the MC: D± over-produced against the inclusive measurements (outside N₂, r_s) | AGREED (2026-11-20), section 3.5 | option (c): the model gets a constrained isospin parameter for PYTHIA single-charm decays (rate moved between lower-vertex charm of the spectator's isospin and the other one, sum kept); definition and prior in piece 4, from BaBar 2007 per flavor, with a width that reflects the weak evidence (single measurement; Belle 2023 D⁰ X agrees with the MC); reported to Belle II (PYTHIA without L = 1 mesons) |
| 3.6 | Upper-vertex Λ_c⁺ (baryonic b → cc̄s) almost absent in the MC; ψ(2S), χ_c too low | AGREED (2026-11-20), section 3.8 | option (a): no new parameter; known MC deficiencies, their effect on the D rates is inside N₂ (prior from the D and D_s rates) and the transfer term; reported to Belle II. Yields: charmonium among the B daughters ≤ 0.24 % (ℓ side only), charmed baryons ≤ 1.3 % (ℓ side, mostly lower-vertex Λ̄_c) |
| 3.7 | The decay file's D₁(2420) decay table: D₁ → Dππ too high relative to D\*π (affects D₁ℓν, D₁π, D₁D_s) | OPEN for piece 4 (2026-11-20), section 3.7; decided in piece 4, point 4.5 (AGREED 2026-11-25, line c DONE 2026-11-26: D\*π × 1.073, Dππ × 0.891) | measured Dππ/D\*π = 0.83 ± 0.11 of the decay file's; B(D₁ → D\*⁺π⁻) = 0.429 ± 0.019 against 0.399; used for the D₁D_s externals; the D⁺ composition of D₁ℓν (D⁺ directly from D₁) to be handled by an event weight on `D_genMotherPDG` in piece 4; reported to Belle II |
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

*3.3 options presented (2026-11-20).* (a) per line c = 1 ± 1, uncorrelated, symmetry where available,
limit lines above their limit limit/2 ± limit/2; (b) by physics as in 3.1: ψ(3770) lines to N₂,
unmeasured single-charm exclusive lines to N₁, the rest 1 ± 1 per line; (c) one common nuisance for
all individually kept lines.

*User's definition (2026-11-20).* N₂: two open-charm hadrons, or charmonium above the open-charm
threshold (ψ(3770) and heavier); N₁: whatever else has charm; PYTHIA and unmeasured decays alike. It
agrees with (b) except for charmonium below threshold (η_c, J/ψ, χ_c, h_c, η_c(2S), ψ(2S): N₁ by the
mass rule). In the decay file only ψ(3770) (3.774 GeV; 18 lines) and ψ(4160) (one measured B⁺ line)
are above the D D̄ threshold (3.730 GeV). Classifying instead by quark process puts below-threshold
charmonium in N₂ (b → cc̄s, the same PYTHIA lines that make N₂'s open charm; about 10 % of their
upper-vertex charm ends in charmonium); the numerical difference is negligible (about 0.45 % of B
decays, with no open charm).

**Decision (2026-11-20): option (b), classes by quark process, DONE.** N₂ = unmeasured exclusive and
PYTHIA decays with a cc̄ pair (two open-charm hadrons or charmonium of any mass among the B
daughters); N₁ = those with a single charm quark; measured lines keep c; charmless unmeasured lines
1 ± 1 per line, PYTHIA charmless weight 1. Reason: same physics and same lack of a measurement, same
parameter; N₂ = unmeasured b → cc̄s and N₁ = unmeasured b → c + light quarks are what the inclusive
priors of 3.2 constrain.

*Implemented (2026-11-20)* in `scripts/tools/pdg_match.py` (end of the node pass; anti-B rows without
a conjugate get the same): every hadronic (not semileptonic) line without a c now has c = 1 ± 1
(σ added on 3659 rows, 1829 of them B⁰/B⁺; no c changed); a limit line above its own limit gets
c = limit/2 ± limit/2 (how `half_limit`, new flag `limit_auto`). c changed on 17 rows (B⁰/B⁺ and
conjugates):

| Line (conjugates) | Mode | c before → after | Max share |
|---|---|---|---|
| 523–526 (2333–2336) | B⁰ → η_c(2S)K\*⁰ | 1 → 0.295 ± 0.295 | 0.015 % |
| 4337 (5880) | B⁺ → D⁺K̄⁰ (wrong strangeness, report item 1.4) | 1 → 0.091 ± 0.091 | 0.006 % (5880: 0.034 %) |
| 1659 (3469) | B⁰ → K′\*⁺π⁻ | 1 → 0.473 ± 0.473 | 0.002 % |
| 2090 (3900) | B⁰ → J/ψD̄⁰ | 1 → 0.096 ± 0.096 | 0 |
| 1181 (2991), 1811 (3620), 915 (2725), 4573 (6116) | b₁K⁰, ρ⁺b₁⁻, ηf₀, b₁⁺π⁰ | 1 → 0.39, 0.23, 0.40, 0.17 | 0 |

No node closure moved; the yield-by-class summary is unchanged. Tests: new
`test_n1_convention_33`; the toy test of a limit line above its limit now expects limit/2 ± limit/2;
36 and 15 pass. The model-level class membership (N₁, N₂, r_s, isospin parameter) is applied when the
model is built (pieces 4, 6).

*Charmless decays in the model (question of the user, 2026-11-20).* Yield shares (all 12 sample files,
e and μ, run1 and run2): measured charmless exclusive ≤ 0.003 %, unmeasured ≤ 0.013 %, PYTHIA charmless
0.09–0.16 % (comb, both sides) and ≤ 0.02 % (hadB); at most 0.17 % in any table. Our events need a D⁺,
so a charmless B enters only as the other B. Treatment: no parameter of their own; measured lines
keep c, unmeasured lines c = 1 ± 1 per line, PYTHIA charmless weight 1 under the common slope a; if
piece 5 shows they matter, one loose normalization (e.g. ±50 %) on the charmless class.

### 3.7 Point 3.4: symmetry and measurement checks of the large unmeasured double-charm lines (2026-11-20)

One checker per family and an adversarial cross-check that recomputed the key numbers (decay-file BFs,
PDG values, yield shares, LHCb ratios and fit fractions, the arithmetic of every estimate) and
checked the cited measurements in the papers. Line numbers refer to the release-08 decay file; c is the
implied correction of the line.

**Family D\*\*D_s(\*)** (B⁰ 623–628, B⁺ 4182–4187; 1.36 % per flavor). No PDG 2026 entry except CLEO 2000
(S041.264, B⁺ → D_s(\*)⁺D̄\*\*⁰ = 2.73 ± 1.15 %, old B(D_s → φπ), assumed D\*\* composition). Measured by
LHCb, JHEP 08 (2024) 165, arXiv:2405.00098 (not in PDG 2026): B(B⁺ → D\*⁻D_s⁺π⁺)/B(B⁰ → D\*⁻D_s⁺) =
0.173 ± 0.012, B(D\*⁻D_s\*⁺π⁺)/B(D\*⁻D_s⁺π⁺) = 1.32 ± 0.16, fit fractions D₁(2420) 74.8 %, D₁(2430) 14.7 %,
D₂\*(2460) 11.7 %; with PDG B⁰ → D\*⁻D_s⁺ = (8.17 ± 0.80)·10⁻³: B(B⁺ → D\*⁻D_s⁺π⁺) = (1.41 ± 0.17)·10⁻³.
Product c per line (decay-file sub-BFs D\*\* → D\*⁻π⁺; B⁰ by isospin, ÷ 1.076):

| Lines (B⁺; B⁰) | Mode | BF | Max share | c |
|---|---|---|---|---|
| 4186; 627 | D̄₂\*⁰D_s⁺ | 4.2·10⁻³ | 9.6 % | 0.15 ± 0.03 |
| 4187; 628 | D̄₂\*⁰D_s\*⁺ | 4.0·10⁻³ | 6.6 % | 0.20 ± 0.08 |
| 4184; 625 | D̄₁⁰D_s⁺ | 1.2·10⁻³ | 1.5 % | 2.2 ± 0.4 (1.3 if D₁ → D\*π were 100 %; the decay file gives D₁ → Dππ 40 %) |
| 4185; 626 | D̄₁⁰D_s\*⁺ | 2.4·10⁻³ | 2.1 % | 1.5 ± 0.5 |
| 4182; 623 | D̄₁′⁰D_s⁺ | 0.6·10⁻³ | 0.7 % | 0.52 ± 0.25 (broad state, model dependent) |
| 4183; 624 | D̄₁′⁰D_s\*⁺ | 1.2·10⁻³ | 1.4 % | 0.34 ± 0.18 |

Family total c ≈ 0.6 ± 0.1; the hierarchy is inverted (the file's D₁′:D₁:D₂\* ≈ 1:2:7 resembles CLEO's
assumption; LHCb finds D₁(2420) dominant). D₂\* → D⁻π⁺ is the family's main direct D⁺ source. The D_s\*
lines assume the D\*\* composition of D\*⁻D_s\*⁺π⁺ equals that of D\*⁻D_s⁺π⁺ (±30 %).

**Family three-body D_s(\*)Dπ** (B⁰ 629–632, B⁺ 4189–4192; 1.10 % per flavor). No PDG entry. LHCb
amplitude analyses of B⁰ → D̄⁰D_s⁺π⁻ and B⁺ → D⁻D_s⁺π⁺ (arXiv:2212.02717, PRD 108 012017; 2212.02716)
give fit fractions only (D₂\* 22 %, virtual D\* 15 %, Dπ S-wave 40–53 %); bridged to the D\*⁻D_s⁺π⁺
measurement through D₂\* (with B(D₂\* → Dπ)/B(D\*π) = 1.50 of the decay file): B(B⁺ → D⁻D_s⁺π⁺) =
(1.11 ± 0.24)·10⁻³ in total, (0.80 ± 0.19)·10⁻³ without D₂\*. *(Corrected 2026-11-25 in piece 4, `docs/B0_PIECE4_REVIEW.md` 3.2: without D₂\* is 1.115 × 0.778 = 0.87·10⁻³; the c of the eight three-body lines rose by about 8 %.)* The decay file has 3.6·10⁻³ on line 4189
alone (5.3·10⁻³ with D₂\*). c ≈ 0.20–0.22 ± 0.05 (4189, 4190, 629, 630) and 0.18–0.20 ± 0.10 for the D_s\*
lines (D_s\*/D_s = 0.9 ± 0.4 assumed). Max shares: 4189 18.9 %, 4191 11.1 %, 631 6.7 %. The phase-space shape
is also wrong (S-wave and virtual D\* dominate in data).

**Family D_s(\*)Dππ** (B⁰ 633–638, B⁺ 4193–4198; 1.48 % per flavor; max shares 4196 11.3 %, 4193 10.7 %).
No measurement (LHCb arXiv:2411.03399 observed the final states only in the D_s1(2460) window). The B⁰/B⁺
isospin mirror is broken in the file (B⁰ D_s⁺D⁻π⁰π⁰ 2.2·10⁻³ against B⁺ D_s⁺D̄⁰π⁰π⁰ 0.8·10⁻³; B⁰ D_s⁺D̄⁰π⁻π⁰
2.2·10⁻³ against B⁺ D_s⁺D⁻π⁺π⁰ 3.3·10⁻³), which moves yield to the B⁺ lines with a charged D.

**Family D(\*)D̄(\*)K\*** (B⁰ 664–677, B⁺ 4221–4234; 2.75 % per flavor). No PDG value or limit. An
unpublished LHCb thesis (H. Bernstein, Syracuse 2022, INSPIRE 2649734) measured 11 D(\*)D̄(\*)K⁺π⁻
branching fractions in the K\*⁰ window; not readable here. Symmetry estimate: B(K\* line) = R × B(the
measured K mode with the same charges, isospin-averaged) × phase-space factor, with a common
R = B(DDK\*)/B(DDK) = 0.8 ± 0.4 (a judgement): family total 2.1 ± 1.1 % (B⁰), 2.3 ± 1.2 % (B⁺), c ≈ 0.8 ± 0.4,
compatible with the file. The file's round 1:1:1:2 pattern (DD:D\*D:DD\*:D\*D\*) is not supported: the DD
lines come out at c ≈ 0.3–0.4 (669 D⁻D⁺K\*⁰ 0.28, 4221 D̄⁰D⁺K\*⁰ 0.40), D\*D lines 0.5–1.6. Colour-suppressed
line 674 (D⁰D̄⁰K\*⁰, 5·10⁻⁴) sits at the bound from LHCb's B⁰ → D⁰D̄⁰K⁺π⁻ (S042.609, arXiv:2007.04280,
3.51 ± 0.47·10⁻⁴); the MC total of that final state, PYTHIA included, is about 10·10⁻⁴, about 3 times the
measurement (PYTHIA alone 5.2·10⁻⁴). Inclusive wrong-sign D does not separate D(\*)D̄(\*)K\* from PYTHIA.

**Consequence for 3.2.** With the LHCb-based c on the D\*\*D_s and three-body D_s Dπ lines, the MC's
D_s X drops from about 12.4 % to about 11.0 %, consistent with Belle 2023 (11.28 ± 0.43 %); the remaining
D_s class (D_s Dππ lines and PYTHIA) would need a scale of about 1.1 ± 0.2 (Belle) or about 0.85 (PDG
average). The prior r_s = 0.80 ± 0.15 of 3.2 assumed all unmeasured D_s lines scale together and would
have to be re-derived.

**Decision (2026-11-20): option (a), DONE.** Options presented: (a) use the LHCb measurements for the
D\*\*D_s and three-body D_s Dπ lines (out of N₂, `external` c), keep D(\*)D̄(\*)K\* and D_s Dππ in N₂,
refit the 3.2 prior; (b) keep all four families in N₂; (c) as (a) and rescale D(\*)D̄(\*)K\* to the
symmetry estimates. Reason: these are measurements, the lines carry up to 19 % of the yield, and they
explain most of the D_s excess.

*Implemented (2026-11-20).* 20 `external` entries (exceptions 79 → 99), source LHCb arXiv:2405.00098
and arXiv:2212.02717, each with its derivation in `reason`; errors include 40 % of the value on the four
D₁(2420) lines (the unmeasured D₁ → Dππ of the decay file) and 30 % on the eight three-body lines (the
bridge between the two analyses). c, B⁰ / B⁺ lines (anti-B conjugates follow):

| B⁰ line | B⁺ line | Mode | c B⁰ | c B⁺ | max share B⁰ / B⁺ |
|---|---|---|---|---|---|
| 623 | 4182 | D₁′D_s⁺ | 0.48 ± 0.23 | 0.52 ± 0.25 | 0.26 / 0.73 % |
| 624 | 4183 | D₁′D_s\*⁺ | 0.32 ± 0.17 | 0.34 ± 0.18 | 0.33 / 1.38 % |
| 625 | 4184 | D₁D_s⁺ | 2.05 ± 0.89 | 2.21 ± 0.95 | 0.52 / 1.53 % |
| 626 | 4185 | D₁D_s\*⁺ | 1.38 ± 0.74 | 1.46 ± 0.80 | 0.79 / 2.12 % |
| 627 | 4186 | D₂\*D_s⁺ | 0.136 ± 0.029 | 0.148 ± 0.029 | 1.07 / 9.57 % |
| 628 | 4187 | D₂\*D_s\*⁺ | 0.19 ± 0.07 | 0.21 ± 0.08 | 3.14 / 6.61 % |
| 630 / 629 | 4189 / 4190 | D_s⁺Dπ | 0.20 ± 0.08 / 0.21 ± 0.08 | 0.22 ± 0.09 / 0.22 ± 0.08 | 0.37, 3.05 / 18.9, 0.20 % |
| 632 / 631 | 4191 / 4192 | D_s\*⁺Dπ | 0.18 ± 0.11 / 0.18 ± 0.11 | 0.20 ± 0.11 / 0.20 ± 0.12 | 0.36, 6.69 / 11.1, 0.23 % |

All were class none with c = 1 ± 1. BF by class: measured_external 0.03 → 2.49 % (B⁰ and B⁺), none
10.86 → 8.40 % (B⁰), 10.43 → 7.98 % (B⁺). Yield by class: "had. none" falls by 1.9–8.8 points per table
(SR hadB 25.5 → 16.8 %), "had. exact" rises accordingly. New test `test_lhcb_ds_lines_34`; the coverage
test's bound on class none lowered to 7 %; 37 and 15 tests pass.

*Refit of the 3.2 prior* (generator truth, `truth_analysis.py` section 5, with these lines now measured):
the MC with N₂ = r_s = 1 now gives D_s X 11.00 % (average), D_s⁺ X 9.62 / 9.65 %, wrong-sign D 11.48 / 11.37 %.
The D_s class (now D_s Dππ lines and PYTHIA D_s) is 3.00 % / 2.85 % of B decays.

| D_s X input | N₂ | r_s | correlation | χ² (dof) |
|---|---|---|---|---|
| Belle 2023 11.28 ± 0.43 % (as in 3.2) | 0.93 ± 0.13 | 1.13 ± 0.21 | −0.75 | 2.1 (3) |
| PDG average 10.6 ± 0.6 % | 0.94 ± 0.13 | 0.90 ± 0.22 | −0.58 | 1.5 (3) |
| PDG average with its scale factor (± 1.0 %) | 0.94 ± 0.13 | 0.87 ± 0.29 | −0.44 | 1.4 (3) |
| BaBar 2007 only | 0.94 ± 0.13 | 0.75 ± 0.45 | −0.30 | 1.3 (2) |

N₂ is stable; r_s moved up from 0.83 and now depends on the D_s X input between 0.75 and 1.13.
Proposal for a revised 3.2 prior (to decide): N₂ = 0.93 ± 0.13, r_s = 1.00 ± 0.25, correlation −0.6.

**Decision (2026-11-20): the 3.2 prior is replaced** by N₂ = 0.93 ± 0.13, r_s = 1.00 ± 0.25, correlation −0.6.
Reason: with the D\*\*D_s and three-body D_s Dπ lines measured, r_s scales only the D_s Dππ lines and
PYTHIA's D_s; its value then depends on the D_s X input (0.75–1.13), which the width covers.

*The two caveats of 3.4 (user's question, 2026-11-20).* (1) LHCb measures B → D₁(2420)D_s only through
D₁ → D\*π; the line's c divides by the decay file's B(D₁ → D\*⁺π⁻) = 0.399 (the file gives D₁ → D\*π
60 %, D₁ → Dππ 40 %; PDG has no absolute D₁ branching fractions). (2) The three-body D_s Dπ lines rest on
fit fractions only (arXiv:2212.02717), bridged to the absolute D\*⁻D_s⁺π⁺ rate through the D₂\* component
with B(D₂\* → Dπ)/B(D₂\* → D\*π) = 1.50 of the decay file; PDG confirms the ratio (BaBar 2010: 1.47 ± 0.16;
M254R3). The remaining uncertainty is the model dependence of the fit fractions (D₂\* fraction stable,
22.1 / 22.2 / 22.4 %); the 30 % extra is kept until an absolute B⁺ → D⁻D_s⁺π⁺ measurement exists; the MC
D_s X after the change (11.0 %) agrees with Belle's 11.28 ± 0.43 %.

*D₁ → Dππ / D\*π, measured.* Same-paper ratios (so that common systematics cancel), relative to the
decay file's D⁰π⁺π⁻/D\*⁺π⁻ = 0.430:

| Source | Dππ / D\*π | Ratio to the decay file |
|---|---|---|
| Belle 2023 (arXiv:2211.09833, PRD 107 092003), B⁺ → D̄₁⁰ℓν: 1.05 / 2.49 (·10⁻³) | 0.42 ± 0.07 | 0.98 ± 0.17 |
| Belle 2023, B⁰ → D₁⁻ℓν: 1.02 / 3.06 (·10⁻³) | 0.33 ± 0.08 | 0.77 ± 0.19 |
| LHCb 2011 (arXiv:1109.6831, PRD 84 092001), B⁺ → D̄₁⁰π⁺: 4.0 % / (9.3 %/0.677) | 0.29 ± 0.09 | 0.68 ± 0.20 |
| mean | | 0.83 ± 0.11 (χ² = 1.5 for 2 dof) |

The PDG averages of the products give the same picture (c from the Dππ product / c from the D\*π product:
0.83, 0.86, 0.61 for lines 368, 4000, 4318). With the decay file's isospin pattern of D₁ → Dππ:
B(D₁ → D\*⁺π⁻) = 0.429 ± 0.019 (decay file 0.399); the c of a line measured through D₁ → D\*π scales by
0.932 ± 0.042.

**Decision (2026-11-20): option (a), DONE.** The four D₁(2420)D_s externals are divided by 0.429 instead
of 0.399, with an extra 10 % (that fraction and the unmeasured isospin pattern of D₁ → Dππ) in place of
40 %: 625 2.05 ± 0.89 → 1.91 ± 0.38; 626 1.38 ± 0.74 → 1.28 ± 0.48; 4184 2.21 ± 0.95 → 2.06 ± 0.39; 4185
1.46 ± 0.80 → 1.36 ± 0.52 (anti-B conjugates follow). The bridge uncertainty (30 %) is kept. Tests: the
expected c of 4184 updated; 37 and 15 pass. Effect on the 3.2 fit: these lines are outside the fitted
class and carry little upper-vertex D_s change (about 0.02 % of B decays), so the prior stands. New
point 3.7 (piece 4): the D₁ table in B → D₁ℓν (lines 368/392, 4000/4027; 0.70 / 0.76 % of B decays) gives
about 20 % too many D⁺ directly from D₁ (D₁ → D⁺π⁻π⁰); a line weight cannot correct a sub-decay, an event
weight on the D's mother (`D_genMotherPDG` = D₁) can.

### 3.8 Point 3.6: upper-vertex Λ_c⁺ and the charmonium composition (2026-11-20)

*Facts* (generator truth, section 3.4; yield shares from the 72 sample tables): MC upper-vertex Λ_c⁺ X
0.09 % (B⁰) / 0.07 % (B⁺) against BaBar 2007 1.6 ± 0.9 % (paper; PDG < 3.1 %) / 2.1 +0.9 −0.6 % (−3.4σ);
lower-vertex Λ̄_c⁻ X agrees. The measurements are not fully consistent: CLEO 1997's upper-vertex fraction
of Λ_c (0.19 ± 0.14; limit Λ_c⁺ X < 1.3 % for the B average) against BaBar 2007's 0.43 ± 0.07. The measured
exclusive modes with an upper-vertex Λ_c⁺ (PDG 2026: B⁺ → Λ_c⁺Ξ̄_c⁰ 9.5·10⁻⁴ (S041.721), Λ_c⁺Ξ̄_c(2790)⁰
1.1·10⁻³ (S041.734), Ξ_c(2930)Λ_c⁺ 1.7·10⁻⁴ (S041.715)) add up to about 0.2 %, far below BaBar's
inclusive 2.1 %. Charmonium: J/ψ X right (1.08 % against 1.094 ± 0.032), ψ(2S) −3.2σ, χ_c1 −4.6σ, χ_c2 −2.5σ,
direct J/ψ +1.9σ. B decays with charmonium among the daughters: 0.20–0.24 % of the ℓ-side yield, 0 on the
D side; with a charmed baryon: 1.2 % of the RS ℓ side (mostly lower-vertex Λ̄_c from PYTHIA), ≤ 0.07 % on
the D side.

*Correction (2026-11-20).* In the presentation of 3.6 I wrote that "the data suggest about 25 %" of
PYTHIA's upper-vertex charm should become a Λ_c. That number is not a branching fraction and not a
measured fraction: it is BaBar's B⁺ → Λ_c⁺X (2.1 %) divided by the PYTHIA lines with an upper-vertex c
quark (7.4 %), i.e. the share PYTHIA would need if all upper-vertex Λ_c⁺ had to come from PYTHIA (the
only source in the MC). Relative to all upper-vertex charm (23.4 % of B⁺ decays, BaBar 2007), 2.1 % is
about 9 %; in the MC it is 0.3 %.

*Options presented.* (a) no new parameter, record, report to Belle II; (b) reweight the charmonium
composition by event weights on the B daughters; (c) a dedicated class for the missing Λ_c⁺ (impossible
by reweighting: events that do not exist cannot be weighted up).

**Decision (2026-11-20): option (a), AGREED.** Reason: charmonium is at most 0.24 % of the ℓ-side yield;
the Λ_c⁺ deficit cannot be reweighted, and its effect on the D and D_s rates is already inside the N₂
prior, which is fitted to them. Reported to Belle II (item 6c).

## 4. Piece 3: summary and what passes to piece 4 (2026-11-20)

Model classes (B.6 updated): measured exclusive lines keep their piece-2 c (now including 20 LHCb-based
`external` lines, 3.4); **N₂** = unmeasured b → cc̄s(d) (exclusive class none/limit and PYTHIA, with a cc̄
pair: two open-charm hadrons or charmonium of any mass), prior N₂ = 0.93 ± 0.13 with its D_s class
scaled by r_s = 1.00 ± 0.25, correlation −0.6 (3.1, 3.2, 3.3); **N₁** = unmeasured b → c + light quarks
(single charm quark, exclusive and PYTHIA), with an isospin parameter for its PYTHIA part (3.5);
charmless: no parameter (3.3); hadronic lines without a c: c = 1 ± 1, limit lines above their limit
limit/2 ± limit/2 (3.3).

For piece 4: c as weights or only as prior widths; correlations (e.g. the BaBar 2011 D D̄ K modes share
systematics); the isospin parameter of 3.5 (definition, prior from BaBar 2007, wide); a composition
nuisance R for the D(\*)D̄(\*)K\* lines (3.4) and B⁰ → D⁰D̄⁰K⁺π⁻ (S042.609) as a closure check including
PYTHIA; the D₁(2420) decay table (3.7: event weight on `D_genMotherPDG`); r_SL and its split; τ modes;
the semileptonic gap modes (with the Belle II recommendation).
