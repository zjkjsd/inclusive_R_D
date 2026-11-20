# Issues found in the Belle II B-meson decay file (release 08) by a comparison with PDG 2026

Date: 2026-11-19
Contact: _(name, analysis group)_
Context: inclusive tagged R(D⁺) analysis; background modeling of generic BB̄ MC (MC16rd)

## Summary

While building a data-driven model of the BB̄ background, we compared every exclusive B⁰ and B⁺ line
of the generic decay file with the 2026 edition of the PDG listings. Most lines agree with PDG, or are
explained by known conventions, but we found a set of lines that are clearly wrong, double-count a
measured rate, or are superseded by newer measurements. We list them below so that they can be
considered for the next update of the decay file. Each item gives the decay-file line, the problem,
the evidence and a suggested fix. The effect on generic MC is given as a fraction of the B decays of
that flavor.

**File.** `decfiles/dec/DECAY_BELLE2.DEC` of basf2 release-08 (byte-identical in release-08-00-10 and
release-08-03-00; the master decay file of the MC16rd generic production). Line numbers refer to this
file. B⁰ block: lines 360–2160; B̄⁰: 2170–3970; B⁺: 3992–5525; B⁻: 5535–7068.

**Still present in the current release?** Checked on 2026-11-19 against `decfiles/dec/DECAY_BELLE2.DEC`
of basf2 light-2609-luna: every item below is still present, with the same text and BF, except item
2.1, which was corrected in 2025 ("BF update – VG 2025": the four D_s0\*(2317) lines now carry the PDG
products). The PYTHIA lines were renormalized (e.g. b̄ → c̄cs̄ 7.19 → 7.52 % in B⁰). Line numbers in
light-2609-luna are shifted:

| release-08 line | light-2609-luna line | release-08 line | light-2609-luna line |
|---|---|---|---|
| 453 | 461 | 1002 | 1010 |
| 464 | 472 | 2616 | 2624 |
| 652 | 660 | 2812 | 2820 |
| 658 | 666 | 4087 | 4095 |
| 692 | 700 | 4162 | 4170 |
| 693 | 701 | 4210 | 4218 |
| 733 | 741 | 4216 | 4224 |
| 797, 798, 799 | 805, 806, 807 | 4269 | 4277 |
| 801, 802 | 809, 810 | 4319 | 4329 |
| 806 | 814 | 4337, 4338 | 4347, 4348 |
| 818 | 826 | 4346, 4347 | 4356, 4357 |
| | | 4349, 5888 | 4359, 5900 |

**Method.** All B⁰/B⁺ branching fractions of PDG 2026 (1273 entries, all levels) were read through the
PDG API. For every PDG final state, the decay-file rate was computed by following the decay file's own
sub-decays of the resonances, and the two were compared entry by entry; papers were read where a
number mattered. Where the issue concerns a line with a CP-conjugate line, the conjugate has the same
problem unless stated otherwise.

## 1. Lines with the wrong flavor or quantum numbers

| # | Line (conjugate) | Decay-file line | Problem | Evidence | Suggested fix |
|---|---|---|---|---|---|
| 1.1 | 806 (2616) | `0.0027185 D*0 pi+ pi+ pi- pi-` in the **B⁰** block | Wrong charm flavor: a B⁰ (b̄) gives D̄\*⁰. 0.27 % of B⁰ decays are generated with a wrong-sign D⁰ (and the corresponding cascade-lepton charge). The right-flavor mode B⁰ → D̄\*⁰π⁺π⁺π⁻π⁻ is absent from the block | PDG S042.195 writes "B0 → D\*(2007)0 π⁺π⁺π⁻π⁻" without the bar, but its sources are titled "First observation of **B̄⁰** → D\*⁰π⁺π⁺π⁻π⁻" (CLEO, Edwards 2002) | Replace by `anti-D*0 pi+ pi+ pi- pi-` in the B⁰ block (and `D*0 ...` in the B̄⁰ block), same BF |
| 1.2 | 1002 (2812) | `1.0e-5 anti-Lambda0 Lambda0 D0` in the B⁰ block | Same: should be D̄⁰ | PDG S042.457 "B0 → Λ̄ΛD0" without bar; source: BaBar 2014, "Evidence for the baryonic decay **B̄⁰** → D⁰ΛΛ̄" (arXiv:1401.5990) | `anti-Lambda0 Lambda0 anti-D0` in the B⁰ block |
| 1.3 | 4349 (5888) | `0.0015 anti-D*0 K+ K*0` in the B⁺ block | ΔS = 2 (two s̄ quarks): not allowed. The allowed mode D̄\*⁰K⁺K̄\*⁰ is line 4345 (5.9·10⁻⁴). The B⁻ block has the two BFs swapped (5888 forbidden at 5.9·10⁻⁴, 5892 allowed at 1.5·10⁻³) | Belle 2002 (hep-ex/0207041) measured B⁻ → D\*⁰K⁻K\*⁰ = 15.3·10⁻⁴, i.e. B⁺ → D̄\*⁰K⁺K̄\*⁰; the K\*⁰ was not flipped when conjugating. The 2021 update (line 4345: total minus the D̄\*⁰a₁⁺ part; PDG S041.289 averages Belle 2002 and Belle II 2024) left the old line in place | Remove 4349 and 5888; keep one allowed line per block with the same BF |
| 1.4 | 4337, 4338 (5880, 5881) | `1.1e-5 D+ anti-K0`, `6e-6 D*+ anti-K0` | Wrong strangeness: the annihilation modes are B⁺ → D⁽\*⁾⁺K⁰ | PDG: B⁺ → D⁺K⁰ < 2·10⁻⁶ (S041.398; the decay file's 1.1·10⁻⁵ is above it), B⁺ → D\*⁺K⁰ < 9·10⁻⁶ (S041.269) | `D+ K0`, `D*+ K0`, at or below the PDG limits |
| 1.5 | 797, 798 (2607, 2608) | `3.0e-4 D*- K0 pi+` and `3.0e-4 D*+ K0 pi-` | The measured charge sum is written on each charge combination (rate counted twice) | BaBar 2005 (hep-ex/0412040) measures B(B⁰ → D\*∓K⁰π±) = (3.0 ± 0.7 ± 0.3)·10⁻⁴, both combinations together, of which D\*∓K\*± = 3.2·10⁻⁴ is resonant (line 754) | Split the sum between the two lines; after the resonant D\*⁻K\*⁺ line (754), the remainder for both lines together is (0.8 ± 1.0)·10⁻⁴ |

## 2. Branching fractions copied onto the wrong lines

| # | Lines | Decay file | PDG | Problem | Suggested fix |
|---|---|---|---|---|---|
| 2.1 (**already fixed in light-2609-luna**, lines 798, 799, 4349, 4350) | 790, 791 (B⁰ → D⁻D_s0\*(2317)⁺, D\*⁻D_s0\*⁺); 4339, 4340 (B⁺ → D̄⁰D_s0\*⁺, D̄\*⁰D_s0\*⁺) | 0.738 %, 1.769 %, 0.762 %, 1.706 % | (0.105 ± 0.016) %, (0.150 ± 0.060) %, (0.079 ± 0.014) %, (0.090 ± 0.070) % (products with D_s0\* → D_sπ⁰, S042.253, S042.97, S041.321, S041.26) | The four values equal, to all digits, the PDG values of the **D_s\*⁺** modes (D⁻D_s\*⁺ etc.): copied onto the D_s0\* lines. The decay file is 7–19 times above PDG; together 2.5 % of B⁰ and 2.5 % of B⁺ decays, much of it D_s production | Use the PDG products (with B(D_s0\* → D_sπ⁰) = 100 +0 −20 %); return the difference to the budget of the b → cc̄s modes |
| 2.2 | 818 (B⁰ → Σ̄_c(2455)⁰pπ⁻) | `1e-5`, comment "Gamma_506 PDGLive ⇒ UL<0.31e-4. Set BF e-5" | (1.08 ± 0.09)·10⁻⁴ (S042.223) | The comment cites the upper limit of the **Σ_c(2520)⁰** mode; the Σ_c(2455)⁰ mode is measured, 11 times higher | 1.08·10⁻⁴ |

## 3. Inclusive totals written as a non-resonant line, with resonance lines on top

The PDG inclusive total of a final state is written into the phase-space ("non-resonant") line, while
resonance lines feed the same final state, so the final state is generated above its measured rate.
The two Belle II tag-mode notes (BELLE2-NOTE-PH-2022-046 and -2022-002) fixed this pattern for the
B → D(\*)nπ tag modes ("LMH 06/2023", "NR 06/2023" lines); the following cases remain in release 08.
"c" is the factor that makes the final state agree with PDG (direct line = PDG total minus the
resonance lines, clamped at 0).

| # | Final state (direct line) | PDG total | Decay file (final state) | c of the direct line | Notes |
|---|---|---|---|---|---|
| 3.1 | B⁰ → D̄⁰π⁺π⁻ (692) | (8.80 ± 0.46)·10⁻⁴ (S042.1) | 16.5·10⁻⁴: the total on line 692, plus D̄⁰ρ⁰ (778), D₀\*⁻π⁺ (738), D₂\*⁻π⁺ (739) | 0.28 ± 0.16 | amplitude analyses (Belle 2007, LHCb 2015) |
| 3.2 | B⁰ → D̄\*⁰π⁺π⁻ (693) | (6.2 ± 2.2)·10⁻⁴ (S042.239) | 16.0·10⁻⁴: line 693 = the total, plus D₁⁻π⁺ (736), D₁′⁻π⁺ (737), D₂\*⁻π⁺ (739), D̄\*⁰ρ⁰ (779) | 0 (clamped) | |
| 3.3 | B⁰ → D⁻K⁰π⁺ (799) | (4.9 ± 0.9)·10⁻⁴ (S042.313) | 7.9·10⁻⁴: line 799 = the total, plus D⁻K\*⁺ (755) | 0.39 ± 0.23 | |
| 3.4 | B⁺ → D⁺D⁻K⁺ (4216) | (2.2 ± 0.7)·10⁻⁴ (S041.336) | 4.4·10⁻⁴: line 4216 = the full total, plus ψ(3770)K⁺ (4162) and ψ(4160)K⁺ (4171) | 0.43 ± 0.48 | LHCb amplitude analysis (arXiv:2009.00026): ψ(3770) is 14.5 % of the total |
| 3.5 | B⁺ → D̄⁰D⁰K⁺ (4210), B⁰ → D⁻D⁺K⁰ (652), B⁰ → D̄⁰D⁰K⁰ (658) | 13.1, 7.5, 2.7 (·10⁻⁴) | 17.2, 9.5, 5.2 (·10⁻⁴) with the ψ(3770)K lines | 0.89, 0.87, 0.54 | |
| 3.6 | The D(\*)D̄\*K modes fed by D_s1(2536)⁺ → D\*K (lines 801, 802, 4346, 4347) | BaBar 2011 (arXiv:1011.3929) measures the 22 D(\*)D̄(\*)K modes "including resonant modes" (D_s1(2536) inside) | 22-mode sum: 3.84 % (B⁰), 4.29 % (B⁺) with the resonance lines | 0.94–0.98 for the D\*K lines | measured sums: (3.68 ± 0.26) %, (4.05 ± 0.30) % |
| 3.7 | B⁰ → J/ψK⁰π⁺π⁻ (453) | (4.46 ± 0.40)·10⁻⁴ (S042.211) | 20.6·10⁻⁴ (4.6 times PDG): line 453 = the total, plus J/ψK⁰ρ⁰ (468), J/ψK\*⁺π⁻ (469), J/ψK₁⁰ (457), J/ψK₂\*⁰, J/ψK₁′⁰ | 0 (clamped) | |
| 3.8 | B → J/ψωK (464; 4087) | (2.3 ± 0.4), (3.2 ± 0.6) (·10⁻⁴) | 3.9, 5.3 (·10⁻⁴): the total plus J/ψK₁ (457; 4079) with K₁ → Kω | 0.31, 0.33 | |

Suggested fix: as in the tag-mode notes, set the direct line to the remainder after the resonance
lines, or to 0 where the resonances saturate the total.

## 4. Lines above a PDG limit

| # | Line | Decay file | PDG | Notes |
|---|---|---|---|---|
| 4.1 | 4319, B⁺ → D̄₁(2420)⁰ρ⁺ (comment "rho+ decays are double BF of pi+ decays", NR 06/2023) | 3.72·10⁻³ | < 1.4·10⁻³ (S041.215, 90 % CL) | 2.7 times above the limit; the "2 × D\*\*π" rule is a prediction |

## 5. New measurements not yet in the file

| # | Lines | Decay file | Measurement | Suggested value |
|---|---|---|---|---|
| 5.1 | 733 (B⁰ → D\*⁻ηπ⁺), 4269 (B⁺ → D̄⁰ηπ⁺); comment "Limit provided to invalidate eta as a candidate for SL gap (to be measured)" | 2.56·10⁻⁴, 4.0·10⁻⁴ | Belle + Belle II, BELLE2-PUB-PH-2025-024: (2.78 ± 0.24 ± 0.19)·10⁻⁴, (5.70 ± 0.26 ± 0.44)·10⁻⁴ (D_s → ηπ band vetoed) | the measured values; about half of B⁰ → D\*⁻ηπ⁺ is D̄\*\*⁰η and about 60 % of B⁺ → D̄⁰ηπ⁺ is D̄⁰a₀/a₂, which phase space does not describe |
| 5.2 | 4162, B⁺ → ψ(3770)K⁺ | 4.88·10⁻⁴ | PDG's total (S041.360, 4.34·10⁻⁴) still contains Belle 2004's value, superseded in the per-channel products (S041.361/362); the products give about 2.6·10⁻⁴, LHCb's fit fraction about 0.8·10⁻⁴ | worth a review |

## 6. Observations on the b → cc̄s budget (no single line wrong)

- 59 exclusive double-charm lines without a PDG measurement carry 6.7 % of B⁰ and 6.7 % of B⁺ decays:
  D(\*)D̄(\*)K\* (2.75 %; lines 664–677, 4221–4234), D_s(\*)D(\*)π(π) (2.58 %; 629–638, 4189–4198),
  D\*\*D_s(\*) (1.36 %; 623–628, 4182–4187). None has a comment giving its source; most sit at round
  values (2.5·10⁻³, 5·10⁻³, 5·10⁻⁴). The PYTHIA b̄ → c̄cs̄ lines (2147, 5512) add about 7 %.
- Against the inclusive measurements: in release 08 the exclusive lines alone give B → D_s X = 12.1 %
  (average of B⁰ and B⁺), against (10.6 ± 0.6) % (PDG; Belle 2023, arXiv:2305.10098: 11.28 ± 0.43 %),
  before any D_s from PYTHIA. Most of this excess is the D_s0\* lines of item 2.1 (fixed in
  light-2609-luna); with all measured exclusive lines at their PDG values the exclusive D_s X is about
  9.8 %. The upper-vertex (wrong-sign) non-strange D remains low: D⁰ + D⁺ from the exclusive lines is
  7.4 % (B⁺) against (11.1 ± 0.9) % (BaBar 2007, hep-ex/0606026), so it must come from PYTHIA or from
  the unmeasured D(\*)D̄(\*)K\* lines. These numbers should be redone for the current file.
- Inclusive charmonium from the exclusive lines is low: J/ψ X 0.88 % (with feed-down) against
  1.094 ± 0.032 %; whether PYTHIA's cc̄s lines make up the difference was not checked.
- Stale comments: the PYTHIA section headers say "Lam_c X / Sigma_c X 4.0 %" and "Xi_c X 2.5 %" above
  lines that now sum to 3.2 % (B⁰) / 3.0 % (B⁺) and 0.87 % / 0.82 %; the inline `Gamma_N` numbers no
  longer match the PDG listing numbers (0 of 42 checked).

## 5b. b → cc̄s lines without a source, now measured or constrained

| # | Lines (release 08) | Decay file | Measurement | Ratio |
|---|---|---|---|---|
| 5b.1 | 4186, 4187; 627, 628: B → D₂\*(2460)D_s(\*) | 4.2, 4.0 (·10⁻³) | LHCb arXiv:2405.00098 (D\*⁻D_s⁺π⁺ amplitude analysis, D₂\* fit fraction 11.7 %): about 0.6, 0.8 (·10⁻³) | 5–7 times too high |
| 5b.2 | 4184, 4185; 625, 626: B → D₁(2420)D_s(\*) | 1.2, 2.4 (·10⁻³) | same paper, D₁(2420) fit fraction 75 %: about 2.6, 3.5 (·10⁻³) with the file's D₁ → D\*π = 40 %; 1.6, 2.1 with 100 % | 1.3–2.2 times too low |
| 5b.3 | 4182, 4183; 623, 624: B → D₁(2430)D_s(\*) | 0.6, 1.2 (·10⁻³) | same paper, 14.7 %: about 0.3, 0.4 (·10⁻³) | 2–3 times too high |
| 5b.4 | 4189–4192; 629–632: B → D_s(\*)Dπ (phase space) | 3.6, 1.8, 3.7, 1.8 (·10⁻³) | LHCb arXiv:2212.02717 (fit fractions; D₂\* bridge to arXiv:2405.00098): B(B⁺ → D⁻D_s⁺π⁺) ≈ (1.1 ± 0.3)·10⁻³ in total | about 4–5 times too high; in data the Dπ S-wave and a virtual D\* dominate, not phase space |
| 5b.5 | 633–638; 4193–4198: B → D_s(\*)Dππ | B⁰ and B⁺ totals equal, charge split not isospin-symmetric (e.g. B⁰ D_s⁺D⁻π⁰π⁰ 2.2·10⁻³ against B⁺ D_s⁺D̄⁰π⁰π⁰ 0.8·10⁻³) | no measurement | – |
| 5b.6 | 664–677; 4221–4234: B → D(\*)D̄(\*)K\* | round values, DD : D\*D : DD\* : D\*D\* = 1:1:1:2 | no measurement (an LHCb thesis, Syracuse 2022, measured 11 of them in the K\*⁰ window, unpublished); the measured D(\*)D̄(\*)K modes have a different D\* pattern (two-pseudoscalar lines about 3 times too high by symmetry) | family total compatible |
| 5b.7 | 674 and ψ(3770)K\*⁰ (597–600), plus PYTHIA | B⁰ → D⁰D̄⁰K⁺π⁻ about 10·10⁻⁴ in generic MC | LHCb arXiv:2007.04280: 3.51 ± 0.47·10⁻⁴ | about 3 times too high (PYTHIA alone 5.2·10⁻⁴) |

| 5b.8 | D₁(2420) decay table (`Decay D_10`, lines 9632–9637, and its charged partner) | D\*π 60 %, Dππ 40 % (D⁰π⁺π⁻ 17.2 %, D⁰π⁰π⁰ 11.5 %, D⁺π⁻π⁰ 11.5 %) | same-paper ratios Dππ/D\*π: Belle 2023 (arXiv:2211.09833) B → D₁ℓν, B⁺ 0.42 ± 0.07, B⁰ 0.33 ± 0.08; LHCb 2011 (arXiv:1109.6831) B⁺ → D₁π⁺ 0.29 ± 0.09; decay file 0.43 | Dππ about 20 % too high relative to D\*π (B(D₁ → D\*⁺π⁻) 0.43 ± 0.02 instead of 0.40); affects the D⁺/D⁰ content of B → D₁ℓν |

None of the lines in 5b.1–5b.6 carries a comment giving its source; the LHCb results of 5b.1–5b.4 are not
yet in the PDG listings (2026 edition).

## 6b. PYTHIA configuration: no orbitally excited charm mesons in generic B decays

- The decay file's PYTHIA settings for L = 1 mesons are commented out ("Control of L=1 mesons ...
  commented out by NB/WP"; `#JetSetPar PARJ(14)=0.05` … `PARJ(17)=0.05`, release-08 lines 291–295, also
  in light-2609-luna), so PYTHIA's default (no L = 1 mesons) applies.
- In the generator truth of MC16rd generic MC (one mixed and one charged file, experiment 18), the
  PYTHIA single-charm B decays (about 35 % of B⁰ and 31 % of B⁺ decays) contain no D\*\* at all; their
  lower-vertex charm is D or D\* with V/(P+V) = 0.46. Without D\*\* (whose charged states go to
  D̄⁽\*⁾⁰π⁻ in 2/3 of their decays), the lower-vertex charm keeps the spectator's isospin.
- Against BaBar's flavor-tagged inclusive rates (hep-ex/0606026), the MC gives B⁰ → D⁻X = 45.7 %
  (36.9 ± 3.3 %) and B⁰ → D̄⁰X = 39.9 % (47.4 ± 2.8 %), with the right sum; the B-averaged D± X is 29.7 %
  against about 25 % (BaBar sum; CLEO 1997: 23.2 ± 1.2 %). The B-averaged D⁰ X agrees with Belle 2023.
- If the L = 1 setting was switched off deliberately (the D\*\* modes are exclusive lines of the
  file), the isospin mixing that D\*\* would bring into the other PYTHIA decays is missing as a side
  effect. Worth a look by the generator experts; for analyses with a D⁺ (or D⁰) in the final state the
  D⁺/D⁰ composition of generic B decays is affected.

## 7. Related PDG listing issues (for information)

These affect anyone who copies PDG values into a decay file:

- S042.195 and S042.457: the bar is missing on the charm meson (items 1.1, 1.2).
- S042.124 and S042.385: "D₂\*(2460)⁻ → D⁰π⁻" and "D₀\*⁻ → D⁰π⁻" should read D̄⁰π⁻ (charge conservation).
- S042.155 (B⁰ → D\*⁺D⁻ = 6.12·10⁻⁴) equals the CP-averaged sum S042.511 (D±D\*∓ = 6.14·10⁻⁴), so read
  literally it double counts one charge state (Belle, arXiv:1203.6647).
- S041.797–799 (LHCb 2023, B⁺ → (K_S⁰Kπ)K, arXiv:2304.14891): the four-body totals and χ_c0K⁰π⁺ appear
  to be normalized 2/B(K_S → π⁺π⁻) = 2.89 times higher than the paper's own ratios times its reference
  imply (eq. 35, Tables 21–24); our reading, not confirmed.

Details of every item (PDG ids, sources, numbers) are available on request.
