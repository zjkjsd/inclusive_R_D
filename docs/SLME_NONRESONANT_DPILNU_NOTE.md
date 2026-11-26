# Non-resonant B → D(\*)πℓν in the generic-MC recommendation: is there still "no room"?

*Note for the SL/ME working group (MC Samples WG1). Date: 26 November 2026.*

## Summary

The xwiki page "Correction of BF(B → Xc ℓν) in generic MC" (version of 2026-11-17) recommends masking the
non-resonant B → D(\*)πℓν decays and putting their rate into the gap modes B → D(\*)ηℓν. The argument is that
the measured B → Dπℓν and D\*πℓν rates are already saturated by D\*\* → D(\*)π, with leftovers of
(0.03 ± 0.09) % and (−0.11 ± 0.11) %.

That subtraction used the HFLAV 2018/2021 value of B(B → D₀\*ℓν), about 0.4 %. In HFLAV 2023, and in the
page's own 2023 table, B(B → D₀\*ℓν) dropped to 0.08 %. Redoing the same subtraction with the 2023 values,
**the Dπℓν channel is no longer saturated:**

- about **0.19 % per charged-pion mode** (B⁰ → D̄⁰π⁻ℓν and B⁻ → D⁺π⁻ℓν) is left over;
- that is **5–6σ** from zero with the page's own uncertainties;
- it is about **0.28–0.30 % summed over both charge modes**.

The D\*πℓν channel stays compatible with zero.

Moving this rate into D(\*)ηℓν keeps the inclusive B → Xcℓν rate right. It cannot reproduce the
measured exclusive B → Dπℓν rate: a generic MC built from the 2023 table under-predicts B → D̄⁰π⁻ℓν by
about 50 %. We ask the group to re-evaluate this point (questions at the end).

## 1. What the recommendation says

- **xwiki page,** section "Other gap modes – what about B → D(\*)πℓν":
  - "Overall, there is not much space for the B → D(\*)πℓν mode",
  - with B(B → D\*\*(→ Dπ)ℓν) − B(B → Dπℓν) = (0.03 ± 0.09) % and
    B(B → D\*\*(→ D\*π)ℓν) − B(B → D\*πℓν) = (−0.11 ± 0.11) %;
  - "it is recommended to mask these kinds of decays and instead fill the gap with alternative possible
    candidate modes like D(\*)η … assigned a 100 % uncertainty".
- **"Update of b → c ℓν MC Branching Fractions & Gap Model"** (M. Prim, indico event 12225), slide 4: the
  Goity–Roberts D(\*)πℓν lines are set to zero, "compatible with zero, epsilon BR absorbed into gap".
- **HFLAV 2023 table on the same page** (per light lepton, B± / B⁰, in %):

  | Mode | B± | B⁰ |
  |---|---|---|
  | D₁ | 0.64 ± 0.10 | 0.59 ± 0.10 |
  | D₀\* | **0.08 ± 0.04** | **0.08 ± 0.03** |
  | D₁′ | 0.28 ± 0.04 | 0.26 ± 0.04 |
  | D₂\* | 0.32 ± 0.03 | 0.30 ± 0.03 |
  | Dη = D\*η | 0.92 ± 0.15 | 0.88 ± 0.14 |
  | Xc | 11.05 ± 0.16 | 10.27 ± 0.15 |

  Non-resonant D(\*)πℓν: 0.
- **The slides** (HFLAV 2021 → 2023) list the D₀\* input change:
  B(B⁻ → D₀\*⁰(→ D⁺π⁻)ℓν) went from 0.280 ± 0.03 ± 0.04 % to 0.084 ± 0.021 %. That is a factor 3.3 on the
  product, and a factor of about 5 on the full D₀\* rate as quoted on the page (0.42 % → 0.08 %).

## 2. The subtraction redone with the 2023 inputs

**Inputs** (HFLAV 2023, arXiv:2411.18639):

- **Measured totals** (Table 33):
  - B̄⁰ → D⁰π⁺ℓ⁻ν̄ = 0.363 ± 0.021 %;
  - B⁻ → D⁺π⁻ℓ⁻ν̄ = 0.380 ± 0.020 %;
  - B̄⁰ → D\*⁰π⁺ℓ⁻ν̄ = 0.542 ± 0.028 %;
  - B⁻ → D\*⁺π⁻ℓ⁻ν̄ = 0.536 ± 0.028 %.
- **D\*\* contributions:**
  - D₀\* contributes 2/3 of its rate to the charged-pion mode.
  - For D₂\*, Dπ/D\*π = 1.54 ± 0.15 (as on the xwiki page), so the charged-pion Dπ fraction is
    (2/3) × 1.54/2.54 = 0.404.
  - D₁ and D₁′ do not decay to Dπ.

**B → Dπℓν, charged-pion mode** (in %):

| | B⁻ → D⁺π⁻ℓν | B̄⁰ → D⁰π⁺ℓν |
|---|---|---|
| measured (HFLAV 2023 Table 33) | 0.380 ± 0.020 | 0.363 ± 0.021 |
| D₀\* → Dπ: 0.08 × 2/3 | 0.053 ± 0.027 | 0.053 ± 0.020 |
| D₂\* → Dπ: 0.32 (0.30) × 0.404 | 0.129 ± 0.012 | 0.121 ± 0.012 |
| **leftover (non-D\*\* Dπ)** | **0.198 ± 0.036** | **0.188 ± 0.031** |
| same subtraction with the old D₀\* (0.42 / 0.39 %) | ≈ −0.02 | ≈ −0.02 |

- The last row reproduces the page's "(0.03 ± 0.09) %, compatible with zero". The conclusion therefore
  rests on the old D₀\* value.
- With isospin (the π⁰ mode is half the charged mode), the leftover is about **0.28 % (B⁰) and 0.30 %
  (B⁺)** summed over both charge modes. That is about 15 % of the present gap.

**B → D\*πℓν** (B⁻ → D\*⁺π⁻ℓν): the D₁ + D₁′ + D₂\* products 0.270 + 0.185 + 0.085 = 0.540 %, against the
measured 0.536 ± 0.028 %. Saturated; masking is fine here.

## 3. Why the gap modes cannot take this rate

- The masked rate moves into B → D(\*)ηℓν, produced through D₀\* → Dη and D₁′ → D\*η in the dedicated
  samples (event types 1196708000/01, 1296708000/01).
- Those final states contain the η decay products. They do not appear in an exclusive B → Dπℓν
  measurement, where the B is fully reconstructed with no extra particles.
- So the inclusive rate is preserved, but the exclusive B → Dπℓν rate of the generic MC becomes
  **about 0.17–0.18 % instead of the measured 0.36–0.38 % (−6σ to −10σ with the measurement error alone).**
- For comparison, the current release-08 generic MC with its Goity–Roberts lines is −3.5σ low.

## 4. Possible interpretation

- **Both numbers come from the same analysis.** Belle 2023 (Meier et al., PRD 107, 092003, arXiv:2211.09833)
  dominates the new D₀\* value: B⁻ product 0.054 % (×3/2 → 0.08 %). The same paper measures the total
  B → Dπℓν rate.
- So in Belle's data most of the D⁺π⁻ (D⁰π⁺) yield outside D₂\* is not described by their D₀\* model. It
  could be broad or non-resonant Dπ, or a D₀\* with a different lineshape.
- HFLAV 2023 notes the Belle–BaBar incompatibility for D₀\* (χ² = 11 for 1 dof) and that the D₀\* model of
  the existing measurements has been questioned.
- **This Dπ component is real in the data whatever its origin.** Labelling it "gap" and generating it as
  D(\*)η changes the final state.

## 5. Questions and suggestions for the group

1. Could the "room for non-resonant D(\*)πℓν" on the xwiki page be re-evaluated with the HFLAV 2023 inputs
   (section 2)?
2. If the leftover is confirmed, possible ways to put it into the generic MC:
   a. Keep a Dπℓν component at the leftover (about 0.19 % per charged mode) and reduce D(\*)ηℓν
      accordingly, so the inclusive stays closed: Dη = D\*η ≈ 0.74 % (B⁰) and 0.77 % (B⁺) instead of
     0.88 / 0.92 %.
   b. Generate it as a broad Dπ state, for example a `D_0*-gap → Dπ` component, in line with the planned
     intermediate-resonance gap model of the slides.
   c. Keep the recommendation and advise analyses to assign a systematic that moves the rate between
     D(\*)η and Dπ (0 ↔ leftover).
3. Uncertainties on the page:
   - the 2023 table prints D(\*)η with ±16 %, while the text assigns 100 % to the gap modes;
   - the D₀\* uncertainty (±0.03 / ±0.04 %) does not reflect the χ² = 11 / 1 of the HFLAV average.
   Which values should analyses use?
4. Related request: do unskimmed MC16rd productions of the dedicated gap samples exist (event types
   1196700001–04, 1196708000/01, 1296700001–04, 1296708000/01)? The FEI-skimmed uDSTs listed in
   BELLE2-NOTE-PH-2026-035 cannot be used by analyses with a different selection.

## Sources

- Belle II xwiki, Physics SLMissing / MC Samples WG1, "Correction of BF(B to Xc l ν) in generic MC",
  version of 2026-11-17.
- M. Prim, "Update of b → c ℓν MC Branching Fractions & Gap Model", indico event 12225.
- HFLAV, "Averages of b-hadron, c-hadron, and τ-lepton properties as of 2023", arXiv:2411.18639,
  PRD 113, 012008 (2026): Tables 33–37, eq. 106.
- Belle, F. Meier et al., PRD 107, 092003 (2023), arXiv:2211.09833.
- PDG 2026: S042.311 (B⁰ → D̄⁰π⁻ℓ⁺ν, 0.364 ± 0.021 %), S041.418 (B⁺ → D⁻π⁺ℓ⁺ν, 0.382 ± 0.020 %),
  S042.418 (B⁰ → D₀\*⁻ℓ⁺ν × B(D₀\*⁻ → D̄⁰π⁻) < 0.044 %, 90 % CL).
- Dedicated decay files: basf2 `decfiles/dec/1196708000.dec`, `1196708001.dec`, `11967000{01-04}.dec`,
  `12967xxxxx.dec`.
