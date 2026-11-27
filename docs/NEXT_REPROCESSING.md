# Variables to add at the next ntuple reprocessing

**Read this before submitting the next reconstruction** (the move to `light-2607-kasei`, or any
earlier rerun of `Recon_scripts/2_Reconstruction.py`). Reprocessing is the expensive step, so all
new variables go in together. Add to this list whenever an analysis step finds a missing variable.

The same list is flagged in `Recon_scripts/2_Reconstruction.py`, next to the `Ancestor_info`
block, so it is seen when the script is edited.

Last updated: 2026-11-27.

## To add, unconditionally

| Variable | For which particles | Suggested basf2 expression (check the name in the release used) | Why |
|---|---|---|---|
| signed PDG code of the ancestor B | D and lepton, RS and WC lists | `varForFirstMCAncestorOfType(511, PDG)` and the same with `521` | The B flavor is not stored. For a neutral B it can't be recovered from the daughters for 0.3–1.9 % of the candidates, and 34 final states have different BFs in the B⁰ and B̄⁰ blocks of the decay file. It also tells directly whether the stored B is the one after mixing. |
| number of daughters of the ancestor B | D and lepton, RS and WC lists | `varForFirstMCAncestorOfType(511, nMCDaughters)` and the same with `521` | Only 7 daughters are stored. Lists with 7 non-photon daughters are 3.5–6.5 % of the combinatorial candidates; their true multiplicity is unknown, and a hidden 8th daughter could be a second charm hadron (it decides between the single- and double-charm classes). |
| more daughter slots: 10 instead of 7 | D and lepton, RS and WC lists | `range(7)` → `range(10)` in the `Ancestor_info` loop | Same reason; with the number of daughters stored, 10 slots cover nearly all decays. |

## To add only if a study asks for it

| Variable | Condition | Why |
|---|---|---|
| K_L ancestry (mother and grandmother PDG) | C1 shows that K_L matters | attribute each K_L to its source and correct the right decay step |
| K_L momentum and polar angle | C1 shows that K_L matters | K_L spectrum and detector response |
| calorimeter cluster from the K_L entering the ROE (flag, energy) | C1 shows that K_L matters; if feasible | K_L detector response |
| daughter list of the tag-side (ROE) B ancestor | the tag-side check (piece 7) matches the H1 fingerprint | apply B-decay parameters to the tag side of all components |

## To check while there

- `nMC_K_L` is used by the analysis (`utilities.analysis_variables`), but in
  `2_Reconstruction.py` it appears only in commented-out lines. Confirm where the branch of the
  current ntuples comes from, and keep it.
- The daughter codes are written with `useFloat=True` (float32). This is exact below 2²⁴ ≈ 1.7·10⁷,
  which covers every code seen so far (largest: 9010443). Codes above that (nuclei) would be rounded.

Background: `docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md`, sections B.12 and "Notes: variables that may
be needed later".

## Added 2026-11-26 (piece 4, point 4.7)

- True four-vectors (or at least p\*, M_X, q²) of the daughters of the ancestor B of the D candidate and of
  the lepton candidate (comb included), and the daughter list of a D\*\* among them: needed to reweight the
  phase-space gap modes of the generic MC to the D\*\*-mediated model (and for an MCAmbulance-type lineshape
  correction of the broad D\*\*). `B0_mcMomTransfer2` is only the candidate's own B and is 0 for comb.

## Added 2026-11-27 (piece 4, point 4.8): MCAmbulance and HAMMER inputs

The working group recommends MCAmbulance (github.com/Herren/MCAmbulance) for B → D\*\*ℓν and HAMMER for the
signal and own-B semileptonic templates. Their per-event inputs are not in the current ntuples.

| Variable | For which particles | Suggested basf2 expression (check the name in the release used) | Why |
|---|---|---|---|
| true mass of the B's first MC daughter | B candidate (own B) | `vm.addAlias('mcDaughter_0_M', 'mcDaughter(0, M)')`; keep the present `daughterMCInvariantMass(0)` under a new name, e.g. `mcRecoD_M` | MCAmbulance input (the D\*\* mass, generator value incl. its rel08 lineshape); the present `mcDaughter_0_M` is the mass of the matched reconstructed D, not of the D\*\* (checked in the release-08 source, 2026-11-27) |
| for both B mesons of the event (i = 0, 1): signed PDG, PDG and mass of daughter 0, PDG of daughters 1 and 2, number of daughters | event level (covers the own B, the other B and the tag side) | `genUpsilon4S(mcDaughter(i, PDG))`, `genUpsilon4S(mcDaughter(i, mcDaughter(0, PDG)))`, `genUpsilon4S(mcDaughter(i, mcDaughter(0, M)))`, `genUpsilon4S(mcDaughter(i, mcDaughter(1, PDG)))`, `genUpsilon4S(mcDaughter(i, mcDaughter(2, PDG)))`, `genUpsilon4S(mcDaughter(i, nMCDaughters))` | MCAmbulance for comb D\*\*ℓν: per B the D\*\* mass, the B charge, the D\*\* type (D₀\* or D₁′) and the lepton flavor (the τ itself for τ modes); the event weight is the product over the two B. Do not rely on the daughter order: find the D\*\* and the lepton by PDG. Only semileptonic D\*\* decays are reweighted (a lepton sibling is required); hadronic B → D\*\*X is not |
| true four-vectors (E, p_x, p_y, p_z) of the B, its daughters, the X_c daughters (D\* → Dπ/γ, D\*\* → D(\*)π) and the τ daughters | own B (needed for the SR HAMMER step); the other B only for a comb HAMMER cross-check | `mcDaughter(i, E)` etc. on the B candidate (or uncomment `cms_mc_kinematics` in `2_Reconstruction.py:52`), nested `mcDaughter(i, mcDaughter(j, ...))`; for the other B `genUpsilon4S(mcDaughter(i, ...))` | HAMMER needs the full decay chain for D\*, D\*\* and τ modes (helicity angles, τ kinematics); q² alone (`B0_mcMomTransfer2`) suffices only for D ℓν. A lighter alternative: store the ancestor-B MC index per candidate and write the B decay trees with `scripts/mc_truth/b_decay_truth_basf2.py`, joined by (exp, run, event, index) |

Notes: MCAmbulance's default D₁′ width is 314 MeV (basf2 release-09); our MC16rd was generated with the
release-08 384 MeV. For our two-body D\*\* lines this only scales the D₁′ℓν weights by 0.960 (e, μ) / 0.981 (τ);
renormalize them or set `m_nom`/`g_nom` to the rel08 values (`docs/B0_PIECE4_REVIEW.md`, 3.8). Narrow D₁, D₂\*
are not covered by MCAmbulance. The upper item of 2026-11-26 ("at least p\*, M_X, q²") is not enough for
HAMMER on D\*, D\*\* and τ; the table above replaces it for that purpose.
