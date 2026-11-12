# Variables to add at the next ntuple reprocessing

**Read this before submitting the next reconstruction** (the move to `light-2607-kasei`, or any
earlier rerun of `Recon_scripts/2_Reconstruction.py`). Reprocessing is the expensive step, so all
new variables go in together. Add to this list whenever an analysis step finds a missing variable.

The same list is flagged in `Recon_scripts/2_Reconstruction.py`, next to the `Ancestor_info`
block, so it is seen when the script is edited.

Last updated: 2026-11-12.

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
