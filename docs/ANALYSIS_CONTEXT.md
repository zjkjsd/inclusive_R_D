# Analysis Context: Inclusive R(D±) Measurement

This file summarizes the analysis strategy, event categorization, fit model,
and validation plan for the inclusive-tagging R(D±) measurement at Belle II
(Proc16 data, MC16rd, `light-2505-deimos`, Run1 + Run2, ~490 fb⁻¹ total).
It is meant as a map for anyone (including future me) picking the repo back
up after a break.

## 1. Analysis Flow

1. **Event reconstruction** (`Recon_scripts/`)
   - `B_sig`: 4 final-state particles, no slow pion — `D± -> K∓ π± π±`
     recombined with a lepton (`e` or `μ`), `B_sig -> D ell`.
   - `B_tag`: ROE (rest-of-event) masks only; no full hadronic/semileptonic
     tag-side reconstruction.
2. **Preselection** (applied during reconstruction)
   - Track-quality and PID-NN requirements on final-state particles
     (`pionIDNN`, `kaonIDNN`, `electronIDNN`, `muonIDNN`).
   - Vertex-quality cuts on the D and B vertices (`vtxReChi2`).
   - Loose selections on D, B, and ROE kinematics (`roeMbc`, `roeDeltae`,
     `TagVReChi2IP`, etc.), left loose enough to retain BDT training
     statistics.
3. **Final candidate / event selection**
   - Multiclass LightGBM BDT (4 classes: signal, fakeD, continuum,
     combinatorial) trained on continuum-suppression + D/B kinematic
     variables (`2_LightGBM_Tuner.py`, `4_LightGBM_Multiclass_Training.py`).
   - Final event selection cuts on the BDT output probabilities
     (`sig_prob`, `fakeD_prob`, `continuum_prob`, `combinatorial_prob`).
4. **MC validation in data control regions**
   - Apply Belle II–recommended corrections first (see §5).
   - Where data/MC discrepancies remain large, derive analysis-specific
     corrections (BBbar background reweighting is the main one so far).
5. **Systematics evaluation**
   - MC statistics, continuum background normalization, D**ℓν and gap-mode
     modeling, D*τν and D**τν feed-down, BBbar background modeling.
6. **Yield extraction**
   - 2D binned likelihood fit with `pyhf`, using `cabinetry` for workspace
     handling (see §4).

## 2. Event Category Definitions

Categories are built by `utilities.classify_mc_dict()` and labeled via
`DecayMode_new`. Definitions:

- **Signal** — `B_sig` is a correctly reconstructed `B -> D τν` decay, and
  `B_tag` is a real (but generic) B decay.
- **Normalization** — same as signal but `B -> D ℓν` (ℓ = e, μ).
- **Signal-like physics backgrounds** — `B_sig` is reconstructed from a
  *true* D and a *true* lepton, but the parent decay has additional
  daughters that were not reconstructed on the signal side. The missing
  daughter(s) end up in the ROE, so `B_tag` is not a genuine independent B
  decay but "real B decay + leftover from B_sig". Example: `B -> D* τν`,
  `D* -> D π`, where the `π` is missed and folds into the ROE. This bucket
  covers:
    - `B -> D*(**) τν`, `B -> D*(**) ℓν`
    - Semileptonic "gap" modes (`D ℓν` gap-pi / gap-eta in code, merged into
      `D ℓν_gap`)
  These are the `hadronicB_secondaryL`-adjacent "signals" category in code
  (`B2D_tau`, `B2D_ell`, `B2Dst_tau`, `B2Dst_ell`, `B2Dstst_*`, etc.)
- **Other backgrounds** (all require `B0_isContinuumEvent==0` unless noted):
    - `bkg_fakeD` — the reconstructed `Kππ` combination does not come from
      a real D± (`0 < D_mcErrors < 512`).
    - `bkg_fakeL` / `bkg_fakeTracks` — the lepton is not a true lepton, or
      one of the D daughters' tracks is a fake/clone track
      (`D_mcErrors == 512`).
    - `bkg_continuum` — true D, true ℓ, but the event is `e+e- -> qq̄`
      (`B0_isContinuumEvent==1`), not `Υ(4S) -> BB̄`.
    - `bkg_combinatorial` — true D, true ℓ, but they come from *different*
      B mesons in the same `BB̄` event.
    - `bkg_hadronicB_secondaryL` — true D, true ℓ, single B decay, but it is
      actually a hadronic B decay such as `B -> D D(s) X` where one of the
      charm mesons decays semileptonically (`D -> K ℓν`) and that
      *secondary* lepton is the one used in the B_sig reconstruction.

These last two ("BBbar background") are the categories re-weighted by
`5_BBbkg_weights_optuna_minuit.py`, further split by
`utilities.reweight_BBbar_background()` into:

- `BBbar_semileptonic`
- `BBbar_measured_hadronic` (dmID appears in `measured_pdg_list`, with a
  PDG-branching-fraction correction applied)
- `BBbar_unmeasured:2-body` … `BBbar_unmeasured:5+-body` (n-body counted
  from `combinatorial_vars_D` / `_ell`, capped at 5+)

An optional many-to-one **mode replacement** (`hadronicB_replacement_map`)
merges some non-resonant 3-/4-body unmeasured modes into resonant 2-body
states (e.g. `Ds D ππ -> Ds D1`) before the weight tuning, preserving total
yield via an `(N_old + N_new)/N_new` rescaling of the surviving mode.

## 3. Fit Model

- **Variables:** `B0_recMissM2` (missing mass squared, `M²_miss`) and
  `p_D_l` (`|p*_D| + |p*_ℓ|`, scalar sum of D and lepton CM-frame momenta).
- **Templates:** 2D histograms built per MC category (see §2).
- **Yield extraction:** binned likelihood fit with `pyhf`
  (`utilities.create_workspace`, `create_templates_new`), fit executed via
  `cabinetry`/`iminuit` (`utilities.pyhf_utils`, `toy_utils`).
- **BBbar background weight tuning** (a separate, upstream step feeding the
  main fit): joint 2D Poisson-deviance minimization
  (`5_BBbkg_weights_optuna_minuit.py`), using Optuna (TPE sampler +
  `MedianPruner`) for global exploration, then `iminuit`
  MIGRAD/HESSE/MINOS for the local refinement and uncertainties. The
  fit-bin mask is built once from raw unweighted MC support
  (`build_fixed_fit_mask`) and held fixed throughout.

## 4. Correction Order

1. Start from Belle II collaboration-recommended corrections
   (`b2help-recommendation`): track momentum scale & energy loss, PID
   efficiency/fake-rate tables (`utilities.apply_pid_corrections`).
2. Layer on analysis-specific corrections where data/MC discrepancies
   persist after step 1:
   - BBbar background reweighting (in active development; several tuning
     trials so far — see slides for weight ranges and χ²/ndf history).
   - FakeD and continuum normalization corrections are being investigated
     next (candidate `normsys` priors already exist for both: fakeD
     ±3.5% from the D-mass sideband, continuum ±15% from the
     `4S_offres` sample).

## 5. Validation Plan / Control Regions

| Region | Purpose |
|---|---|
| `q² < 3` (inverse D* veto excluded) | `B -> D ℓν` dominant — checks core fit variables |
| D* veto region (`0.135 < ΔM_D* < 0.145`) | `B -> D* ℓν` dominant |
| Υ(4S) off-resonance sample | Continuum background modeling |
| BDT combinatorial-enhanced sideband (`combinatorial_prob>0.6, sig_prob<0.1, fakeD_prob<0.1, continuum_prob<0.4`) | BBbar background modeling |
| Wrong-charge (WC) lepton/D reconstruction sample | Independent cross-check of BBbar background weights derived from the BDT sideband |
| D-mass (`m(Kππ)`) sidebands | FakeD modeling / normalization |

Known open issues surfaced during validation (see slides for details):
Ecms/experiment-level data-MC discrepancies in several run1/run2
experiments (hypothesized to trace to background-overlay run mismatches),
a fixed (non-run-dependent) BBbar cross section in MC16rd despite Ecms
drift, and slow-π0 efficiency corrections for the D* veto region still
pending an updated MC16rd table (expected summer 2026).

## 6. Repository Map

| Path | Role |
|---|---|
| `Recon_scripts/1_Reconstruction_test.py`, `2_Reconstruction.py` | basf2 steering scripts, signal-side + ROE/tag-side reconstruction, e and μ channels, includes wrong-charge reconstruction for `4S` energy |
| `Recon_scripts/submit_local_jobs.sh` | Local batch submission helper for the steering scripts |
| `2_LightGBM_Tuner.py` | Optuna/LightGBMTunerCV hyperparameter tuning for the continuum-suppression BDT |
| `3_LightGBM_Binary_Training.py` | Binary data/MC BDT (off-resonance validation) |
| `4_LightGBM_Multiclass_Training.py` | Final 4-class signal-selection BDT training |
| `5_BBbkg_weights_optuna_minuit.py` | BBbar background weight tuning (Optuna + iminuit joint 2D Poisson fit) |
| `utilities.py` | Shared library: variable lists, MC classification (`classify_mc_dict`), BBbar reweighting, template/workspace construction for `pyhf`, plotting helpers |

## 7. Decisions Pending

- For internal review / referees: is a 10% unblinding ready?
- Is a full unblinding of the signal region possible on the current
  timeline?

## 8. Open Questions

- Is the BBbar background, after reweighting, adequate for a `pyhf` fit to
  data (i.e. do the tuned weights generalize beyond the BDT sideband they
  were fit in, given the WC-sample cross-check currently underperforms)?
- Are the remaining systematics (MC statistics, D**ℓν/gap-mode modeling,
  D*τν/D**τν feed-down, fakeD/continuum normalization) small enough for
  the final fit, or do they need further reduction first?
