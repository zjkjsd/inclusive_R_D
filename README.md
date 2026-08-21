<p align="left">
  <img src="https://www.belle2.org/common/logo.png" alt="Belle II logo" width="20%">
</p>

# inclusive_R_D

Belle II analysis code for measuring \(R(D^{\pm})\) with an inclusive-tagging
method. The signal-side candidate is reconstructed from a charged D meson and
a lepton, while the tag side is inferred from rest-of-event (ROE) masks rather
than an exclusive tag reconstruction.

The repository covers the analysis chain from grid reconstruction and ntuple
production to LightGBM training, BBbar-background reweighting, template
construction, and a two-dimensional binned likelihood fit with `pyhf`.

> [!IMPORTANT]
> Read [the analysis context](docs/ANALYSIS_CONTEXT.md) and the latest dated
> analysis status slides before changing selections, MC categories, fit bins,
> corrections, or other physics logic. Status slides are collaboration
> material and are not stored in this repository.

The analysis is normally split by data-taking period (`run1` or `run2`) and
lepton channel (`e` or `mu`). The current samples are Proc16 + Prompt16 data
and MC16rd simulation for Run 1 and Run 2. The current basf2 release is
`light-2505-deimos`; the next major production update is expected to use
`light-2607-kasei`.

## Repository layout

| Path | Purpose |
|---|---|
| `Recon_scripts/` | basf2 steering scripts for grid production, plus a local `bsub` test wrapper |
| `2_LightGBM_Tuner.py` | Optuna/LightGBMTunerCV tuner for the off-resonance binary data/MC check (see the limitation below) |
| `3_LightGBM_Binary_Training.py` | Off-resonance data/MC binary-classifier check |
| `4_LightGBM_Multiclass_Training.py` | Four-class signal-selection BDT training |
| `5_BBbkg_weights_optuna_minuit.py` | BBbar MC weight tuning with Optuna and iminuit |
| `utilities.py` | Shared variables, MC classification, corrections, templates, fitting, toys, and plotting helpers |
| `Notebooks/` | Training-set preparation, PID studies, truth checks, fitting examples, and validation plots |
| `Fit_toys/` | Toy-fit workflow and example workspaces |
| `docs/ANALYSIS_CONTEXT.md` | Physics categories, fit model, control regions, open issues, and editing guardrails |
| `Old_scripts/` | Superseded implementations retained for reference; do not use as the default workflow |
| `Samples/` | Local grid ntuples (ignored by Git; create/download locally) |

Numbered scripts and notebooks broadly follow pipeline order. When adding a
new stage, preserve that convention where practical and update this table and
the workflow below.

## Workflow

1. **Reconstruct on the grid.** Run the steering scripts in
   `Recon_scripts/` in a full basf2 environment. The main script produces
   electron, muon, and wrong-charge ntuples.
2. **Download ntuples.** Store grid output below `Samples/`, organized by MC
   campaign and channel as expected by the consuming script or notebook.
3. **Prepare training inputs.** See
   `Notebooks/2_create_BDTtrainingSet.ipynb` for a working example.
4. **Run the current binary-BDT tuner, if needed.**
   `2_LightGBM_Tuner.py` currently loads off-resonance data and MC and assigns
   binary data/MC labels. Run it with `--objective binary`; its default
   `multiclass` objective has no corresponding `num_class` configuration and
   is incompatible with those labels. Its Optuna study is separate from the
   four-class model in the next step, which uses hard-coded parameters and
   does not consume the study. The tuner must be updated before it can support
   both binary data/MC tuning and four-class signal-selection tuning; do not
   treat it as tuning the final classifier in its current form. Optuna SQLite
   studies and LightGBM artifacts are local, regenerable outputs.
5. **Train classifiers.** `3_LightGBM_Binary_Training.py` performs the
   off-resonance data/MC check. `4_LightGBM_Multiclass_Training.py` trains the
   four-class model whose outputs (`sig_prob`, `fakeD_prob`,
   `continuum_prob`, and `combinatorial_prob`) are used in later selections.
6. **Tune BBbar background weights.** Run
   `5_BBbkg_weights_optuna_minuit.py` for each run/channel combination. Use
   `python 5_BBbkg_weights_optuna_minuit.py --help` for the current command
   line options. The fit uses a fixed raw-MC support mask and a 2D Poisson
   deviance, followed by MIGRAD/HESSE/MINOS refinement.
7. **Build templates and fit.** `utilities.py` provides
   `create_templates_new`, `create_workspace`, `pyhf_utils`, and `toy_utils`.
   The `Notebooks/3_fitting_cabinetry_*.ipynb` notebooks demonstrate one- and
   two-dimensional yield extraction.
8. **Validate.** Use the notebooks in `Notebooks/Plotting/` for data/MC
   comparisons in unblinded control regions. These checks do not imply that
   the signal region is unblinded.

Additional notebook entry points include:

- `0_Print_MCDecayString.ipynb` — compare MC classification with decay trees.
- `1_create_PIDweights.ipynb` — study PID corrections supplied through the
  Belle II systematics-corrections framework.
- `2_create_BDTtrainingSet.ipynb` — build signal and background BDT samples.
- `3_fitting_cabinetry_1d_SR.ipynb` and
  `3_fitting_cabinetry_2d_SR.ipynb` — `pyhf`/`cabinetry` fit examples.

## Environment

The offline workflow uses Python packages including:

```text
cabinetry  iminuit  lightgbm  matplotlib  numpy  optuna
pandas     pyhf     termcolor uncertainties uproot
```

No pinned environment or lockfile is currently provided, so use versions
compatible with the active Belle II analysis environment and record them when
producing results. The scripts in `Recon_scripts/` additionally require a full
basf2 release and cannot be imported or meaningfully checked in plain Python.

## Before making changes

- Do not commit ntuples, Optuna databases, ROOT output, or large/regenerable
  fit artifacts. See `.gitignore`.
- Treat category names and PDG-code products in `utilities.py` as physics
  interfaces, not cosmetic strings or arbitrary integers.
- Preserve the run/channel split unless a combined treatment is intentional.
- Parameterize user-specific grid and calibration paths when making code
  reusable.
- Keep fit-support masks independent of fitted weights, and flag changes to
  bins, cuts, floated parameters, or systematic assumptions to the analyst.
- Prefer surgical edits to `utilities.py`; it contains active code alongside
  commented historical implementations.

See [Analysis Context](docs/ANALYSIS_CONTEXT.md#9-editing-and-analysis-conventions)
for the detailed conventions and physics-specific cautions.
