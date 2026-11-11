# `scripts/`

Offline analysis scripts (after grid reconstruction) and the Python modules
they share. Notebooks in `Notebooks/` and `Fit_toys/` import the same modules.

```text
scripts/
├── 2_LightGBM_Tuner.py                 # Optuna tuner, off-resonance binary data/MC check
├── 3_LightGBM_Binary_Training.py       # off-resonance binary data/MC BDT
├── 4_LightGBM_Multiclass_Training.py   # four-class signal-selection BDT
├── 5_BBbkg_weights_optuna_minuit.py    # BBbar MC weight tuning (Optuna + iminuit)
├── tools/                              # shared modules, imported by scripts and notebooks
│   ├── utilities.py
│   ├── fitting.py
│   ├── plotting.py
│   └── bdt_cut_optimization.py
└── tests/
    └── test_bdt_cut_optimization.py
```

## Running the scripts

Run every script from the **repository root**:

```bash
python scripts/4_LightGBM_Multiclass_Training.py -o multiclass
```

Input and output paths inside the scripts (`Samples/...`, `BDTs/LightGBM/...`,
`BBbkg_weights/...`) are relative to the current working directory, so
running from inside `scripts/` will not find the samples. Each script adds
`scripts/tools/` to `sys.path` itself, so `import utilities` works from any
working directory.

| Script | Purpose | Main inputs and outputs |
|---|---|---|
| `2_LightGBM_Tuner.py` | Optuna / `LightGBMTunerCV` tuning for the off-resonance binary data/MC check. Run with `-o binary`; the default `multiclass` objective does not match its binary labels. It does not tune the final four-class model. | reads `Samples/MC16rd/...4Soffres...`, `Samples/Data/...4Soffres...`; writes the study to `sqlite:///BDTs/LightGBM/lgbm_<objective>_study.db` |
| `3_LightGBM_Binary_Training.py` | Off-resonance data/MC binary classifier, as a modeling check | writes `BDTs/LightGBM/lgbm_<objective>.txt` and metric plots |
| `4_LightGBM_Multiclass_Training.py` | Four-class BDT (signal, fakeD, continuum, combinatorial) used in the selection | reads `Samples/BDT_training_sample.root`; writes `BDTs/LightGBM/lgbm_<objective>_v3.txt`, `eval_result_v3.json`, metric plots |
| `5_BBbkg_weights_optuna_minuit.py` | BBbar background weights from a joint 2D binned Poisson fit, Optuna then MIGRAD/HESSE/MINOS. Run once per run/channel. | reads `/home/belle/zhangboy/inclusive_R_D/Samples/4S_<run>_deimos_BDT_<channel>_3.root`; writes JSON results and an Optuna study under `BBbkg_weights/` |

Common options: scripts 2–4 take `-o/--objective` and `-c/--continue_train`.
Script 5 requires `--run {run1,run2,run1+run2}` and `--channel {e,mu}`; see
`python scripts/5_BBbkg_weights_optuna_minuit.py --help` for the rest.

## Shared modules (`tools/`)

The former single `utilities.py` is split into three modules by topic.
`bdt_cut_optimization.py` sits next to them because it is a library used by
`Notebooks/7_bdt_cut_optimization.ipynb`, not a script.

| Module | Contents |
|---|---|
| `utilities.py` | Variable lists and cut strings (`training_variables`, `all_relevant_variables`, `offline_cut`, `lgb_loose`, `lgb_tight`, ...), fit binning constants, truth categories (`get_truth_categories`), corrections (`apply_pid_corrections`, `apply_pi0_eff_correction`), PID performance evaluation, MVA application and best-candidate selection (`apply_mva_bcs`), MC classification (`classify_mc_dict`), BBbar reweighting (`prepare_bbbar_reweighting`, `apply_bbbar_weights`, `reweight_BBbar_background`), the BDT/fit-variable decorrelation study, and statistics helpers shared by the other two modules (`get_weights`, `binom_error`, `poisson_error`, `round_uarray`, `excess_variance`, `mismodeling_label`) |
| `fitting.py` | Template construction (`create_templates_new`, `create_2d_template_from_1d`, rebinning), `pyhf` workspaces (`create_workspace`, `extract_temp_asimov_channels`), `compare_2d_hist`, the D-mass fit (`fit_Dmass`), `fit_pull_linearity`, `pyhf_utils`, `toy_utils` |
| `plotting.py` | Reordered tab20 colormap (`my_cmap`), `plot_PID_weight_heatmap_mpl`, the `mpl` data/MC plotting class, fit-projection plots (`fit_project_cabinetry`, `mpl_projection_residual_iMinuit`) |
| `bdt_cut_optimization.py` | Candidate loading, fast figure-of-merit evaluation, control regions, cut search and working points for the BDT selection |

### Import direction

Imports go one way only:

```text
utilities  ←  fitting
utilities  ←  plotting
utilities  ←  bdt_cut_optimization
```

`fitting.py` and `plotting.py` import the names they need from
`utilities.py`. `utilities.py` imports neither of them, and `fitting.py` does
not import `plotting.py`. Keep it that way: a helper needed by both
`fitting.py` and `plotting.py` belongs in `utilities.py`. A circular import
would make which names exist depend on import order.

A side effect is that `import utilities` no longer pulls in `pyhf`,
`cabinetry` or `iminuit`. Those are only needed for `import fitting`.

### Using the modules in a notebook

```python
import sys
sys.path.append('/home/belle/zhangboy/inclusive_R_D/scripts/tools/')
import utilities as util
import fitting
import plotting

samples = util.classify_mc_dict(df, 'e')
templates = fitting.create_templates_new(samples, bins_sr, bins_sb, ...)
mpl = plotting.mpl(samples, data)
```

Change the path to your own checkout. Calls that used to be
`util.create_workspace`, `util.mpl`, `util.my_cmap`, etc. are now
`fitting.create_workspace`, `plotting.mpl`, `plotting.my_cmap`. A call left on
the old module fails with `AttributeError: module 'utilities' has no
attribute ...`, which tells you which name to move.

## Tests

```bash
python3 scripts/tests/test_bdt_cut_optimization.py
```

Builds synthetic multi-candidate events (no ntuples needed) and checks
`bdt_cut_optimization` against a direct pandas implementation.
