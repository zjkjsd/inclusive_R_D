import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union, Optional

import numpy as np
import pandas as pd
from termcolor import colored
import uproot
import optuna

import sys
sys.path.insert(0, "/home/belle/zhangboy/B2SW/2025_VirginiaTech/sysvar/src/")

import utilities as util


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Chi2Target:
    variable: str
    bins: np.ndarray
    cut: str  # pandas query string


CATEGORIES = [
    "measured_hadronic",
    "semileptonic",
    "unmeasured:2-body",
    "unmeasured:3-body",
    "unmeasured:4-body",
    "unmeasured:5-body",
    "unmeasured:6-body",
    "unmeasured:7-body",
]

FIXED_WEIGHTS = {
    "measured_hadronic": 1,
    "semileptonic": 1,
    "unmeasured:2-body": 1,
    # e.g. "unmeasured:7-body": 0.0,
}

LUMINOSITY_SCALE = {"all_mc": 0.25}
TUNE_WEIGHT_COL = "BB_weight"
BASE_WEIGHT_COL = "__weight__"
PID_WEIGHT_COL = 'total_PID_weight'


# -----------------------------
# Helpers: selection + hist + chi2
# -----------------------------
def _safe_query(df: pd.DataFrame, cut: str) -> pd.DataFrame:
    if not cut or cut.strip() == "":
        return df
    return df.query(cut, engine="python")


def _hist_with_uncertainty(x: np.ndarray, bins: np.ndarray, w: Optional[np.ndarray] = None):
    """
    hist = sum(w)
    err  = sqrt(sum(w^2))
    """
    if w is None:
        hist, _ = np.histogram(x, bins=bins)
        err = np.sqrt(hist.astype(float))
        return hist.astype(float), err

    w = np.asarray(w, dtype=float)
    hist, _ = np.histogram(x, bins=bins, weights=w)
    w2, _ = np.histogram(x, bins=bins, weights=w * w)
    err = np.sqrt(w2)
    return hist.astype(float), err


def reduced_chi2_data_mc(
    df_data: pd.DataFrame,
    samples_mc: Dict[str, pd.DataFrame],
    target: Chi2Target,
    *,
    luminosity_scale: float = 1.0,
    base_weight_col: str = BASE_WEIGHT_COL,
    tune_weight_col: Optional[str] = TUNE_WEIGHT_COL,
    event_weight_col: Optional[str] = PID_WEIGHT_COL,
    drop_empty_bins: bool = True,
) -> float:
    # Data
    dsel = _safe_query(df_data, target.cut)
    dx = dsel[target.variable].to_numpy()
    d_hist, d_err = _hist_with_uncertainty(dx, target.bins, w=None)

    # MC total
    mc_hist_tot = np.zeros(len(target.bins) - 1, dtype=float)
    mc_err2_tot = np.zeros(len(target.bins) - 1, dtype=float)

    for _, df in samples_mc.items():
        if df is None or len(df) == 0:
            continue
        msel = _safe_query(df, target.cut)
        if len(msel) == 0:
            continue

        mx = msel[target.variable].to_numpy()

        # base weights
        if base_weight_col in msel.columns:
            w = msel[base_weight_col].to_numpy(dtype=float)
        else:
            w = np.ones(len(msel), dtype=float)

        # event-by-event correction
        if event_weight_col is not None and event_weight_col in msel.columns:
            w = w * msel[event_weight_col].to_numpy(dtype=float)

        # weights to be tuned
        if tune_weight_col is not None and tune_weight_col in msel.columns:
            w = w * msel[tune_weight_col].to_numpy(dtype=float)

        # luminosity scaling
        w = w * float(luminosity_scale)

        mh, me = _hist_with_uncertainty(mx, target.bins, w=w)
        mc_hist_tot += mh
        mc_err2_tot += me * me

    mc_err = np.sqrt(mc_err2_tot)

    denom2 = d_err * d_err + mc_err * mc_err
    if drop_empty_bins:
        mask = denom2 > 0
    else:
        mask = np.ones_like(denom2, dtype=bool)

    if not np.any(mask):
        return np.inf

    chi2 = np.sum(((d_hist[mask] - mc_hist_tot[mask]) ** 2) / denom2[mask])
    ndf = int(np.sum(mask))
    return float(chi2 / ndf) if ndf > 0 else np.inf


# -----------------------------
# Objective (single or multiple targets)
# -----------------------------
TargetsInput = Union[Chi2Target, Sequence[Chi2Target]]


def make_objective(
    df_data: pd.DataFrame,
    df_mc: pd.DataFrame,
    mode: str,
    targets: TargetsInput,
    *,
    cap_nbody: Optional[int] = None,
    warn_missing_weight_keys: bool = True,
    verbose_reweight: bool = False,
):
    # normalize targets to a list
    if isinstance(targets, Chi2Target):
        targets_list: List[Chi2Target] = [targets]
    else:
        targets_list = list(targets)

    # classify ONCE
    samples_base = util.classify_mc_dict(df_mc, mode, template=False)

    def objective(trial: optuna.Trial) -> float:
        # suggest weights
        weights: Dict[str, float] = {}
        for k in CATEGORIES:
            if k in FIXED_WEIGHTS:
                weights[k] = float(FIXED_WEIGHTS[k])
            else:
                weights[k] = trial.suggest_float(k, 0.01, 1.0, log=True)

        # apply weights -> writes EVENT_WEIGHT_COL into the dfs inside samples_base
        util.reweight_BBbar_background(
            samples_base,
            weight_map=weights,
            out_weight_col=TUNE_WEIGHT_COL,
            verbose=verbose_reweight,
            cap_nbody=cap_nbody,
            warn_missing_weight_keys=warn_missing_weight_keys,
        )

        # objective = SUM of reduced chi^2 across all targets
        per_target_rchi2 = {}
        sum_rchi2 = 0.0
        for t in targets_list:
            r = reduced_chi2_data_mc(
                df_data,
                samples_base,
                t,
                luminosity_scale=LUMINOSITY_SCALE["all_mc"],
                base_weight_col=BASE_WEIGHT_COL,
                tune_weight_col=TUNE_WEIGHT_COL,
                event_weight_col=PID_WEIGHT_COL,
                drop_empty_bins=True,
            )
            per_target_rchi2[t.variable] = float(r)
            sum_rchi2 += float(r)

        # log useful info
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("per_target_rchi2", per_target_rchi2)
        trial.set_user_attr("sum_rchi2", float(sum_rchi2))

        # optional pruning
        trial.report(sum_rchi2, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(sum_rchi2)

    return objective


# -----------------------------
# Run example
# -----------------------------
if __name__ == "__main__":
    # Load data and mc
    print(colored('Loading data and initializing configrations', 'blue'))
    columns = util.all_relevant_variables
    pre_cut = '(B0_roeMbc_my_mask>5) & (-4<B0_roeDeltae_my_mask) & (B0_roeDeltae_my_mask<1) & (B0_dr<0.1)'
    MC_4S_comb = uproot.concatenate(['Samples/MC16rd/e_channel/lgb_comb_BDT1_2.root:B0'],
                          library="np",
                          cut = pre_cut,
                          filter_branch=lambda branch: branch.name in columns)

    data_4S_comb = uproot.concatenate(['Samples/Data/e_channel/proc16_4S_lgb_comb_BDT1_2.root:B0'],
                          library="np",
                          cut = pre_cut,
                          filter_branch=lambda branch: branch.name in columns)

    df_mc_4S_comb = pd.DataFrame(MC_4S_comb)
    df_data_4S_comb = pd.DataFrame(data_4S_comb)

    # Load PID corrections
    from sysvar import add_weights_to_dataframe
    
    add_weights_to_dataframe(df = df_mc_4S_comb,systematic= "eID_eff",MC_production= "MC15rd",
                             prefix= "ell",weightname ="eID_eff_weight",)
    
    add_weights_to_dataframe(df = df_mc_4S_comb,systematic= "eID_fake",MC_production= "MC15rd",
                             prefix= "ell",weightname ="eID_fake_weight",)

    df_mc_4S_comb[PID_WEIGHT_COL] = df_mc_4S_comb[["ell_eID_eff_weight", "ell_eID_fake_weight",]].product(axis = 1)

    
    # Define ONE target (single-variable optimization)
    target_single1 = Chi2Target(
        variable="B0_recMissM2",
        bins=np.linspace(-4, 10, 31),
        cut="1.855<D_M<1.885 and B0_vtxReChi2>3",
    )

    target_single2 = Chi2Target(
        variable="p_D_l",
        bins=np.linspace(0.2, 4, 31),
        cut="1.855<D_M<1.885 and B0_vtxReChi2>3",
    )

    # Define MULTIPLE targets (multi-variable optimization)
    targets_multi = [target_single1, target_single2]

    # Choose which to optimize:
    targets_to_use = target_single2  # or target_single
    print(colored(f'Set up the target variable \n {targets_to_use}', 'green'))


    objective = make_objective(
        df_data=df_data_4S_comb,
        df_mc=df_mc_4S_comb,
        mode="e",
        targets=targets_to_use,
        cap_nbody=None,
        warn_missing_weight_keys=True,
        verbose_reweight=False,
    )

    print(colored(f'Optimization starts', 'red'))
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
    )

    study.optimize(objective, n_trials=200, n_jobs=1)

    print("\nBest sum(reduced chi2):", study.best_value)
    print("Best weights:")
    for k, v in study.best_params.items():
        print(f"  {k:>18s}: {v:.6f}")

    out = {
        "best_value_sum_reduced_chi2": float(study.best_value),
        "best_weights": study.best_params,
        "fixed_weights": FIXED_WEIGHTS,
    }
    with open("best_bbbar_weights_optuna.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: best_bbbar_weights_optuna.json")