import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union, Optional

import numpy as np
import optuna
import pandas as pd
from termcolor import colored
import uproot

import utilities as util


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class Chi2Target:
    variable: str
    bins: np.ndarray
    cut: str  # pandas query string


CATEGORIES = [
    'bkg_fakeD',
    'BBbar_measured_hadronic',
    'BBbar_semileptonic',
    'BBbar_unmeasured:2-body',
    'BBbar_unmeasured:3-body',
    'BBbar_unmeasured:4-body',
    'BBbar_unmeasured:5+-body',
]

FIXED_WEIGHTS = {
    "BBbar_semileptonic": 1.0,
    'bkg_fakeD':0.87,
}

LUMINOSITY_SCALE = {
    "all_mc": 0.25,
}

TUNE_WEIGHT_COL = "BB_weight"
BASE_WEIGHT_COL = "__weight__"
PID_WEIGHT_COL = "total_PID_weight"


# ============================================================
# Command-line parser
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune BBbar MC weights using a joint 2D "
            "binned Poisson likelihood."
        )
    )

    parser.add_argument(
        "--run",
        required=True,
        choices=["run1", "run2"],
        help="Data-taking period.",
    )

    parser.add_argument(
        "--channel",
        required=True,
        choices=["e", "mu"],
        help="Lepton channel.",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Number of Optuna trials. Default: 200.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for Optuna. Default: 123.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON filename. By default, the filename "
            "is constructed from run and channel."
        ),
    )
    parser.add_argument(
        "--mode_replacement",
        action='store_true',
        help=(
            "Replace some non-resonant 3/4 body modes by 2 body modes "
            "before tuning weights."
        ),
    )

    return parser.parse_args()


# ============================================================
# Selection and histogram helpers
# ============================================================

def _safe_query(df: pd.DataFrame, cut: str,) -> pd.DataFrame:
    if not cut or not cut.strip():
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
    replacement_map = None,
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
                if k in ['BBbar_measured_hadronic','BBbar_unmeasured:2-body',]:
                    weights[k] = trial.suggest_float(k, 0.001, 5.0, log=True)
                elif k in ['BBbar_unmeasured:3-body','BBbar_unmeasured:4-body']:
                    weights[k] = trial.suggest_float(k, 0.001, 1.0, log=True)
                elif k=='BBbar_unmeasured:5+-body':
                    weights[k] = trial.suggest_float(k, 0.001, 0.1, log=True)

        # apply weights -> writes EVENT_WEIGHT_COL into the dfs inside samples_base
        samples_weighted = util.reweight_BBbar_background(samples_base,
                                                      weight_map = weights,
                                                      out_weight_col = TUNE_WEIGHT_COL,
                                                      weight_ell_side = True, 
                                                      verbose= False, 
                                                      cap_nbody = cap_nbody, 
                                                      warn_missing_weight_keys = warn_missing_weight_keys,
                                                      D_replacement_map = replacement_map,
                                                      ell_replacement_map = replacement_map,)

        # objective = SUM of reduced chi^2 across all targets
        per_target_rchi2 = {}
        sum_rchi2 = 0.0
        for t in targets_list:
            r = reduced_chi2_data_mc(
                df_data,
                samples_weighted,
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


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_arguments()

    run = args.run
    channel = args.channel
    mode_replacement = args.mode_replacement
    replace_map = util.hadronicB_replacement_map

    print(
        colored(
            f"Loading {run}, channel={channel}",
            "blue",
        )
    )

    columns = util.all_relevant_variables

    pre_cut = (
        "(B0_roeMbc_my_mask > 5)"
        " & (-4 < B0_roeDeltae_my_mask)"
        " & (B0_roeDeltae_my_mask < 1)"
        " & (B0_dr < 0.1)"
    )

    input_file = (
        "/home/belle/zhangboy/inclusive_R_D/"
        f"Samples/4S_{run}_deimos_BDT_{channel}_3.root"
    )

    mc_tree = f"MC_{channel}_comb"
    data_tree = f"Data_{channel}_comb"

    print(f"Input file: {input_file}")
    print(f"MC tree:     {mc_tree}")
    print(f"Data tree:   {data_tree}")

    MC_4S_comb = uproot.concatenate(
        [f"{input_file}:{mc_tree}"],
        library="pd",
        cut=pre_cut,
        filter_branch=lambda branch: branch.name in columns,
    )

    data_4S_comb = uproot.concatenate(
        [f"{input_file}:{data_tree}"],
        library="pd",
        cut=pre_cut,
        filter_branch=lambda branch: branch.name in columns,
    )

    # --------------------------------------------------------
    # PID corrections
    # --------------------------------------------------------
    MC_4S_comb = util.apply_pid_corrections(
        df=MC_4S_comb,
        run=run,
        channel=channel,
        corr_col_name=PID_WEIGHT_COL,
    )

    # --------------------------------------------------------
    # Chi^2 target
    # --------------------------------------------------------
    # Define ONE target (single-variable optimization)
    target_single1 = Chi2Target(
        variable="B0_recMissM2",
        bins=np.linspace(-4, 10, 31),
        cut="1.855<D_M<1.885 and fakeD_prob<0.1 and sig_prob<0.1",
    )

    target_single2 = Chi2Target(
        variable="p_D_l",
        bins=np.linspace(0.2, 4, 31),
        cut="1.855<D_M<1.885 and fakeD_prob<0.1 and sig_prob<0.1",
    )

    # Define MULTIPLE targets (multi-variable optimization)
    targets_multi = [target_single1, target_single2]

    # Choose which to optimize:
    targets_to_use = targets_multi  # or target_single
    print(
        colored(
            f"Chi^2 target:\n{targets_to_use}",
            "green",
        )
    )

    objective = make_objective(
        df_data=data_4S_comb,
        df_mc=MC_4S_comb,
        mode=channel,
        targets=targets_to_use,
        cap_nbody=5,
        warn_missing_weight_keys=True,
        verbose_reweight=False,
        replacement_map = replace_map if mode_replacement else None,
    )

    print(colored("Optimization starts", "red"))

    study_name = f"BBbar_Chi2_{run}_{channel}_replace{mode_replacement}"
    storage_url = "sqlite:///bbbar_chi2_optuna_studies.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20,),
    )
    print(f"Study: {study.study_name}")
    print(f"Existing trials: {len(study.trials)}")

    study.optimize( objective, n_trials=args.n_trials, n_jobs=1,)

    best_weights = { **study.best_params, **FIXED_WEIGHTS,}
    print( "\nBest Reduced Chi^2:", study.best_value,)
    print("Best weights:")
    for category in CATEGORIES:
        print(
            f"  {category:>22s}: "
            f"{best_weights[category]:.6f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    if args.output is None:
        output_filename = (
            f"best_bbbar_weights_ReChi2_"
            f"{run}_{channel}.json"
        )
    else:
        output_filename = args.output

    output = {
        "run": run,
        "channel": channel,
        "objective": "Reduced_Chi2",
        "best_reChi2": float(study.best_value),
        "best_weights": best_weights,
        "fitted_weights": study.best_params,
        "fixed_weights": FIXED_WEIGHTS,
        "luminosity_scale": LUMINOSITY_SCALE["all_mc"],
        "replace_modes": replace_map if mode_replacement else None,
        "best_trial_number": study.best_trial.number,
        "total_trials": len(study.trials),
    }

    with open(output_filename, "w") as output_file:
        json.dump(output, output_file, indent=2)

    print(f"\nSaved: {output_filename}")


if __name__ == "__main__":
    main()