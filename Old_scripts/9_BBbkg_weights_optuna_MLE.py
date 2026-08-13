import argparse
import json
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from termcolor import colored
import uproot
import optuna

import utilities as util


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class JointLikelihoodTarget:
    x_variable: str
    x_bins: np.ndarray
    y_variable: str
    y_bins: np.ndarray
    cut: str = ""


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


def _histogram2d(
    x: np.ndarray,
    y: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return a 2D histogram with shape:

        (len(x_bins) - 1, len(y_bins) - 1)
    """
    hist, _, _ = np.histogram2d(
        x=x,
        y=y,
        bins=[x_bins, y_bins],
        weights=weights,
    )

    return hist.astype(float)


def get_joint_data_mc_histograms(
    df_data: pd.DataFrame,
    samples_mc: Dict[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    *,
    luminosity_scale: float = 1.0,
    base_weight_col: Optional[str] = BASE_WEIGHT_COL,
    tune_weight_col: Optional[str] = TUNE_WEIGHT_COL,
    event_weight_col: Optional[str] = PID_WEIGHT_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct joint 2D histograms for data and total MC.

    Returns
    -------
    data_hist:
        Unweighted observed event counts.

    mc_hist:
        Weighted expected MC event counts.
    """

    # --------------------------------------------------------
    # Data histogram
    # --------------------------------------------------------
    data_selected = _safe_query(df_data, target.cut)

    data_hist = _histogram2d(
        x=data_selected[target.x_variable].to_numpy(dtype=float),
        y=data_selected[target.y_variable].to_numpy(dtype=float),
        x_bins=target.x_bins,
        y_bins=target.y_bins,
        weights=None,
    )

    # --------------------------------------------------------
    # Total MC histogram
    # --------------------------------------------------------
    histogram_shape = ( len(target.x_bins) - 1, len(target.y_bins) - 1,)
    mc_hist_total = np.zeros( histogram_shape, dtype=float,)

    for sample_name, df_sample in samples_mc.items():
        if df_sample is None or df_sample.empty:
            continue

        mc_selected = _safe_query( df_sample, target.cut,)

        if mc_selected.empty:
            continue

        weights = np.ones( len(mc_selected), dtype=float,)

        # Generator/base event weight
        if (
            base_weight_col is not None
            and base_weight_col in mc_selected.columns
        ):
            weights *= mc_selected[base_weight_col].to_numpy(dtype=float)

        # PID or other event-by-event correction
        if (
            event_weight_col is not None
            and event_weight_col in mc_selected.columns
        ):
            weights *= mc_selected[event_weight_col].to_numpy(dtype=float)

        # Tuned BBbar weight
        if (
            tune_weight_col is not None
            and tune_weight_col in mc_selected.columns
        ):
            weights *= mc_selected[tune_weight_col].to_numpy(dtype=float)

        weights *= float(luminosity_scale)

        sample_hist = _histogram2d(
            x=mc_selected[target.x_variable].to_numpy(dtype=float),
            y=mc_selected[target.y_variable].to_numpy(dtype=float),
            x_bins=target.x_bins,
            y_bins=target.y_bins,
            weights=weights,
        )

        mc_hist_total += sample_hist


    # --------------------------------------------------------
    # Trimming and flattening
    # --------------------------------------------------------
    # Determine which bins pass the threshold based on sum of all templates
    indices_threshold = np.where(mc_hist_total >= 1)

    # Flatten the templates after cutting
    mc_hist_total_flat = mc_hist_total[indices_threshold]
    # Asimov data is the sum of all templates
    data_hist_flat = data_hist[indices_threshold]

    return data_hist_flat, mc_hist_total_flat


# ============================================================
# Joint Poisson likelihood
# ============================================================

def poisson_deviance(observed: np.ndarray, expected: np.ndarray,) -> float:
    """
    Calculate the Poisson deviance over all 2D bins:

        D = 2 * sum_i [
            mu_i - n_i + n_i * log(n_i / mu_i)
        ]

    The two-dimensional arrays are flattened implicitly by
    the NumPy operations.
    """

    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)

    if observed.shape != expected.shape:
        raise ValueError(
            "Observed and expected histograms have "
            f"different shapes: {observed.shape} and "
            f"{expected.shape}."
        )

    if np.any(~np.isfinite(observed)):
        return np.inf

    if np.any(~np.isfinite(expected)):
        return np.inf

    if np.any(observed < 0):
        raise ValueError("Observed Poisson counts cannot be negative.")

    # Negative expected yields are not valid Poisson means.
    if np.any(expected < 0):
        return np.inf

    # If n > 0 and mu = 0, the likelihood is zero.
    if np.any((observed > 0) & (expected <= 0)):
        return np.inf

    terms = expected - observed

    positive_data_mask = observed > 0

    terms[positive_data_mask] += (
        observed[positive_data_mask]
        * np.log(
            observed[positive_data_mask]
            / expected[positive_data_mask]
        )
    )

    return float(2.0 * np.sum(terms))


def joint_poisson_deviance_data_mc(
    df_data: pd.DataFrame,
    samples_mc: Dict[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    *,
    luminosity_scale: float = 1.0,
    base_weight_col: Optional[str] = BASE_WEIGHT_COL,
    tune_weight_col: Optional[str] = TUNE_WEIGHT_COL,
    event_weight_col: Optional[str] = PID_WEIGHT_COL,
) -> float:
    data_hist, mc_hist = get_joint_data_mc_histograms(
        df_data=df_data,
        samples_mc=samples_mc,
        target=target,
        luminosity_scale=luminosity_scale,
        base_weight_col=base_weight_col,
        tune_weight_col=tune_weight_col,
        event_weight_col=event_weight_col,
    )

    return poisson_deviance( observed=data_hist, expected=mc_hist,)


# ============================================================
# Optuna objective
# ============================================================

def make_objective(
    df_data: pd.DataFrame,
    df_mc: pd.DataFrame,
    mode: str,
    target: JointLikelihoodTarget,
    *,
    cap_nbody: Optional[int] = 5,
    warn_missing_weight_keys: bool = True,
    verbose_reweight: bool = False,
    replacement_map = None,
):
    # Classify the MC only once.
    samples_base = util.classify_mc_dict( df_mc, mode, template=False,)

    def objective(trial: optuna.Trial) -> float:
        weights: Dict[str, float] = {}

        for cat in CATEGORIES:
            if cat in FIXED_WEIGHTS:
                weights[cat] = float(FIXED_WEIGHTS[cat])
            else:
                if cat in ['BBbar_measured_hadronic','BBbar_unmeasured:2-body',]:
                    weights[cat] = trial.suggest_float(cat, 0.001, 5.0, log=True)
                elif cat in ['BBbar_unmeasured:3-body','BBbar_unmeasured:4-body']:
                    weights[cat] = trial.suggest_float(cat, 0.001, 1.0, log=True)
                elif cat=='BBbar_unmeasured:5+-body':
                    weights[cat] = trial.suggest_float(cat, 0.001, 0.1, log=True)

               
        # This updates BB_weight on every trial.
        samples_weighted = util.reweight_BBbar_background(samples_base,
                                                      weight_map = weights,
                                                      out_weight_col = TUNE_WEIGHT_COL,
                                                      weight_ell_side = True, 
                                                      verbose= False, 
                                                      cap_nbody = cap_nbody, 
                                                      warn_missing_weight_keys = warn_missing_weight_keys,
                                                      D_replacement_map = replacement_map,
                                                      ell_replacement_map = replacement_map,)


        deviance = joint_poisson_deviance_data_mc(
            df_data=df_data,
            samples_mc=samples_weighted,
            target=target,
            luminosity_scale=LUMINOSITY_SCALE["all_mc"],
            base_weight_col=BASE_WEIGHT_COL,
            tune_weight_col=TUNE_WEIGHT_COL,
            event_weight_col=PID_WEIGHT_COL,
        )

        trial.set_user_attr("weights", weights)
        trial.set_user_attr(
            "joint_poisson_deviance",
            float(deviance),
        )

        trial.report(deviance, step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(deviance)

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
    # Joint 2D likelihood target
    # --------------------------------------------------------
    joint_target = JointLikelihoodTarget(
        x_variable="B0_recMissM2",
        x_bins=np.linspace(-4.0, 10.0, 21),
        y_variable="p_D_l",
        y_bins=np.linspace(0.2, 4.0, 21),
        cut=(
            "1.855 < D_M < 1.885 "
            "and fakeD_prob<0.1 and sig_prob<0.1"
        ),
    )

    print(
        colored(
            f"Joint likelihood target:\n{joint_target}",
            "green",
        )
    )

    objective = make_objective(
        df_data=data_4S_comb,
        df_mc=MC_4S_comb,
        mode=channel,
        target=joint_target,
        cap_nbody=5,
        warn_missing_weight_keys=True,
        verbose_reweight=False,
        replacement_map = replace_map if mode_replacement else None,
    )

    print(colored("Optimization starts", "red"))

    study_name = f"BBbar_mle_{run}_{channel}_replace{mode_replacement}"
    storage_url = "sqlite:///bbbar_mle_optuna_studies.db"
    
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
    print("\nBest joint Poisson deviance:", study.best_value,)
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
            f"best_bbbar_weights_poisson_2d_"
            f"{run}_{channel}.json"
        )
    else:
        output_filename = args.output

    output = {
        "run": run,
        "channel": channel,
        "objective": "joint_2d_poisson_deviance",
        "best_deviance": float(study.best_value),
        "best_weights": best_weights,
        "fitted_weights": study.best_params,
        "fixed_weights": FIXED_WEIGHTS,
        "luminosity_scale": LUMINOSITY_SCALE["all_mc"],
        "replace_modes": replace_map if mode_replacement else None,
        "target": {
            "x_variable": joint_target.x_variable,
            "x_bins": joint_target.x_bins.tolist(),
            "y_variable": joint_target.y_variable,
            "y_bins": joint_target.y_bins.tolist(),
            "cut": joint_target.cut,
        },
        "best_trial_number": study.best_trial.number,
        "total_trials": len(study.trials),
    }

    with open(output_filename, "w") as output_file:
        json.dump(output, output_file, indent=2)

    print(f"\nSaved: {output_filename}")


if __name__ == "__main__":
    main()