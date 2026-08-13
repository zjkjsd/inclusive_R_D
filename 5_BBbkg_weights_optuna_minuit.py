"""Tune Belle II BBbar background weights with a joint 2D Poisson fit.

The script uses Optuna for global exploration and iminuit/MIGRAD for the final
local minimization. HESSE supplies the covariance matrix and MINOS supplies
profile-likelihood intervals. The fitted-bin mask is built once from raw,
unweighted nominal MC support and is then held fixed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional
from termcolor import colored
from time import perf_counter

import numpy as np
import pandas as pd
import uproot
import optuna
from optuna.trial import TrialState
from iminuit import Minuit

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


@dataclass(frozen=True)
class ParameterSpec:
    category: str
    lower: float
    upper: float


# Safe Python/Minuit parameter name -> MC category and allowed range.
PARAMETER_SPECS: Dict[str, ParameterSpec] = {
    "measured": ParameterSpec("BBbar_measured_hadronic", 0.5, 5.0),
    "unmeasured_2body": ParameterSpec("BBbar_unmeasured:2-body", 0.001, 5.0),
    "unmeasured_3body": ParameterSpec("BBbar_unmeasured:3-body", 0.01, 1.0),
    "unmeasured_4body": ParameterSpec("BBbar_unmeasured:4-body", 0.001, 1.0),
    "unmeasured_5plus": ParameterSpec("BBbar_unmeasured:5+-body", 0.001, 0.5),
}

FIXED_WEIGHTS: Dict[str, float] = {
    "BBbar_semileptonic": 1.0,
    "bkg_fakeD": 0.87,
}

CATEGORIES = [
    "bkg_fakeD",
    "BBbar_measured_hadronic",
    "BBbar_semileptonic",
    "BBbar_unmeasured:2-body",
    "BBbar_unmeasured:3-body",
    "BBbar_unmeasured:4-body",
    "BBbar_unmeasured:5+-body",
]

LUMINOSITY_SCALE = {"all_mc": 0.25}
TUNE_WEIGHT_COL = "BB_weight"
BASE_WEIGHT_COL = "__weight__"
PID_WEIGHT_COL = "total_PID_weight"


# ============================================================
# Command-line arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune BBbar MC weights with a joint 2D binned Poisson likelihood, "
            "using Optuna followed by iminuit."
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
        help="Number of NEW Optuna trials to add. Default: 200.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Optuna random seed. Default: 123.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. A run/channel-specific name is used by default.",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///bbbar_mle_optuna_studies.db",
        help="Optuna storage URL.",
    )
    parser.add_argument(
        "--mode-replacement",
        "--mode_replacement",
        dest="mode_replacement",
        action="store_true",
        help="Replace configured non-resonant 3/4-body modes before tuning.",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip new Optuna trials and use the stored best trial as Minuit start.",
    )
    parser.add_argument(
        "--skip-minos",
        action="store_true",
        help="Run MIGRAD and HESSE but skip the slower MINOS calculation.",
    )
    return parser.parse_args()


# ============================================================
# Histogram and likelihood helpers
# ============================================================


def _safe_query(df: pd.DataFrame, cut: str) -> pd.DataFrame:
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
    hist, _, _ = np.histogram2d(
        x=x,
        y=y,
        bins=[x_bins, y_bins],
        weights=weights,
    )
    return hist.astype(float)


def get_data_histogram(
    df_data: pd.DataFrame,
    target: JointLikelihoodTarget,
) -> np.ndarray:
    selected = _safe_query(df_data, target.cut)
    return _histogram2d(
        selected[target.x_variable].to_numpy(dtype=float),
        selected[target.y_variable].to_numpy(dtype=float),
        target.x_bins,
        target.y_bins,
    )


def get_raw_mc_support_histogram(
    samples_mc: Mapping[str, pd.DataFrame],
    target: JointLikelihoodTarget,
) -> np.ndarray:
    """Count raw MC events; do not use luminosity or correction weights."""
    shape = (len(target.x_bins) - 1, len(target.y_bins) - 1)
    total = np.zeros(shape, dtype=float)

    for df_sample in samples_mc.values():
        if df_sample is None or df_sample.empty:
            continue
        selected = _safe_query(df_sample, target.cut)
        if selected.empty:
            continue
        total += _histogram2d(
            selected[target.x_variable].to_numpy(dtype=float),
            selected[target.y_variable].to_numpy(dtype=float),
            target.x_bins,
            target.y_bins,
        )
    return total


def build_fixed_fit_mask(
    samples_mc: Mapping[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    minimum_raw_mc_events: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    raw_mc_hist = get_raw_mc_support_histogram(samples_mc, target)
    fit_mask = raw_mc_hist >= minimum_raw_mc_events
    if not np.any(fit_mask):
        raise RuntimeError("No 2D bins pass the raw-MC support requirement.")
    return fit_mask, raw_mc_hist


def get_weighted_mc_histogram(
    samples_mc: Mapping[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    *,
    luminosity_scale: float,
    base_weight_col: Optional[str] = BASE_WEIGHT_COL,
    tune_weight_col: Optional[str] = TUNE_WEIGHT_COL,
    event_weight_col: Optional[str] = PID_WEIGHT_COL,
) -> np.ndarray:
    shape = (len(target.x_bins) - 1, len(target.y_bins) - 1)
    total = np.zeros(shape, dtype=float)

    for df_sample in samples_mc.values():
        if df_sample is None or df_sample.empty:
            continue
        selected = _safe_query(df_sample, target.cut)
        if selected.empty:
            continue

        weights = np.ones(len(selected), dtype=float)
        for column in (base_weight_col, event_weight_col, tune_weight_col):
            if column is not None and column in selected.columns:
                weights *= selected[column].to_numpy(dtype=float)
        weights *= float(luminosity_scale)

        if np.any(~np.isfinite(weights)):
            return np.full(shape, np.nan, dtype=float)

        total += _histogram2d(
            selected[target.x_variable].to_numpy(dtype=float),
            selected[target.y_variable].to_numpy(dtype=float),
            target.x_bins,
            target.y_bins,
            weights=weights,
        )
    return total


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    """Return -2 log(L/L_saturated) for independent Poisson bins."""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)

    if observed.shape != expected.shape:
        raise ValueError(
            f"Observed and expected shapes differ: {observed.shape} vs "
            f"{expected.shape}."
        )
    if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(expected)):
        return np.inf
    if np.any(observed < 0):
        raise ValueError("Observed Poisson counts cannot be negative.")
    if np.any(expected < 0):
        return np.inf
    if np.any((observed > 0) & (expected <= 0)):
        return np.inf

    terms = expected - observed
    positive = observed > 0
    terms[positive] += observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return float(2.0 * np.sum(terms))


def apply_category_weights(
    samples_base: Dict[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    replacement_map,
) -> Dict[str, pd.DataFrame]:
    """Apply the current parameter point using the user's utility function."""
    samples_weighted = util.reweight_BBbar_background(
        samples_base,
        weight_map=dict(weights),
        out_weight_col=TUNE_WEIGHT_COL,
        weight_ell_side=True,
        verbose=False,
        cap_nbody=5,
        warn_missing_weight_keys=True,
        D_replacement_map=replacement_map,
        ell_replacement_map=replacement_map,
    )
    # Enforce the externally determined fake-D normalization explicitly. This
    # avoids relying on the BBbar-specific utility to handle this component.
    if "bkg_fakeD" in samples_weighted:
        samples_weighted["bkg_fakeD"][TUNE_WEIGHT_COL] = float(
            weights["bkg_fakeD"]
        )
    return samples_weighted


def evaluate_weights(
    weights: Mapping[str, float],
    *,
    data_hist: np.ndarray,
    fit_mask: np.ndarray,
    samples_base: Dict[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    replacement_map,
) -> float:
    samples_weighted = apply_category_weights(
        samples_base,
        weights,
        replacement_map=replacement_map,
    )
    mc_hist = get_weighted_mc_histogram(
        samples_weighted,
        target,
        luminosity_scale=LUMINOSITY_SCALE["all_mc"],
    )
    return poisson_deviance(data_hist[fit_mask], mc_hist[fit_mask])


# ============================================================
# Optuna global exploration
# ============================================================


def make_optuna_objective(
    *,
    data_hist: np.ndarray,
    fit_mask: np.ndarray,
    samples_base: Dict[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    replacement_map,
):
    def objective(trial: optuna.Trial) -> float:
        weights = dict(FIXED_WEIGHTS)
        for spec in PARAMETER_SPECS.values():
            weights[spec.category] = trial.suggest_float(
                spec.category,
                spec.lower,
                spec.upper,
                log=True,
            )

        deviance = evaluate_weights(
            weights,
            data_hist=data_hist,
            fit_mask=fit_mask,
            samples_base=samples_base,
            target=target,
            replacement_map=replacement_map,
        )
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("joint_poisson_deviance", float(deviance))
        trial.report(deviance, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(deviance)

    return objective


# ============================================================
# Minuit local fit and uncertainties
# ============================================================


def run_minuit(
    *,
    start_by_category: Mapping[str, float],
    data_hist: np.ndarray,
    fit_mask: np.ndarray,
    samples_base: Dict[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    replacement_map,
    run_minos: bool,
) -> Minuit:
    parameter_names = tuple(PARAMETER_SPECS)

    def cost(*parameter_values: float) -> float:
        weights = dict(FIXED_WEIGHTS)
        for name, value in zip(parameter_names, parameter_values):
            weights[PARAMETER_SPECS[name].category] = float(value)
        return evaluate_weights(
            weights,
            data_hist=data_hist,
            fit_mask=fit_mask,
            samples_base=samples_base,
            target=target,
            replacement_map=replacement_map,
        )

    # The cost is a Poisson deviance (-2 log likelihood ratio), so Delta=1
    # corresponds to a one-parameter asymptotic 68% interval.
    cost.errordef = 1.0

    start_values = [
        float(start_by_category[PARAMETER_SPECS[name].category])
        for name in parameter_names
    ]
    minuit = Minuit(cost, *start_values, name=parameter_names)

    for name, start in zip(parameter_names, start_values):
        spec = PARAMETER_SPECS[name]
        minuit.limits[name] = (spec.lower, spec.upper)
        minuit.errors[name] = max(0.01 * (spec.upper - spec.lower), 0.1 * start)

    minuit.strategy = 1
    
    print("MIGRAD starts", flush=True)
    start = perf_counter()
    minuit.migrad()
    print(f"MIGRAD finished in {perf_counter() - start:.1f} s",  flush=True,)
    
    print("HESSE starts", flush=True)
    start = perf_counter()
    minuit.hesse()
    print( f"HESSE finished in {perf_counter() - start:.1f} s", flush=True, )


    if run_minos and minuit.valid:
        print("MINOS starts", flush=True)
        start = perf_counter()

        try:
            minuit.minos()
        except RuntimeError as error:
            print(f"MINOS failed: {error}", flush=True)
        else:
            print( f"MINOS finished in " f"{perf_counter() - start:.1f} s", flush=True, )

    return minuit


# ============================================================
# Output helpers
# ============================================================


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize object of type {type(value).__name__}")


def _json_safe(value):
    """Recursively make replacement maps with non-string keys JSON-safe."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def minuit_results(minuit: Minuit) -> dict:
    names = list(PARAMETER_SPECS)
    fitted_by_safe_name = {name: float(minuit.values[name]) for name in names}
    fitted_by_category = {
        PARAMETER_SPECS[name].category: float(minuit.values[name]) for name in names
    }
    all_weights = {**fitted_by_category, **FIXED_WEIGHTS}
    hesse_errors = {name: float(minuit.errors[name]) for name in names}

    minos_errors = {}
    for name in names:
        if name not in minuit.merrors:
            continue
        error = minuit.merrors[name]
        minos_errors[name] = {
            "lower": float(error.lower),
            "upper": float(error.upper),
            "is_valid": bool(getattr(error, "is_valid", False)),
            "lower_valid": bool(getattr(error, "lower_valid", False)),
            "upper_valid": bool(getattr(error, "upper_valid", False)),
            "at_lower_limit": bool(getattr(error, "at_lower_limit", False)),
            "at_upper_limit": bool(getattr(error, "at_upper_limit", False)),
        }

    covariance = None
    correlation = None
    if minuit.covariance is not None:
        covariance = [
            [float(minuit.covariance[a, b]) for b in names] for a in names
        ]
        corr = minuit.covariance.correlation()
        correlation = [[float(corr[a, b]) for b in names] for a in names]

    return {
        "minimum_deviance": float(minuit.fval),
        "valid_minimum": bool(minuit.valid),
        "accurate_covariance": bool(minuit.accurate),
        "edm": float(minuit.fmin.edm),
        "has_parameters_at_limit": bool(minuit.fmin.has_parameters_at_limit),
        "has_posdef_covar": bool(minuit.fmin.has_posdef_covar),
        "parameter_order": names,
        "fitted_parameters": fitted_by_safe_name,
        "best_weights": all_weights,
        "hesse_errors": hesse_errors,
        "minos_errors": minos_errors,
        "covariance": covariance,
        "correlation": correlation,
    }


# ============================================================
# Main
# ============================================================


def main() -> None:
    args = parse_arguments()
    run = args.run
    if run=='run2':
        FIXED_WEIGHTS['bkg_fakeD'] = 1
    channel = args.channel
    mode_replacement = args.mode_replacement
    replacement_map = util.hadronicB_replacement_map if mode_replacement else None

    print(colored(f"Loading {run}, channel={channel}", "blue"))
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

    mc_combined = uproot.concatenate(
        [f"{input_file}:{mc_tree}"],
        library="pd",
        cut=pre_cut,
        filter_branch=lambda branch: branch.name in columns,
    )
    data_combined = uproot.concatenate(
        [f"{input_file}:{data_tree}"],
        library="pd",
        cut=pre_cut,
        filter_branch=lambda branch: branch.name in columns,
    )
    mc_combined = util.apply_pid_corrections(
        df=mc_combined,
        run=run,
        channel=channel,
        corr_col_name=PID_WEIGHT_COL,
    )

    joint_target = JointLikelihoodTarget(
        x_variable="B0_recMissM2",
        x_bins=np.linspace(-4.0, 10.0, 21),
        y_variable="p_D_l",
        y_bins=np.linspace(0.2, 4.0, 21),
        cut="1.855 < D_M < 1.885 and fakeD_prob < 0.1 and sig_prob < 0.1",
    )
    print(colored(f"Joint likelihood target:\n{joint_target}", "green"))

    # Classify once and make a fixed fit-bin definition before optimization.
    samples_base = util.classify_mc_dict(mc_combined, channel, template=False)
    data_hist = get_data_histogram(data_combined, joint_target)
    fit_mask, raw_mc_hist = build_fixed_fit_mask(
        samples_base,
        joint_target,
        minimum_raw_mc_events=1,
    )
    print(f"Included bins:      {int(fit_mask.sum())} / {fit_mask.size}")
    print(f"Included data:      {data_hist[fit_mask].sum():.0f}")
    print(f"Excluded data:      {data_hist[~fit_mask].sum():.0f}")
    print(f"Raw MC in fit bins: {raw_mc_hist[fit_mask].sum():.0f}")

    objective = make_optuna_objective(
        data_hist=data_hist,
        fit_mask=fit_mask,
        samples_base=samples_base,
        target=joint_target,
        replacement_map=replacement_map,
    )
    study_name = f"BBbar_mle_{run}_{channel}_replace{mode_replacement}"
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
    )
    print(f"Study: {study.study_name}")
    print(f"Existing trials: {len(study.trials)}")

    if not args.skip_optuna and args.n_trials > 0:
        print(colored("Optuna global exploration starts", "red"))
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

    completed_trials = [
        trial for trial in study.trials if trial.state == TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError(
            "The study has no completed trial. Run without --skip-optuna first."
        )

    optuna_best_weights = {**study.best_params, **FIXED_WEIGHTS}
    print(f"\nBest Optuna deviance over all stored trials: {study.best_value:.6f}")

    print(colored("iminuit MIGRAD/HESSE refinement starts", "red"))
    minuit = run_minuit(
        start_by_category=optuna_best_weights,
        data_hist=data_hist,
        fit_mask=fit_mask,
        samples_base=samples_base,
        target=joint_target,
        replacement_map=replacement_map,
        run_minos=not args.skip_minos,
    )
    print(minuit.fmin)
    print(minuit.params)
    if minuit.covariance is not None:
        print(minuit.covariance)

    result = minuit_results(minuit)
    print(f"\nFinal Minuit deviance: {result['minimum_deviance']:.6f}")
    print("Final weights:")
    for category in CATEGORIES:
        print(f"  {category:>32s}: {result['best_weights'][category]:.6f}")

    output_path = args.output
    if output_path is None:
        output_path = Path(
            f"best_bbbar_weights_poisson_2d_{run}_{channel}_"
            f"replace{mode_replacement}_minuit.json"
        )

    trial_counts = {
        state.name.lower(): sum(trial.state == state for trial in study.trials)
        for state in TrialState
    }
    output = {
        "run": run,
        "channel": channel,
        "objective": "joint_2d_poisson_deviance",
        "replace_modes": mode_replacement,
        "replacement_map": replacement_map,
        "luminosity_scale": LUMINOSITY_SCALE["all_mc"],
        "fixed_weights": FIXED_WEIGHTS,
        "target": {
            "x_variable": joint_target.x_variable,
            "x_bins": joint_target.x_bins.tolist(),
            "y_variable": joint_target.y_variable,
            "y_bins": joint_target.y_bins.tolist(),
            "cut": joint_target.cut,
            "minimum_raw_mc_events_per_included_bin": 1,
            "included_bins": int(fit_mask.sum()),
            "total_bins": int(fit_mask.size),
            "included_data_events": float(data_hist[fit_mask].sum()),
            "excluded_data_events": float(data_hist[~fit_mask].sum()),
        },
        "optuna": {
            "study_name": study.study_name,
            "storage": args.storage,
            "best_deviance": float(study.best_value),
            "best_trial_number": int(study.best_trial.number),
            "best_weights": optuna_best_weights,
            "total_trials": len(study.trials),
            "trial_counts": trial_counts,
        },
        "minuit": result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(_json_safe(output), output_file, indent=2, default=_json_default)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
