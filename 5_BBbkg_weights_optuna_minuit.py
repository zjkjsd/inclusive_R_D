"""Tune Belle II BBbar background weights with configurable binned fits.

The script uses Optuna for global exploration and iminuit/MIGRAD for the final
local minimization. HESSE supplies the covariance matrix and MINOS supplies
profile-likelihood intervals. The fitted-bin mask is built once from raw,
unweighted nominal MC support and is then held fixed.  The fit can use the
original (missing-mass squared, D-lepton momentum) target, a joint
(missing-mass squared, ROE track multiplicity) target, or the original target
plus a shape-only ROE track-multiplicity constraint.
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
import bbbar_reweighting as bbbar


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
class OneDimensionalLikelihoodTarget:
    variable: str
    bins: np.ndarray
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

# This correction is measured separately for each data-taking period.  Keep it
# in one named mapping so that the likelihood setup and the serialized weight
# representation cannot silently diverge.
FAKE_D_WEIGHT_BY_RUN_PERIOD: Dict[str, float] = {
    "run1": 0.87,
    "run2": 1.0,
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
RUN_PERIOD_COL = "__run_period__"
RUN_WEIGHT_COL = "__run_weight__"
ROE_VARIABLE = "B0_nROE_Tracks_my_mask"
DEFAULT_ROE_TAIL_START = 14


def make_roe_bins(tail_start: int) -> np.ndarray:
    """Return unit-width integer bins followed by one ``>= tail_start`` bin."""
    if tail_start < 1:
        raise ValueError("ROE tail start must be at least 1.")
    return np.concatenate(
        (np.arange(-0.5, tail_start + 0.5, 1.0), np.array([np.inf]))
    )


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
        "--fit-model",
        choices=["kinematic-2d", "missm2-roe-2d", "kinematic-2d-plus-roe"],
        default="kinematic-2d-plus-roe",
        help=(
            "Fit objective: original kinematic 2D, missing-mass/ROE 2D, or "
            "kinematic 2D plus a shape-only ROE constraint. Default: "
            "kinematic-2d-plus-roe."
        ),
    )
    parser.add_argument(
        "--roe-strength",
        type=float,
        default=1.0,
        help=(
            "Coefficient multiplying the shape-only ROE deviance in the "
            "composite model. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--roe-tail-start",
        type=int,
        default=DEFAULT_ROE_TAIL_START,
        help=(
            "First ROE track multiplicity included in the merged overflow "
            f"bin. Default: {DEFAULT_ROE_TAIL_START} (the final bin is >=14)."
        ),
    )
    parser.add_argument(
        "--run",
        required=True,
        choices=["run1", "run2", "run1+run2"],
        help="Data-taking period, including the combined run1+run2 sample.",
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
    args = parser.parse_args()
    if args.roe_strength < 0 or not np.isfinite(args.roe_strength):
        parser.error("--roe-strength must be a finite non-negative number.")
    if args.roe_tail_start < 1:
        parser.error("--roe-tail-start must be at least 1.")
    return args


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


def _histogram1d(
    values: np.ndarray,
    bins: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    return hist.astype(float)


def combined_event_weights(
    df: pd.DataFrame,
    *,
    columns: tuple[Optional[str], ...] = (
        BASE_WEIGHT_COL,
        PID_WEIGHT_COL,
        TUNE_WEIGHT_COL,
        RUN_WEIGHT_COL,
    ),
    scale: float = 1.0,
) -> np.ndarray:
    """Multiply available event-weight columns and reject invalid inputs."""
    weights = np.full(len(df), float(scale), dtype=float)
    for column in columns:
        if column is not None and column in df.columns:
            weights *= df[column].to_numpy(dtype=float)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Event weights contain non-finite values.")
    return weights


def get_data_histogram_1d(
    df_data: pd.DataFrame,
    target: OneDimensionalLikelihoodTarget,
) -> np.ndarray:
    selected = _safe_query(df_data, target.cut)
    return _histogram1d(
        selected[target.variable].to_numpy(dtype=float), target.bins
    )


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


def get_raw_mc_support_histogram_1d(
    samples_mc: Mapping[str, pd.DataFrame],
    target: OneDimensionalLikelihoodTarget,
) -> np.ndarray:
    total = np.zeros(len(target.bins) - 1, dtype=float)
    for df_sample in samples_mc.values():
        if df_sample is None or df_sample.empty:
            continue
        selected = _safe_query(df_sample, target.cut)
        if selected.empty:
            continue
        total += _histogram1d(
            selected[target.variable].to_numpy(dtype=float), target.bins
        )
    return total


def build_fixed_fit_mask_1d(
    samples_mc: Mapping[str, pd.DataFrame],
    target: OneDimensionalLikelihoodTarget,
    minimum_raw_mc_events: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    raw_mc_hist = get_raw_mc_support_histogram_1d(samples_mc, target)
    fit_mask = raw_mc_hist >= minimum_raw_mc_events
    if not np.any(fit_mask):
        raise RuntimeError("No 1D bins pass the raw-MC support requirement.")
    return fit_mask, raw_mc_hist


def get_weighted_mc_histogram(
    samples_mc: Mapping[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    *,
    luminosity_scale: float,
    base_weight_col: Optional[str] = BASE_WEIGHT_COL,
    tune_weight_col: Optional[str] = TUNE_WEIGHT_COL,
    event_weight_col: Optional[str] = PID_WEIGHT_COL,
    run_weight_col: Optional[str] = RUN_WEIGHT_COL,
) -> np.ndarray:
    shape = (len(target.x_bins) - 1, len(target.y_bins) - 1)
    total = np.zeros(shape, dtype=float)

    for df_sample in samples_mc.values():
        if df_sample is None or df_sample.empty:
            continue
        selected = _safe_query(df_sample, target.cut)
        if selected.empty:
            continue

        weights = combined_event_weights(
            selected,
            columns=(base_weight_col, event_weight_col, tune_weight_col, run_weight_col),
            scale=luminosity_scale,
        )

        total += _histogram2d(
            selected[target.x_variable].to_numpy(dtype=float),
            selected[target.y_variable].to_numpy(dtype=float),
            target.x_bins,
            target.y_bins,
            weights=weights,
        )
    return total


def get_weighted_mc_histogram_1d(
    samples_mc: Mapping[str, pd.DataFrame],
    target: OneDimensionalLikelihoodTarget,
    *,
    luminosity_scale: float,
    base_weight_col: Optional[str] = BASE_WEIGHT_COL,
    tune_weight_col: Optional[str] = TUNE_WEIGHT_COL,
    event_weight_col: Optional[str] = PID_WEIGHT_COL,
    run_weight_col: Optional[str] = RUN_WEIGHT_COL,
) -> np.ndarray:
    shape = (len(target.bins) - 1,)
    total = np.zeros(shape, dtype=float)
    for df_sample in samples_mc.values():
        if df_sample is None or df_sample.empty:
            continue
        selected = _safe_query(df_sample, target.cut)
        if selected.empty:
            continue
        weights = combined_event_weights(
            selected,
            columns=(base_weight_col, event_weight_col, tune_weight_col, run_weight_col),
            scale=luminosity_scale,
        )
        total += _histogram1d(
            selected[target.variable].to_numpy(dtype=float),
            target.bins,
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


def shape_only_poisson_deviance(
    observed: np.ndarray, expected: np.ndarray
) -> float:
    """Poisson shape deviance after matching the expected data integral."""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError(
            f"Observed and expected shapes differ: {observed.shape} vs "
            f"{expected.shape}."
        )
    if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(expected)):
        return np.inf
    observed_sum = observed.sum()
    expected_sum = expected.sum()
    if observed_sum <= 0 or expected_sum <= 0:
        return np.inf
    return poisson_deviance(observed, expected * (observed_sum / expected_sum))


def apply_category_weights(
    samples_base: Dict[str, pd.DataFrame],
    weights: Mapping[str, float],
    *,
    replacement_map,
) -> Dict[str, pd.DataFrame]:
    """Apply a parameter point to the invariant, prepared classification."""
    samples_weighted = bbbar.apply_bbbar_weights(
        samples_base,
        weights,
        out_weight_col=TUNE_WEIGHT_COL,
        weight_ell_side=True,
        copy=False,
        warn_missing_weight_keys=True,
    )
    # Enforce the externally determined fake-D normalization explicitly. This
    # avoids relying on the BBbar-specific utility to handle this component.
    if "bkg_fakeD" in samples_weighted:
        samples_weighted["bkg_fakeD"][TUNE_WEIGHT_COL] = float(
            weights["bkg_fakeD"]
        )
    return samples_weighted


def evaluate_loss_components(
    weights: Mapping[str, float],
    *,
    data_hist: np.ndarray,
    fit_mask: np.ndarray,
    samples_base: Dict[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    roe_data_hist: Optional[np.ndarray],
    roe_fit_mask: Optional[np.ndarray],
    roe_target: Optional[OneDimensionalLikelihoodTarget],
    roe_strength: float,
    replacement_map,
) -> Dict[str, float]:
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
    deviance_2d = poisson_deviance(data_hist[fit_mask], mc_hist[fit_mask])
    deviance_roe = 0.0
    if roe_target is not None:
        if roe_data_hist is None or roe_fit_mask is None:
            raise ValueError("ROE data histogram and mask are required.")
        roe_mc_hist = get_weighted_mc_histogram_1d(
            samples_weighted,
            roe_target,
            luminosity_scale=LUMINOSITY_SCALE["all_mc"],
        )
        deviance_roe = shape_only_poisson_deviance(
            roe_data_hist[roe_fit_mask], roe_mc_hist[roe_fit_mask]
        )
    return {
        "joint_2d": float(deviance_2d),
        "roe_1d_shape": float(deviance_roe),
        "roe_strength": float(roe_strength),
        "total": float(deviance_2d + roe_strength * deviance_roe),
    }


def evaluate_weights(**kwargs) -> float:
    return evaluate_loss_components(**kwargs)["total"]


# ============================================================
# Optuna global exploration
# ============================================================


def make_optuna_objective(
    *,
    data_hist: np.ndarray,
    fit_mask: np.ndarray,
    samples_base: Dict[str, pd.DataFrame],
    target: JointLikelihoodTarget,
    roe_data_hist: Optional[np.ndarray],
    roe_fit_mask: Optional[np.ndarray],
    roe_target: Optional[OneDimensionalLikelihoodTarget],
    roe_strength: float,
    replacement_map,
    fixed_weights: Mapping[str, float] = FIXED_WEIGHTS,
):
    def objective(trial: optuna.Trial) -> float:
        weights = dict(fixed_weights)
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
            roe_data_hist=roe_data_hist,
            roe_fit_mask=roe_fit_mask,
            roe_target=roe_target,
            roe_strength=roe_strength,
            replacement_map=replacement_map,
        )
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("total_deviance", float(deviance))
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
    roe_data_hist: Optional[np.ndarray],
    roe_fit_mask: Optional[np.ndarray],
    roe_target: Optional[OneDimensionalLikelihoodTarget],
    roe_strength: float,
    replacement_map,
    run_minos: bool,
    fixed_weights: Mapping[str, float] = FIXED_WEIGHTS,
) -> Minuit:
    parameter_names = tuple(PARAMETER_SPECS)

    def cost(*parameter_values: float) -> float:
        weights = dict(fixed_weights)
        for name, value in zip(parameter_names, parameter_values):
            weights[PARAMETER_SPECS[name].category] = float(value)
        return evaluate_weights(
            weights,
            data_hist=data_hist,
            fit_mask=fit_mask,
            samples_base=samples_base,
            target=target,
            roe_data_hist=roe_data_hist,
            roe_fit_mask=roe_fit_mask,
            roe_target=roe_target,
            roe_strength=roe_strength,
            replacement_map=replacement_map,
        )

    # errordef=1 is exact for either single 2D Poisson model.  For the
    # correlated 2D+1D composite objective it is a useful convention, but its
    # intervals require bootstrap/pseudoexperiment validation before they can
    # be interpreted as exact 68% frequentist intervals.
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
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def weights_by_run_period(
    weights: Mapping[str, float],
    run_periods: tuple[str, ...],
    *,
    fake_d_correction_applied: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Return the effective category weights for each included run period.

    In the combined fit, ``weights['bkg_fakeD']`` is the common category
    factor and the period correction is carried by ``RUN_WEIGHT_COL``.  This
    expansion records their product, which is the representation downstream
    users need when they do not have the fit's private event-level columns.
    """
    expanded = {}
    for period in run_periods:
        if period not in FAKE_D_WEIGHT_BY_RUN_PERIOD:
            raise ValueError(f"No fake-D correction is configured for {period!r}.")
        period_weights = {key: float(value) for key, value in weights.items()}
        if not fake_d_correction_applied:
            period_weights["bkg_fakeD"] *= FAKE_D_WEIGHT_BY_RUN_PERIOD[period]
        expanded[period] = period_weights
    return expanded


def minuit_results(
    minuit: Minuit,
    fixed_weights: Mapping[str, float] = FIXED_WEIGHTS,
) -> dict:
    names = list(PARAMETER_SPECS)
    fitted_by_safe_name = {name: float(minuit.values[name]) for name in names}
    fitted_by_category = {
        PARAMETER_SPECS[name].category: float(minuit.values[name]) for name in names
    }
    all_weights = {**fitted_by_category, **fixed_weights}
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
    # The combined sample uses an event-level run correction below, so its
    # common fake-D factor is unity rather than applying the run1 factor to all
    # events.
    fixed_weights = {
        **FIXED_WEIGHTS,
        # A single-period fit can put the correction directly in the category
        # weight.  A combined fit needs a common factor here and applies the
        # differing corrections through RUN_WEIGHT_COL below.
        "bkg_fakeD": FAKE_D_WEIGHT_BY_RUN_PERIOD[run] if run != "run1+run2" else 1.0,
    }
    channel = args.channel
    mode_replacement = args.mode_replacement
    replacement_map = util.hadronicB_replacement_map if mode_replacement else None
    fit_model = args.fit_model
    roe_bins = make_roe_bins(args.roe_tail_start)

    print(colored(f"Loading {run}, channel={channel}", "blue"))
    columns = util.all_relevant_variables
    pre_cut = (
        "(B0_roeMbc_my_mask > 5)"
        " & (-4 < B0_roeDeltae_my_mask)"
        " & (B0_roeDeltae_my_mask < 1)"
        " & (B0_dr < 0.1)"
    )
    run_periods = ("run1", "run2") if run == "run1+run2" else (run,)
    input_files = {
        period: (
            "/home/belle/zhangboy/inclusive_R_D/"
            f"Samples/4S_{period}_deimos_BDT_{channel}_3.root"
        )
        for period in run_periods
    }
    mc_tree = f"MC_{channel}_comb"
    data_tree = f"Data_{channel}_comb"
    for period, input_file in input_files.items():
        print(f"Input file ({period}): {input_file}")
    print(f"MC tree:     {mc_tree}")
    print(f"Data tree:   {data_tree}")

    mc_frames = []
    data_frames = []
    for period, input_file in input_files.items():
        mc_period = uproot.concatenate(
            [f"{input_file}:{mc_tree}"],
            library="pd",
            cut=pre_cut,
            filter_branch=lambda branch: branch.name in columns,
        )
        data_period = uproot.concatenate(
            [f"{input_file}:{data_tree}"],
            library="pd",
            cut=pre_cut,
            filter_branch=lambda branch: branch.name in columns,
        )
        # PID corrections are run-dependent and must be evaluated before the
        # frames are combined.
        mc_period = util.apply_pid_corrections(
            df=mc_period,
            run=period,
            channel=channel,
            corr_col_name=PID_WEIGHT_COL,
        )
        mc_period[RUN_PERIOD_COL] = period
        data_period[RUN_PERIOD_COL] = period
        mc_frames.append(mc_period)
        data_frames.append(data_period)
    mc_combined = pd.concat(mc_frames, ignore_index=True, copy=False)
    data_combined = pd.concat(data_frames, ignore_index=True, copy=False)

    common_cut = "1.855 < D_M < 1.885 and fakeD_prob < 0.1 and sig_prob < 0.1"
    y_variable = ROE_VARIABLE if fit_model == "missm2-roe-2d" else "p_D_l"
    y_bins = roe_bins if fit_model == "missm2-roe-2d" else np.linspace(0.2, 4.0, 21)
    joint_target = JointLikelihoodTarget(
        x_variable="B0_recMissM2",
        x_bins=np.linspace(-4.0, 10.0, 21),
        y_variable=y_variable,
        y_bins=y_bins,
        cut=common_cut,
    )
    roe_target = None
    if fit_model == "kinematic-2d-plus-roe":
        roe_target = OneDimensionalLikelihoodTarget(
            variable=ROE_VARIABLE, bins=roe_bins, cut=common_cut
        )
    print(colored(f"Joint likelihood target:\n{joint_target}", "green"))

    # Classify once and make a fixed fit-bin definition before optimization.
    samples_base = util.classify_mc_dict(mc_combined, channel, template=False)
    # The externally determined fake-D normalization differs between run1
    # (0.87) and run2 (1.0).  Preserve that distinction in the combined fit
    # with an event-level factor; all other categories receive unity.
    for sample in samples_base.values():
        sample[RUN_WEIGHT_COL] = 1.0
    if run == "run1+run2" and "bkg_fakeD" in samples_base:
        fake_d = samples_base["bkg_fakeD"]
        fake_d[RUN_WEIGHT_COL] = fake_d[RUN_PERIOD_COL].map(
            FAKE_D_WEIGHT_BY_RUN_PERIOD
        )
        if fake_d[RUN_WEIGHT_COL].isna().any():
            unknown_periods = sorted(
                fake_d.loc[fake_d[RUN_WEIGHT_COL].isna(), RUN_PERIOD_COL]
                .astype(str)
                .unique()
            )
            raise ValueError(
                "No fake-D correction is configured for run period(s): "
                f"{unknown_periods}."
            )
    # Decay classification and replacement factors do not depend on fit
    # parameters, so calculate them once rather than during every likelihood
    # evaluation.
    samples_base = bbbar.prepare_bbbar_reweighting(
        samples_base,
        weight_ell_side=True,
        cap_nbody=5,
        D_replacement_map=replacement_map,
        ell_replacement_map=replacement_map,
        copy=False,
    )
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

    roe_data_hist = None
    roe_fit_mask = None
    roe_raw_mc_hist = None
    if roe_target is not None:
        roe_data_hist = get_data_histogram_1d(data_combined, roe_target)
        roe_fit_mask, roe_raw_mc_hist = build_fixed_fit_mask_1d(
            samples_base, roe_target, minimum_raw_mc_events=1
        )
        print(f"ROE included bins:  {int(roe_fit_mask.sum())} / {roe_fit_mask.size}")
        print(f"ROE included data:  {roe_data_hist[roe_fit_mask].sum():.0f}")
        print(f"ROE excluded data:  {roe_data_hist[~roe_fit_mask].sum():.0f}")

    objective = make_optuna_objective(
        data_hist=data_hist,
        fit_mask=fit_mask,
        samples_base=samples_base,
        target=joint_target,
        roe_data_hist=roe_data_hist,
        roe_fit_mask=roe_fit_mask,
        roe_target=roe_target,
        roe_strength=args.roe_strength,
        replacement_map=replacement_map,
        fixed_weights=fixed_weights,
    )
    uses_roe = fit_model != "kinematic-2d"
    strength_tag = f"_alpha{args.roe_strength:g}" if roe_target is not None else ""
    roe_bins_tag = f"_roetail{args.roe_tail_start}" if uses_roe else ""
    study_name = (
        f"BBbar_mle_{run}_{channel}_{fit_model}{strength_tag}{roe_bins_tag}_"
        f"replace{mode_replacement}"
    )
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

    optuna_best_weights = {**study.best_params, **fixed_weights}
    print(f"\nBest Optuna deviance over all stored trials: {study.best_value:.6f}")

    print(colored("iminuit MIGRAD/HESSE refinement starts", "red"))
    minuit = run_minuit(
        start_by_category=optuna_best_weights,
        data_hist=data_hist,
        fit_mask=fit_mask,
        samples_base=samples_base,
        target=joint_target,
        roe_data_hist=roe_data_hist,
        roe_fit_mask=roe_fit_mask,
        roe_target=roe_target,
        roe_strength=args.roe_strength,
        replacement_map=replacement_map,
        run_minos=not args.skip_minos,
        fixed_weights=fixed_weights,
    )
    print(minuit.fmin)
    print(minuit.params)
    if minuit.covariance is not None:
        print(minuit.covariance)

    result = minuit_results(minuit, fixed_weights)
    loss_components = evaluate_loss_components(
        result["best_weights"],
        data_hist=data_hist,
        fit_mask=fit_mask,
        samples_base=samples_base,
        target=joint_target,
        roe_data_hist=roe_data_hist,
        roe_fit_mask=roe_fit_mask,
        roe_target=roe_target,
        roe_strength=args.roe_strength,
        replacement_map=replacement_map,
    )
    print(f"\nFinal Minuit deviance: {result['minimum_deviance']:.6f}")
    print(
        "Loss components: "
        f"2D={loss_components['joint_2d']:.6f}, "
        f"ROE shape={loss_components['roe_1d_shape']:.6f}, "
        f"alpha={loss_components['roe_strength']:.6g}"
    )
    print("Final weights:")
    for category in CATEGORIES:
        print(f"  {category:>32s}: {result['best_weights'][category]:.6f}")

    output_path = args.output
    if output_path is None:
        output_path = Path(
            f"best_bbbar_weights_{fit_model}{strength_tag}{roe_bins_tag}_{run}_{channel}_"
            f"replace{mode_replacement}_minuit.json"
        )

    trial_counts = {
        state.name.lower(): sum(trial.state == state for trial in study.trials)
        for state in TrialState
    }
    output = {
        "run": run,
        "run_periods": list(run_periods),
        "input_files": input_files,
        "channel": channel,
        "objective": fit_model,
        "roe_strength": args.roe_strength if roe_target is not None else None,
        "roe_tail_start": args.roe_tail_start if uses_roe else None,
        "replace_modes": mode_replacement,
        "replacement_map": replacement_map,
        "luminosity_scale": LUMINOSITY_SCALE["all_mc"],
        "fixed_weights": fixed_weights,
        "fixed_weights_by_run_period": weights_by_run_period(
            fixed_weights,
            run_periods,
            fake_d_correction_applied=run != "run1+run2",
        ),
        "weight_representation": {
            "category_column": TUNE_WEIGHT_COL,
            "run_period_column": RUN_PERIOD_COL,
            "run_factor_column": RUN_WEIGHT_COL,
            "event_weight_product": [
                BASE_WEIGHT_COL,
                PID_WEIGHT_COL,
                TUNE_WEIGHT_COL,
                RUN_WEIGHT_COL,
            ],
            "fakeD_period_correction": FAKE_D_WEIGHT_BY_RUN_PERIOD,
            "fakeD_period_correction_application": (
                RUN_WEIGHT_COL if run == "run1+run2" else TUNE_WEIGHT_COL
            ),
        },
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
        "roe_target": None if roe_target is None else {
            "variable": roe_target.variable,
            "bins": roe_target.bins.tolist(),
            "cut": roe_target.cut,
            "minimum_raw_mc_events_per_included_bin": 1,
            "included_bins": int(roe_fit_mask.sum()),
            "total_bins": int(roe_fit_mask.size),
            "included_data_events": float(roe_data_hist[roe_fit_mask].sum()),
            "excluded_data_events": float(roe_data_hist[~roe_fit_mask].sum()),
            "raw_mc_events_in_included_bins": float(
                roe_raw_mc_hist[roe_fit_mask].sum()
            ),
        },
        "loss_components": loss_components,
        "optuna": {
            "study_name": study.study_name,
            "storage": args.storage,
            "best_deviance": float(study.best_value),
            "best_trial_number": int(study.best_trial.number),
            "best_weights": optuna_best_weights,
            "best_weights_by_run_period": weights_by_run_period(
                optuna_best_weights,
                run_periods,
                fake_d_correction_applied=run != "run1+run2",
            ),
            "total_trials": len(study.trials),
            "trial_counts": trial_counts,
        },
        "minuit": result,
    }
    result["best_weights_by_run_period"] = weights_by_run_period(
        result["best_weights"],
        run_periods,
        fake_d_correction_applied=run != "run1+run2",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(_json_safe(output), output_file, indent=2, default=_json_default)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
