#!/usr/bin/env python3
"""Turn tuned BBbar family weights into uncorrelated nuisance parameters.

``5_BBbkg_weights_optuna_minuit.py`` fits five generic-BBbar family weights
jointly, so their uncertainties are strongly correlated: in the stored Run-1
electron fit the measured-hadronic and 3-body weights have a correlation of
-0.77.  Assigning one independent ``normsys`` per family in the pyhf fit
would discard that structure.

This script diagonalises the covariance matrix instead.  Each eigenvector
becomes one nuisance parameter that moves all five family weights coherently
by ``sqrt(lambda_i) * v_i``, so the parameters are uncorrelated by
construction while the correlations between families are preserved.  Because
the spectrum is usually steep, two or three parameters normally carry
essentially all of the variance.

The script refuses to emit variations from a fit it cannot trust:

* a minimum sitting on a parameter bound, where the parabolic error is not
  the uncertainty on the parameter;
* stored HESSE errors that disagree with the diagonal of the stored
  covariance, which is what happens to a limited parameter pinned at its
  bound;
* a near-singular covariance, whose null directions would assign essentially
  zero uncertainty to whole decay families.

Use ``--force`` to inspect such a fit anyway; the output is then diagnostic
only and must not be used as a systematic.

Example
-------
    python3 scripts/bbbar_eigen_systematics.py \
        BBbkg_weights/best_bbbar_weights_kinematic-2d-plus-roe_alpha1_\
roetail14_run1_e_replaceFalse_minuit.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TUNING_SCRIPT = REPO_ROOT / "5_BBbkg_weights_optuna_minuit.py"

# The tuning script imports utilities and bbbar_reweighting from the
# repository root, which is not on sys.path when this script is run directly.
sys.path.insert(0, str(REPO_ROOT))

# A limited parameter pinned at its bound gets an external covariance entry
# driven to zero while `Minuit.errors` keeps a finite value.  Disagreement far
# beyond this factor means the two stored representations describe different
# things.
HESSE_COVARIANCE_TOLERANCE = 1.5
# Below this fraction of the leading eigenvalue a direction carries no usable
# information and is treated as a null direction.
NULL_EIGENVALUE_FRACTION = 1e-6
# Directions are emitted until this fraction of the total variance is covered.
DEFAULT_VARIANCE_TARGET = 0.999


def variance_fraction(value: str) -> float:
    """Parse a fraction in the interval (0, 1]."""
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return fraction


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights_json", type=pathlib.Path,
                        help="Output of 5_BBbkg_weights_optuna_minuit.py.")
    parser.add_argument("--variance-target", type=variance_fraction,
                        default=DEFAULT_VARIANCE_TARGET,
                        help="Cumulative variance fraction to cover.")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Write the variations to this JSON file.")
    parser.add_argument("--force", action="store_true",
                        help="Emit variations even if the fit fails validation.")
    return parser.parse_args()


def load_parameter_bounds():
    """Read the fitted parameter ranges from the tuning script.

    The result files do not record the bounds, so they are taken from the
    single place that defines them.  Returns None if the tuning script cannot
    be imported, in which case only positivity is enforced.
    """
    try:
        spec = importlib.util.spec_from_file_location("bbbar_tuning", TUNING_SCRIPT)
        if spec is None or spec.loader is None:
            return None, f"cannot build an import spec for {TUNING_SCRIPT}"
        module = importlib.util.module_from_spec(spec)
        sys.modules["bbbar_tuning"] = module
        spec.loader.exec_module(module)
        return {
            name: (float(item.lower), float(item.upper))
            for name, item in module.PARAMETER_SPECS.items()
        }, None
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        return None, f"{type(error).__name__}: {error}"


def check_variation_bounds(order, up, down, bounds):
    """Reject variations that leave the physical range of a family weight.

    A symmetric +/-1 sigma shift is only a valid approximation while it stays
    inside the range the weight was fitted in.  When the uncertainty exceeds
    the distance to zero or to a bound, the shifted vector contains a weight
    that would give a negative or out-of-range template yield, and the
    symmetric approximation has broken down as well.
    """
    problems = []
    for index, name in enumerate(order):
        if bounds is not None and name in bounds:
            lower, upper = bounds[name]
        else:
            lower, upper = 0.0, float("inf")
        for label, vector in (("up", up), ("down", down)):
            value = float(vector[index])
            if value < lower or value > upper:
                problems.append(
                    f"{label} variation puts '{name}' at {value:.6g}, outside "
                    f"[{lower:.6g}, {upper:.6g}]"
                )
    return problems


def validate(minuit: dict, order: list[str], covariance: np.ndarray) -> list[str]:
    """Return the reasons this fit cannot supply a systematic, if any."""
    problems: list[str] = []

    if not minuit.get("valid_minimum", False):
        problems.append("MIGRAD did not report a valid minimum.")
    if not minuit.get("accurate_covariance", False):
        problems.append("HESSE did not report an accurate covariance matrix.")
    if minuit.get("has_parameters_at_limit", False):
        problems.append(
            "The minimum sits on a parameter bound.  At a boundary the "
            "parabolic error is not the uncertainty on the parameter, so "
            "neither the HESSE errors nor the covariance can be propagated."
        )

    if not np.all(np.isfinite(covariance)):
        problems.append("The covariance contains non-finite values.")
        return problems
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        problems.append("The covariance matrix is not symmetric.")
        return problems

    hesse = minuit.get("hesse_errors", {})
    for index, name in enumerate(order):
        stored = float(hesse.get(name, float("nan")))
        from_covariance = float(np.sqrt(max(covariance[index, index], 0.0)))
        if not np.isfinite(stored) or from_covariance <= 0.0:
            continue
        ratio = max(stored, from_covariance) / max(
            min(stored, from_covariance), 1e-300
        )
        if ratio > HESSE_COVARIANCE_TOLERANCE:
            problems.append(
                f"Parameter '{name}': stored HESSE error {stored:.6g} "
                f"disagrees with sqrt(cov_ii) = {from_covariance:.6g} "
                f"by a factor {ratio:.3g}."
            )

    eigenvalues = np.linalg.eigvalsh(covariance)
    leading = eigenvalues.max()
    if leading <= 0.0:
        problems.append("The covariance is not positive definite.")
        return problems
    null_directions = int(np.sum(eigenvalues < NULL_EIGENVALUE_FRACTION * leading))
    if null_directions:
        problems.append(
            f"The covariance has {null_directions} null direction(s) "
            f"(condition number {np.linalg.cond(covariance):.3g}).  Building "
            "a systematic from it would assign essentially zero uncertainty "
            "along those directions."
        )
    return problems


def main() -> int:
    args = parse_arguments()
    payload = json.loads(args.weights_json.read_text())
    minuit = payload["minuit"]
    order = list(minuit["parameter_order"])
    central = np.array([minuit["fitted_parameters"][name] for name in order])

    # The tuning script serialises covariance as null when HESSE produced none.
    # Converting that would give a 0-d array and crash validate() while
    # indexing it, instead of following the documented refusal path.
    if minuit.get("covariance") is None:
        print(f"Fit      : {args.weights_json.name}")
        print(f"Run      : {payload.get('run')}   channel: {payload.get('channel')}")
        print("\nValidation")
        print("-" * 70)
        print("  FAIL  HESSE produced no covariance matrix for this fit, so there "
              "is nothing to propagate.")
        print("\nRefusing to emit variations.  Re-run the tuning until HESSE "
              "returns an accurate covariance.")
        return 1
    covariance = np.asarray(minuit["covariance"], dtype=float)
    if covariance.ndim != 2 or covariance.shape != (len(order), len(order)):
        print(f"  FAIL  covariance has shape {covariance.shape}, expected "
              f"{(len(order), len(order))}.")
        return 1

    print(f"Fit      : {args.weights_json.name}")
    print(f"Run      : {payload.get('run')}   channel: {payload.get('channel')}")
    print(f"Objective: {payload.get('objective')}")
    print(f"Parameters: {', '.join(order)}")
    print(f"Central   : {np.array2string(central, precision=4)}")

    problems = validate(minuit, order, covariance)
    print("\nValidation")
    print("-" * 70)
    if problems:
        for problem in problems:
            print(f"  FAIL  {problem}")
        if not args.force:
            print(
                "\nRefusing to emit variations.  Re-run the tuning so that the "
                "minimum is interior -- for example by widening or removing the "
                "lower bounds in PARAMETER_SPECS, or by merging families the "
                "tuning region cannot constrain separately.  Use --force to "
                "inspect the decomposition anyway (diagnostic only)."
            )
            return 1
        print("\n  --force given: the output below is DIAGNOSTIC ONLY.")
    else:
        print("  PASS  the fit is interior and its covariance is usable.")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order_desc = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order_desc]
    eigenvectors = eigenvectors[:, order_desc]
    total = eigenvalues.sum()
    cumulative = np.cumsum(eigenvalues) / total

    keep = int(np.searchsorted(cumulative, args.variance_target) + 1)
    keep = min(keep, len(eigenvalues))

    print("\nEigen-decomposition")
    print("-" * 70)
    print(f"  {'i':>3} {'eigenvalue':>14} {'sqrt':>12} {'var frac':>10} {'cumulative':>11}")
    for index, value in enumerate(eigenvalues):
        print(
            f"  {index + 1:>3} {value:>14.6g} {np.sqrt(abs(value)):>12.6g} "
            f"{value / total:>10.4f} {cumulative[index]:>11.4f}"
        )
    print(f"\n  {keep} direction(s) cover {cumulative[keep - 1]:.4%} of the variance.")

    bounds, bounds_error = load_parameter_bounds()
    if bounds is None:
        print(f"\n  FAIL  could not read PARAMETER_SPECS from the tuning script "
              f"({bounds_error}).  Every configured parameter has a nonzero lower "
              "and a finite upper bound, so a positivity-only check would let a "
              "variation cross a real bound while the output still claimed to be "
              "validated.")
    variations = []
    bound_problems = []
    if bounds is None:
        bound_problems.append(
            f"parameter bounds unavailable ({bounds_error}); the physical-range "
            "check could not be performed"
        )
    print("\nNuisance parameters (+1 sigma shifts of the family weights)")
    print("-" * 70)
    for index in range(keep):
        shift = np.sqrt(abs(eigenvalues[index])) * eigenvectors[:, index]
        up = central + shift
        down = central - shift
        print(f"\n  BBbar_shape_np{index + 1} "
              f"(variance fraction {eigenvalues[index] / total:.4f})")
        for name, low, mid, high in zip(order, down, central, up):
            print(f"    {name:22s} {low:9.5f} <- {mid:9.5f} -> {high:9.5f}")
        crossings = check_variation_bounds(order, up, down, bounds)
        if crossings:
            for problem in crossings:
                print(f"    OUT OF RANGE: {problem}")
            bound_problems.extend(
                f"BBbar_shape_np{index + 1}: {problem}" for problem in crossings
            )
        variations.append({
            "name": f"BBbar_shape_np{index + 1}",
            "out_of_range": crossings,
            "variance_fraction": float(eigenvalues[index] / total),
            "eigenvalue": float(eigenvalues[index]),
            "parameter_order": order,
            "central": central.tolist(),
            "up": up.tolist(),
            "down": down.tolist(),
        })

    if bound_problems:
        print("\n  FAIL  Some variations leave the physical range of a family "
              "weight:")
        for problem in bound_problems:
            print(f"    {problem}")
        print("  A negative or out-of-range weight would give negative template "
              "yields, and\n  a symmetric shift is not a valid approximation that "
              "close to a bound.  Use a\n  positivity-preserving parameterisation, "
              "or reduce the number of retained\n  directions, before using this "
              "output as a systematic.")

    if args.output:
        args.output.write_text(json.dumps({
            "source": str(args.weights_json),
            "run": payload.get("run"),
            "channel": payload.get("channel"),
            "objective": payload.get("objective"),
            "validated": not problems and not bound_problems,
            "validation_problems": problems,
            "out_of_range_problems": bound_problems,
            "variations": variations,
        }, indent=2))
        print(f"\nWrote {args.output}")

    return 0 if not problems and not bound_problems else 1


if __name__ == "__main__":
    sys.exit(main())
