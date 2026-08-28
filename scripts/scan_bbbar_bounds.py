#!/usr/bin/env python3
"""Test whether the BBbar family weights are degenerate or merely bounded.

Every result in ``BBbkg_weights/`` sits on a parameter bound, always on an
*unmeasured* n-body weight and never on the measured-hadronic weight, and the
correlations between unmeasured families reach 0.64--0.93.  Two explanations
fit that pattern:

* the bounds are simply too tight, in which case loosening them yields an
  interior minimum and a usable covariance;
* the tuning region cannot separate the unmeasured families, in which case
  the likelihood is flat along a direction and loosening the bounds only lets
  the minimum slide further before pinning again.

The two are distinguished by watching where the minimum lands as the bounds
are relaxed.  This script re-runs the iminuit stage over a sequence of bound
scalings and reports, for each, whether the minimum is interior, how much the
deviance improved, and the largest correlation among the unmeasured families.

If the minimum keeps pinning and the deviance barely moves while the weights
run away, the families are degenerate and the fix is to merge them or to add
an observable that separates them -- not to widen the bounds further.

This needs the ntuples, so it must be run in the analysis environment.

Example
-------
    python3 scripts/scan_bbbar_bounds.py --run run1 --channel e \
        --scales 1 3 10 30
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
import uproot

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import utilities as util  # noqa: E402
import bbbar_reweighting as bbbar  # noqa: E402

TUNING_SCRIPT = REPO_ROOT / "5_BBbkg_weights_optuna_minuit.py"


def relaxation_factor(value: str) -> float:
    """Parse a bound-relaxation factor, which must not tighten the bounds.

    ``relaxed_specs()`` divides the lower bound by this factor and multiplies
    the upper by it, so it only has relaxation semantics at 1 or above.  Below
    1 it tightens the range and can invert it outright -- at 0.1 the measured
    weight's range becomes [5.0, 0.5] -- which either fails while assigning
    Minuit's limits or yields a "bounds were binding" verdict from a scan that
    was tightening them.
    """
    number = float(value)
    if not np.isfinite(number) or number < 1.0:
        raise argparse.ArgumentTypeError(
            "must be a finite number at least 1; a smaller factor tightens the "
            "bounds instead of relaxing them")
    return number


def nonnegative_float(value: str) -> float:
    """Parse a finite, non-negative command-line number."""
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be a finite number at least 0")
    return number


def profile_fraction(value: str) -> float:
    """Parse a profile step as a fraction strictly between zero and one."""
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number < 1.0:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return number


def load_tuning_module():
    """Import the tuning script, whose name is not a valid module name."""
    spec = importlib.util.spec_from_file_location("bbbar_tuning", TUNING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {TUNING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bbbar_tuning"] = module
    spec.loader.exec_module(module)
    return module


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, choices=["run1", "run2", "run1+run2"])
    parser.add_argument("--channel", required=True, choices=["e", "mu"])
    parser.add_argument("--fit-model", default="kinematic-2d-plus-roe",
                        choices=["kinematic-2d", "missm2-roe-2d", "kinematic-2d-plus-roe"])
    parser.add_argument("--roe-strength", type=nonnegative_float, default=1.0)
    parser.add_argument("--roe-tail-start", type=int, default=14)
    parser.add_argument("--scales", type=relaxation_factor, nargs="+",
                        default=[1.0, 3.0, 10.0, 30.0],
                        help="Bound-relaxation factors to scan. 1 reproduces the "
                             "current PARAMETER_SPECS.")
    parser.add_argument("--sample-dir", default="/home/belle/zhangboy/inclusive_R_D/Samples",
                        help="Directory holding the BDT ntuples.")
    parser.add_argument("--minos", dest="run_minos", action="store_true",
                        help="Run the optional, slow MINOS calculation (disabled "
                             "by default; it is not needed for this diagnostic).")
    parser.add_argument("--skip-minos", dest="run_minos", action="store_false",
                        help=argparse.SUPPRESS)
    parser.add_argument("--deviance-tolerance", type=nonnegative_float, default=1.0,
                        help="Largest objective change across the scan that still "
                             "counts as 'the bounds do not matter'.  For the pure "
                             "kinematic-2d model the objective is a single Poisson "
                             "deviance with errordef = 1, so one unit is the "
                             "1-sigma scale of one parameter.  For the composite "
                             "models that add the weighted ROE term this unit is a "
                             "convention only and must be calibrated with "
                             "pseudoexperiments.  Default: 1.0.")
    parser.add_argument("--skip-profile", action="store_true",
                        help="Skip the inward profile of pinned parameters.  Without "
                             "it the flat-direction verdict cannot be issued.")
    parser.add_argument("--profile-fractions", type=profile_fraction, nargs="+",
                        default=[0.02, 0.10],
                        help="Fractions of the allowed range to step inward from a "
                             "pinned bound when profiling. Default: 0.02 0.10.")
    return parser.parse_args()


def relaxed_specs(tuning, base_specs, scale: float) -> dict:
    """Widen every range in ``base_specs`` by ``scale``.

    ``base_specs`` must always be the *original* mapping.  Reading it from
    ``tuning.PARAMETER_SPECS`` instead would compound the relaxations, because
    the scan installs each result on the module as it goes.
    """
    specs = {}
    for name, spec in base_specs.items():
        lower = spec.lower / scale
        upper = spec.upper * scale
        specs[name] = tuning.ParameterSpec(spec.category, lower, upper)
    return specs


def at_limit(value, error, spec, range_fraction=1e-6):
    """Is ``value`` effectively sitting on one of ``spec``'s bounds?

    Mirrors iminuit's own ``has_parameters_at_limit`` criterion, which compares
    the distance to the nearest bound with half the parameter's error rather
    than with the magnitude of the bound.  A bound-relative test is unusable
    here: the lower limits shrink as 1/scale, so it becomes stricter the more
    the bounds are relaxed -- about 3e-8 for the 2-body weight at the default
    30x scale -- and an effectively pinned fit would be reported as interior.
    The range floor covers a missing or zero error.
    """
    distance = min(value - spec.lower, spec.upper - value)
    width = spec.upper - spec.lower
    threshold = range_fraction * width
    if error is not None and np.isfinite(error) and error > 0:
        threshold = max(threshold, 0.5 * error)
    return distance < threshold


def objective_consistent(row, valid_rows, tolerance):
    """Is this interior minimum at least as good as every narrower fit?

    ``relaxed_specs()`` nests the ranges -- a smaller scale's box is contained
    in every larger one -- so any point found at a narrower scale is feasible
    here too.  If one of them reached a lower objective, MIGRAD settled on an
    inferior local minimum at this scale: the point is interior, but it is not
    *the* minimum, and recommending its covariance would propagate the wrong
    fit.
    """
    scale, deviance = row[0], row[1]
    return not [other for other in valid_rows
                if other[0] <= scale and other[1] < deviance - tolerance]


def profile_pinned_inward(cost, names, specs, central, pinned, fractions):
    """Profile each pinned parameter inward, re-minimising the others.

    A one-parameter scan is not enough here: along a genuinely degenerate
    direction, moving one weight while holding the rest fixed also raises the
    objective, because the compensating movement is not followed.  Only a
    profile -- fix the pinned parameter, re-minimise everything else --
    separates the two cases:

    * profile stays flat  -> the objective really is flat in that direction,
      i.e. the families are degenerate;
    * profile rises       -> the constrained optimum is genuinely at the
      bound and the data prefer a value outside the allowed range, which is a
      modelling problem rather than a degeneracy;
    * profile *falls*     -> a constrained refit found a better objective than
      the nominal fit, so the nominal fit was not the minimum.  Neither of the
      two conclusions above can be drawn from it.

    Every sampled rise is returned, not a summary of them.  The caller needs
    the largest to test flatness (flat means every sampled point stayed flat)
    and the smallest to detect the third case, and a single reduced number
    cannot serve both.  ``None`` marks a parameter whose refit failed.
    """
    from iminuit import Minuit

    base = cost(*[central[name] for name in names])
    results = {}
    for parameter in pinned:
        spec = specs[parameter]
        at_lower = abs(central[parameter] - spec.lower) <= abs(
            central[parameter] - spec.upper
        )
        bound = spec.lower if at_lower else spec.upper
        far = spec.upper if at_lower else spec.lower
        rises = []
        failed = False
        for fraction in fractions:
            target = bound + fraction * (far - bound)
            trial = Minuit(cost, *[central[name] for name in names], name=tuple(names))
            if not hasattr(cost, "errordef"):
                trial.errordef = 1.0
            for name in names:
                trial.limits[name] = (specs[name].lower, specs[name].upper)
                trial.errors[name] = max(0.01 * (specs[name].upper - specs[name].lower),
                                         0.1 * abs(central[name]))
            trial.values[parameter] = target
            trial.fixed[parameter] = True
            trial.migrad()
            if not trial.valid:
                # A constrained refit can fail precisely in the flat and
                # near-degenerate cases this diagnostic targets.  Its fval is
                # then unreliable and must not reach the verdict.
                failed = True
                break
            rises.append(float(trial.fval) - base)
        results[parameter] = None if failed else rises
    return results


def format_profile(rises, failed_text="FAILED"):
    """Render one parameter's profile result for the tables."""
    if rises is None:
        return failed_text
    if len(rises) == 1:
        return f"{rises[0]:+.3f}"
    return (f"max {max(rises):+.3f}, min {min(rises):+.3f} "
            f"({len(rises)} points)")


def load_samples(tuning, args):
    """Replicate the data loading of the tuning script's main()."""
    run_periods = ("run1", "run2") if args.run == "run1+run2" else (args.run,)
    pre_cut = (
        "(B0_roeMbc_my_mask > 5)"
        " & (-4 < B0_roeDeltae_my_mask)"
        " & (B0_roeDeltae_my_mask < 1)"
        " & (B0_dr < 0.1)"
    )
    columns = set(util.all_relevant_variables)
    mc_frames, data_frames = [], []
    for period in run_periods:
        path = (f"{args.sample_dir}/4S_{period}_deimos_BDT_{args.channel}_3.root")
        print(f"  reading {path}")
        mc = uproot.concatenate([f"{path}:MC_{args.channel}_comb"], library="pd",
                                cut=pre_cut,
                                filter_branch=lambda br: br.name in columns)
        data = uproot.concatenate([f"{path}:Data_{args.channel}_comb"], library="pd",
                                  cut=pre_cut,
                                  filter_branch=lambda br: br.name in columns)
        mc = util.apply_pid_corrections(df=mc, run=period, channel=args.channel,
                                        corr_col_name=tuning.PID_WEIGHT_COL)
        mc[tuning.RUN_PERIOD_COL] = period
        data[tuning.RUN_PERIOD_COL] = period
        mc_frames.append(mc)
        data_frames.append(data)
    return (pd.concat(mc_frames, ignore_index=True, copy=False),
            pd.concat(data_frames, ignore_index=True, copy=False))


# --------------------------------------------------------------------------
# ARCHIVED -- THIS SCRIPT WILL NOT RUN
#
# Merged as a record of an approach that was explored, not as working tooling.
# It has never been run on real ntuples, and its verdict -- whether the BBbar
# families are degenerate and should therefore be merged -- is an automated
# physics judgement that was never checked against a physicist's reading of
# the fits.
#
# Of the 24 findings review raised against these scripts, 13 were against this
# file alone, and they were still arriving at the same rate when the work
# stopped.  Almost all were cases of the scan reaching a *confident and wrong*
# conclusion rather than crashing.  That is the nature of the thing: every
# guard added another decision boundary that could itself be wrong.
#
# To take this up again, delete the guard at the top of main() deliberately,
# and read the "Review status" and "Before the output is trusted" sections of
# scripts/README.md first.
# --------------------------------------------------------------------------
ARCHIVED = (
    "scan_bbbar_bounds.py is archived and does not run.  It was never validated\n"
    "end to end.  See scripts/README.md, then delete the ARCHIVED guard at\n"
    "the top of main() in this file if you intend to take the work up again."
)


def main() -> int:
    raise SystemExit(ARCHIVED)
    args = parse_arguments()
    tuning = load_tuning_module()
    original_specs = dict(tuning.PARAMETER_SPECS)

    roe_bins = tuning.make_roe_bins(args.roe_tail_start)
    common_cut = "1.855 < D_M < 1.885 and fakeD_prob < 0.1 and sig_prob < 0.1"
    y_variable = tuning.ROE_VARIABLE if args.fit_model == "missm2-roe-2d" else "p_D_l"
    y_bins = roe_bins if args.fit_model == "missm2-roe-2d" else np.linspace(0.2, 4.0, 21)
    joint_target = tuning.JointLikelihoodTarget(
        x_variable="B0_recMissM2", x_bins=np.linspace(-4.0, 10.0, 21),
        y_variable=y_variable, y_bins=y_bins, cut=common_cut,
    )
    roe_target = None
    if args.fit_model == "kinematic-2d-plus-roe":
        roe_target = tuning.OneDimensionalLikelihoodTarget(
            variable=tuning.ROE_VARIABLE, bins=roe_bins, cut=common_cut)

    print(f"Loading {args.run}, channel={args.channel}")
    mc_combined, data_combined = load_samples(tuning, args)

    samples_base = util.classify_mc_dict(mc_combined, args.channel, template=False)
    for sample in samples_base.values():
        sample[tuning.RUN_WEIGHT_COL] = 1.0
    if args.run == "run1+run2" and "bkg_fakeD" in samples_base:
        fake_d = samples_base["bkg_fakeD"]
        fake_d[tuning.RUN_WEIGHT_COL] = fake_d[tuning.RUN_PERIOD_COL].map(
            tuning.FAKE_D_WEIGHT_BY_RUN_PERIOD)
    samples_base = bbbar.prepare_bbbar_reweighting(
        samples_base, weight_ell_side=True, cap_nbody=5,
        D_replacement_map=None, ell_replacement_map=None, copy=False)

    data_hist = tuning.get_data_histogram(data_combined, joint_target)
    fit_mask, _ = tuning.build_fixed_fit_mask(samples_base, joint_target,
                                              minimum_raw_mc_events=1)
    roe_data_hist = roe_fit_mask = None
    if roe_target is not None:
        roe_data_hist = tuning.get_data_histogram_1d(data_combined, roe_target)
        roe_fit_mask, _ = tuning.build_fixed_fit_mask_1d(samples_base, roe_target,
                                                         minimum_raw_mc_events=1)

    fixed_weights = {
        **tuning.FIXED_WEIGHTS,
        "bkg_fakeD": (tuning.FAKE_D_WEIGHT_BY_RUN_PERIOD[args.run]
                      if args.run != "run1+run2" else 1.0),
    }

    def make_cost(specs):
        """Rebuild the objective run_minuit() minimises, for profiling."""
        parameter_names = tuple(specs)

        def cost(*parameter_values):
            weights = dict(fixed_weights)
            for name, value in zip(parameter_names, parameter_values):
                weights[specs[name].category] = float(value)
            return tuning.evaluate_weights(
                weights, data_hist=data_hist, fit_mask=fit_mask,
                samples_base=samples_base, target=joint_target,
                roe_data_hist=roe_data_hist, roe_fit_mask=roe_fit_mask,
                roe_target=roe_target, roe_strength=args.roe_strength,
                replacement_map=None,
            )

        cost.errordef = 1.0
        return cost

    # missm2-roe-2d puts the ROE variable on the second axis of one joint
    # Poisson deviance; only kinematic-2d-plus-roe adds a separately weighted
    # shape-only term, and only that model's errordef is a bare convention.
    # At --roe-strength 0 the ROE term is multiplied by zero, so what is
    # actually minimised is the single 2D deviance and errordef = 1 keeps its
    # sigma interpretation.
    composite = roe_target is not None and args.roe_strength > 0
    if composite:
        print(f"\n  NOTE: objective '{args.fit_model}' combines the 2D deviance with "
              "a separately weighted\n  shape-only ROE term.  errordef = 1 is a "
              "convention for it, not a calibrated\n  1-sigma scale, so "
              f"--deviance-tolerance ({args.deviance_tolerance:g}) carries no sigma "
              "interpretation\n  here and should be calibrated with "
              "pseudoexperiments before the verdict is trusted.")

    print(f"\n{'scale':>7} {'deviance':>12} {'valid':>6} {'interior':>9} "
          f"{'max|rho| unmeas':>16}  pinned parameters")
    print("-" * 96)
    rows = []
    try:
        for scale in args.scales:
            specs = relaxed_specs(tuning, original_specs, scale)
            tuning.PARAMETER_SPECS = specs
            start = {spec.category: float(np.sqrt(max(spec.lower, 1e-6) * spec.upper))
                     for spec in specs.values()}
            minuit = tuning.run_minuit(
                start_by_category=start, data_hist=data_hist, fit_mask=fit_mask,
                samples_base=samples_base, target=joint_target,
                roe_data_hist=roe_data_hist, roe_fit_mask=roe_fit_mask,
                roe_target=roe_target, roe_strength=args.roe_strength,
                replacement_map=None, run_minos=args.run_minos,
                fixed_weights=fixed_weights,
            )
            names = list(specs)
            pinned = [
                n for n, spec in specs.items()
                if at_limit(float(minuit.values[n]), float(minuit.errors[n]), spec)
            ]
            # HESSE can fail to produce a covariance exactly when the
            # likelihood is flat, which is the case this scan exists to find.
            # Guard it the way the tuning script's minuit_results() does.
            if minuit.covariance is None:
                max_rho = float("nan")
                rho_text = "no covariance"
            else:
                correlation = np.array(minuit.covariance.correlation())
                unmeasured = [i for i, n in enumerate(names) if n != "measured"]
                max_rho = max(
                    (abs(correlation[a, b])
                     for a, b in itertools.combinations(unmeasured, 2)),
                    default=float("nan"),
                )
                rho_text = f"{max_rho:.3f}"
            print(f"{scale:>7g} {minuit.fval:>12.3f} "
                  f"{'yes' if minuit.valid else 'NO':>6} "
                  f"{'yes' if not pinned else 'NO':>9} {rho_text:>16}  "
                  f"{', '.join(pinned) if pinned else '-'}")

            profile = None
            if pinned and minuit.valid and not args.skip_profile:
                central = {n: float(minuit.values[n]) for n in names}
                profile = profile_pinned_inward(
                    make_cost(specs), names, specs, central, pinned,
                    args.profile_fractions,
                )
                for parameter, rises in profile.items():
                    print(f"{'':>7} profile inward, {parameter:22s} "
                          f"delta(objective) = {format_profile(rises)}")

            rows.append((scale, float(minuit.fval), not pinned, max_rho,
                         {n: float(minuit.values[n]) for n in names},
                         minuit.covariance is not None, bool(minuit.valid),
                         profile))
    finally:
        tuning.PARAMETER_SPECS = original_specs

    return interpret_scan(rows, args, composite)


def interpret_scan(rows, args, composite) -> int:
    """Turn the scanned rows into the scan's verdict.

    Kept separate from main() so that every branch can be exercised
    against constructed rows.  Each of the verdicts below is a physics
    conclusion, and review has repeatedly found ways for the wrong one
    to be issued confidently, so they need to be testable without a fit.
    """
    print("\nInterpretation")
    print("-" * 96)
    if not rows:
        print("  No fits completed; nothing to interpret.")
        return 1

    # A failed fit says nothing about the bounds: its parameter values and its
    # objective are both unreliable, so it must not enter the verdict.
    valid_rows = [row for row in rows if row[6]]
    invalid_scales = [row[0] for row in rows if not row[6]]
    if invalid_scales:
        print("  Excluded from the verdict (MIGRAD did not converge): "
              f"scale(s) {', '.join(f'{value:g}' for value in invalid_scales)}")

    if not valid_rows:
        print("\n  VERDICT: no valid fit at any scale.  The scan cannot "
              "distinguish tight bounds from a flat direction; investigate the "
              "minimisation itself before drawing a physics conclusion.")
        return 1

    deviances = [row[1] for row in valid_rows]
    valid_scales = [row[0] for row in valid_rows]
    # Count *distinct* scales.  Repeating a scale (``--scales 1 1``) produces
    # two converged rows whose deviance span is trivially zero, which would
    # otherwise satisfy the flatness test without any range of bounds having
    # been examined at all.
    distinct_scales = sorted(set(valid_scales))
    span = max(deviances) - min(deviances)
    scale_range = max(valid_scales) / min(valid_scales)
    tolerance = args.deviance_tolerance
    interior_rows = [row for row in valid_rows if row[2]]
    usable_interior = [row for row in interior_rows
                       if objective_consistent(row, valid_rows, tolerance)]
    inferior_interior = [row for row in interior_rows if row not in usable_interior]

    print(f"  valid scales                    : "
          f"{', '.join(f'{value:g}' for value in valid_scales)}"
          f"{' (repeated)' if len(distinct_scales) < len(valid_scales) else ''}")
    print(f"  deviance change across them     : {span:.3f}")
    print(f"  bound range covered             : {scale_range:g}x")
    print(f"  tolerance (--deviance-tolerance): {tolerance:g}")
    print()
    if inferior_interior:
        print("  Interior minima excluded as inferior local minima (a narrower "
              "scale, whose\n  range they contain, reached a lower objective): "
              + ", ".join(f"scale {row[0]:g} at {row[1]:.3f}"
                          for row in inferior_interior))

    if usable_interior:
        first = usable_interior[0]
        print(f"  VERDICT: bounds were the binding constraint.  A valid interior "
              f"minimum is reached at scale {first[0]:g}.")
        if first[5]:
            print("  Re-run the tuning with these bounds and feed the covariance to "
                  "scripts/bbbar_eigen_systematics.py.")
        else:
            print("  NOTE: HESSE returned no covariance at that scale, so the fit "
                  "is interior but its uncertainties are not yet propagable.")
    elif inferior_interior:
        # Every interior minimum found is beaten by a narrower fit nested
        # inside it.  The scan cannot conclude the bounds were binding, and it
        # cannot conclude flatness either: the pinned rows are not the whole
        # picture once MIGRAD is known to be missing minima at these scales.
        print("  VERDICT: inconclusive -- the minimisation is unreliable.  The "
              "only interior minima found are beaten by a fit at a narrower "
              "scale whose range they contain, so MIGRAD is settling on local "
              "minima rather than the best one at each scale.  Re-minimise from "
              "several starting points before reading anything off this scan.")
    elif len(distinct_scales) < 2:
        repeated = len(valid_rows) > 1
        print("  VERDICT: inconclusive.  "
              + ("Every valid fit came from the same relaxation scale "
                 f"({distinct_scales[0]:g}), so the scan covered no range of "
                 "bounds at all and its zero deviance span carries no "
                 "information."
                 if repeated else
                 "Only one scale produced a valid fit, so there is no range of "
                 "bounds to compare and neither a binding bound nor a flat "
                 "direction can be established.")
              + "  Extend or adjust the scan so that at least two distinct "
                "scales converge.")
    else:
        # A small deviance span is necessary but not sufficient for flatness.
        # If the unconstrained optimum lies outside every relaxed range -- a
        # family whose preferred weight is at or below zero -- then every fit
        # pins, and the span shrinks towards zero as the bound approaches zero,
        # with no degeneracy anywhere.  The profile separates the two: it stays
        # flat for a real degeneracy and rises for a boundary-constrained
        # optimum.
        profiles = [row[7] for row in valid_rows if row[7]]
        # A failed constrained refit carries no information.  Be conservative:
        # do not silently discard it and then declare flatness from a different
        # scale.  Likewise, inspect every successful profile rather than only
        # the last one; a rise at any scanned scale disproves the claim that all
        # of the profiled directions are flat.
        failed_profiles = {
            parameter
            for prof in profiles
            for parameter, rise in prof.items() if rise is None
        }
        sampled = {
            (scale, parameter): values
            for scale, *_, profile in valid_rows
            if profile
            for parameter, values in profile.items()
            if values is not None
        }
        # Flatness is tested against the largest sampled rise, so that a
        # direction rising past the tolerance at any sampled point is not
        # called flat by a flatter neighbouring point.
        rises = {key: max(values) for key, values in sampled.items()}
        # A refit that lands *below* the nominal objective says the nominal fit
        # was not the minimum.  Such a delta is negative, can never satisfy
        # ``rise > tolerance``, and would otherwise pass silently into the
        # flat-direction verdict.
        improving = {key: min(values) for key, values in sampled.items()}
        if args.skip_profile or not profiles:
            print("  VERDICT: inconclusive without a profile.  The deviance moved by "
                  f"{span:.3f} across a {scale_range:g}x range of bounds with every "
                  "minimum pinned, which is consistent with a flat direction but "
                  "equally with an optimum lying outside the allowed range.  Re-run "
                  "without --skip-profile to separate the two.")
        elif failed_profiles:
            print("  VERDICT: inconclusive.  At least one inward profile had a "
                  "constrained refit that did not converge "
                  f"({', '.join(sorted(failed_profiles))}), so the claim that every "
                  "pinned direction is flat cannot be tested.  A failed refit is "
                  "itself common in near-degenerate problems; try more profile "
                  "fractions or a different starting point.")
        elif any(rise < -tolerance for rise in improving.values()):
            best = min(improving.items(), key=lambda item: item[1])
            print("  VERDICT: inconclusive -- the nominal fit is unreliable.  "
                  f"Fixing {best[0][1]} away from its bound at scale "
                  f"{best[0][0]:g} and re-minimising the rest found an "
                  f"objective {best[1]:+.3f} BELOW the nominal minimum, more "
                  f"than the {tolerance:g} tolerance.  A constrained fit cannot "
                  "beat the unconstrained one, so the nominal fit reached a "
                  "local or inaccurate minimum and neither its deviance nor its "
                  "pinned parameters describe the likelihood.  Re-minimise from "
                  "several starting points before interpreting this scan.")
        else:
            rising = {key: rise for key, rise in rises.items() if rise > tolerance}
            if rising:
                worst = max(rising.items(), key=lambda item: item[1])
                print("  VERDICT: boundary-constrained optimum, NOT a flat "
                      "direction.  Profiling the pinned parameters inward raises "
                      f"the objective (largest: {worst[0][1]} at scale "
                      f"{worst[0][0]:g} by {worst[1]:+.3f}), so "
                      "the data prefer a value outside the allowed range rather "
                      "than being indifferent along that direction.  Merging "
                      "families would not address this; investigate why the "
                      "preferred weight lies outside its physical range.")
            elif span <= tolerance:
                print("  VERDICT: flat likelihood direction.  No valid scale gives "
                      f"an interior minimum, the deviance moved by {span:.3f} across "
                      f"a {scale_range:g}x range of bounds, and profiling every "
                      "pinned parameter inward changes the objective by no more "
                      f"than the {tolerance:g} tolerance.  The families are "
                      "degenerate in this region: merge the unmeasured families, or "
                      "add an observable that separates them.")
                if composite:
                    print("  Calibrate --deviance-tolerance with pseudoexperiments "
                          "before acting on this verdict; for the composite "
                          "objective it is a convention, not a 1-sigma scale.")
            else:
                print("  VERDICT: inconclusive.  No valid scale gives an interior "
                      f"minimum, but the deviance improved by {span:.3f}, more than "
                      f"the {tolerance:g} tolerance, so the bounds are still "
                      "materially affecting the fit.  Extend the scan to larger "
                      "scales until either an interior minimum appears or the "
                      "deviance stops improving.")

    for scale, deviance, is_interior, max_rho, values, has_cov, valid, profile in rows:
        rho_text = "n/a (no covariance)" if not has_cov else f"{max_rho:.3f}"
        flag = "" if valid else "   [EXCLUDED: MIGRAD did not converge]"
        print(f"\n  scale {scale:g}: deviance {deviance:.3f}, interior={is_interior}, "
              f"valid={valid}, max|rho|(unmeasured)={rho_text}{flag}")
        if profile:
            for parameter, rises in profile.items():
                print(f"    profile inward {parameter:22s} "
                      f"{format_profile(rises, 'FAILED (excluded)')}")
        for name, value in values.items():
            print(f"    {name:22s} {value:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
