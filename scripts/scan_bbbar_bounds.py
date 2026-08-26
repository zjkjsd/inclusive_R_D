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
        --scales 1 3 10 30 --skip-minos
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
    parser.add_argument("--roe-strength", type=float, default=1.0)
    parser.add_argument("--roe-tail-start", type=int, default=14)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0],
                        help="Bound-relaxation factors to scan. 1 reproduces the "
                             "current PARAMETER_SPECS.")
    parser.add_argument("--sample-dir", default="/home/belle/zhangboy/inclusive_R_D/Samples",
                        help="Directory holding the BDT ntuples.")
    parser.add_argument("--skip-minos", action="store_true", default=True,
                        help="MINOS is slow and not needed for this diagnostic.")
    parser.add_argument("--deviance-tolerance", type=float, default=1.0,
                        help="Largest deviance change across the scan that still "
                             "counts as 'the bounds do not matter'.  The cost has "
                             "errordef = 1, so one unit is the 1-sigma scale of a "
                             "single parameter.  Default: 1.0.")
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


def main() -> int:
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
                replacement_map=None, run_minos=not args.skip_minos,
                fixed_weights=fixed_weights,
            )
            names = list(specs)
            pinned = [
                n for n, spec in specs.items()
                if abs(minuit.values[n] - spec.lower) / max(abs(spec.lower), 1e-9) < 1e-3
                or abs(minuit.values[n] - spec.upper) / max(abs(spec.upper), 1e-9) < 1e-3
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
            rows.append((scale, float(minuit.fval), not pinned, max_rho,
                         {n: float(minuit.values[n]) for n in names},
                         minuit.covariance is not None, bool(minuit.valid)))
    finally:
        tuning.PARAMETER_SPECS = original_specs

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
    span = max(deviances) - min(deviances)
    scale_range = max(valid_scales) / min(valid_scales)
    tolerance = args.deviance_tolerance
    interior_rows = [row for row in valid_rows if row[2]]

    print(f"  valid scales                    : "
          f"{', '.join(f'{value:g}' for value in valid_scales)}")
    print(f"  deviance change across them     : {span:.3f}")
    print(f"  bound range covered             : {scale_range:g}x")
    print(f"  tolerance (--deviance-tolerance): {tolerance:g}")
    print()
    if interior_rows:
        first = interior_rows[0]
        print(f"  VERDICT: bounds were the binding constraint.  A valid interior "
              f"minimum is reached at scale {first[0]:g}.")
        if first[5]:
            print("  Re-run the tuning with these bounds and feed the covariance to "
                  "scripts/bbbar_eigen_systematics.py.")
        else:
            print("  NOTE: HESSE returned no covariance at that scale, so the fit "
                  "is interior but its uncertainties are not yet propagable.")
    elif len(valid_rows) < 2:
        print("  VERDICT: inconclusive.  Only one scale produced a valid fit, so "
              "there is no range of bounds to compare and neither a binding bound "
              "nor a flat direction can be established.  Extend or adjust the scan "
              "so that at least two scales converge.")
    elif span <= tolerance:
        print("  VERDICT: flat likelihood direction.  No valid scale gives an "
              f"interior minimum, and the deviance moved by {span:.3f}, within the "
              f"{tolerance:g} tolerance, across a {scale_range:g}x range of bounds. "
              "Widening the bounds further will not help; merge the unmeasured "
              "families, or add an observable that separates them.")
    else:
        print("  VERDICT: inconclusive.  No valid scale gives an interior minimum, "
              f"but the deviance improved by {span:.3f}, more than the "
              f"{tolerance:g} tolerance, so the bounds are still materially "
              "affecting the fit.  This is not evidence of a flat direction.  "
              "Extend the scan to larger scales until either an interior minimum "
              "appears or the deviance stops improving.")

    for scale, deviance, is_interior, max_rho, values, has_cov, valid in rows:
        rho_text = "n/a (no covariance)" if not has_cov else f"{max_rho:.3f}"
        flag = "" if valid else "   [EXCLUDED: MIGRAD did not converge]"
        print(f"\n  scale {scale:g}: deviance {deviance:.3f}, interior={is_interior}, "
              f"valid={valid}, max|rho|(unmeasured)={rho_text}{flag}")
        for name, value in values.items():
            print(f"    {name:22s} {value:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
