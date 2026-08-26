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
    return parser.parse_args()


def relaxed_specs(tuning, scale: float) -> dict:
    """Widen every parameter range by ``scale`` about its geometric centre."""
    specs = {}
    for name, spec in tuning.PARAMETER_SPECS.items():
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

    print(f"\n{'scale':>7} {'deviance':>12} {'interior':>9} "
          f"{'max|rho| unmeas':>16}  pinned parameters")
    print("-" * 88)
    rows = []
    try:
        for scale in args.scales:
            specs = relaxed_specs(tuning, scale)
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
            correlation = np.array(minuit.covariance.correlation())
            unmeasured = [i for i, n in enumerate(names) if n != "measured"]
            max_rho = max(
                (abs(correlation[a, b]) for a, b in itertools.combinations(unmeasured, 2)),
                default=float("nan"),
            )
            print(f"{scale:>7g} {minuit.fval:>12.3f} "
                  f"{'yes' if not pinned else 'NO':>9} {max_rho:>16.3f}  "
                  f"{', '.join(pinned) if pinned else '-'}")
            rows.append((scale, minuit.fval, not pinned, max_rho,
                         {n: float(minuit.values[n]) for n in names}))
    finally:
        tuning.PARAMETER_SPECS = original_specs

    print("\nInterpretation")
    print("-" * 88)
    deviances = [row[1] for row in rows]
    interior = [row[2] for row in rows]
    if any(interior):
        first = args.scales[interior.index(True)]
        print(f"  An interior minimum is reached at scale {first:g}: the bounds were "
              "the binding constraint.  Re-run the tuning with these bounds and "
              "feed the covariance to scripts/bbbar_eigen_systematics.py.")
    else:
        span = max(deviances) - min(deviances)
        print(f"  No scale gives an interior minimum, and the deviance moved by only "
              f"{span:.2f} across a {max(args.scales) / min(args.scales):g}x range of "
              "bounds.  That is the signature of a flat likelihood direction: the "
              "families are degenerate in this region.  Widening bounds will not "
              "help; merge the unmeasured families, or add an observable that "
              "separates them.")
    for scale, deviance, is_interior, max_rho, values in rows:
        print(f"\n  scale {scale:g}: deviance {deviance:.3f}, "
              f"interior={is_interior}, max|rho|(unmeasured)={max_rho:.3f}")
        for name, value in values.items():
            print(f"    {name:22s} {value:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
