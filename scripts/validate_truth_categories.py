#!/usr/bin/env python3
"""Check that the MC truth categories are exclusive and exhaustive.

``utilities.classify_mc_dict()`` builds each category with an independent
``DataFrame.query``.  Nothing in the code guarantees that the resulting
subsets partition the input: exclusivity relies on invariants of the MC truth
record, and the fake-object branch has no catch-all, so candidates with
``D_mcErrors > 512`` match no category at all and are dropped silently.

This script measures both properties on a real ntuple so the analysis note can
state them as a measurement rather than an assumption.

Example
-------
    python3 scripts/validate_truth_categories.py \
        Samples/4S_run1_deimos_BDT_e_3.root --tree MC_e_comb --channel e
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

# Invoked as "python3 scripts/validate_truth_categories.py", Python puts
# scripts/ on sys.path, not the repository root, so "import utilities" would
# fail in a normal checkout.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import uproot

import utilities as util

# classify_mc_dict() builds the merged gap category with
# pd.concat(..., ignore_index=True) (utilities.py:632), so that one sample's
# DataFrame index is renumbered from zero and no longer identifies the
# original candidate.  Membership is therefore tracked through an explicit
# identifier column that survives concatenation, never through the index.
CANDIDATE_ID = "__cand_id__"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", help="ROOT ntuple to check.")
    parser.add_argument("--tree", required=True, help="Tree name, e.g. MC_e_comb.")
    parser.add_argument("--channel", required=True, choices=["e", "mu"])
    parser.add_argument(
        "--cut",
        default="",
        help="Optional pre-selection applied before classification.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Read at most this many entries (for a quick check).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    columns = set(util.all_relevant_variables)

    print(f"Reading {args.input_file}:{args.tree}")
    df = uproot.concatenate(
        [f"{args.input_file}:{args.tree}"],
        library="pd",
        cut=args.cut or None,
        filter_branch=lambda branch: branch.name in columns,
        entry_stop=args.max_entries,
    )
    df = df.reset_index(drop=True)
    df[CANDIDATE_ID] = range(len(df))
    print(f"Candidates read: {len(df)}")

    samples = util.classify_mc_dict(df, args.channel, template=False)
    for name, subset in samples.items():
        if len(subset) and CANDIDATE_ID not in subset.columns:
            print(
                f"ERROR: category '{name}' lost the {CANDIDATE_ID} column; "
                "membership cannot be tracked reliably.",
                file=sys.stderr,
            )
            return 2

    print("\nPer-category counts")
    print("-" * 56)
    total_assigned = 0
    for name, subset in samples.items():
        print(f"  {name:36s} {len(subset):10d}")
        total_assigned += len(subset)
    print(f"  {'SUM OF CATEGORIES':36s} {total_assigned:10d}")

    indices = {
        name: set(subset[CANDIDATE_ID]) if len(subset) else set()
        for name, subset in samples.items()
    }
    covered = set().union(*indices.values()) if indices else set()
    unclassified = set(df[CANDIDATE_ID]) - covered

    print("\nExhaustiveness")
    print("-" * 56)
    print(f"  candidates            : {len(df)}")
    print(f"  in at least one class : {len(covered)}")
    print(f"  UNCLASSIFIED          : {len(unclassified)}")
    if unclassified:
        lost = df[df[CANDIDATE_ID].isin(unclassified)]
        print("\n  D_mcErrors of unclassified candidates:")
        for value, count in lost["D_mcErrors"].value_counts().items():
            print(f"    D_mcErrors = {int(value):6d} : {count:10d}")
        print(
            "\n  NOTE: values above 512 are expected here.  bkg_fakeD covers "
            "0 < D_mcErrors < 512 and bkg_fakeTracks covers exactly 512, so "
            "the fake branch has no catch-all."
        )
    else:
        print("  -> the classification is exhaustive on this sample.")

    print("\nExclusivity")
    print("-" * 56)
    overlaps = [
        (a, b, len(ia & ib))
        for (a, ia), (b, ib) in itertools.combinations(indices.items(), 2)
        if ia & ib
    ]
    if overlaps:
        for a, b, count in sorted(overlaps, key=lambda row: -row[2]):
            print(f"  OVERLAP {a} & {b}: {count}")
    else:
        print("  -> no candidate appears in more than one category.")

    print("\nCategories excluded from the fit templates")
    print("-" * 56)
    excluded = ["bkg_fakeTracks", "bkg_other_TDTl", "bkg_other_signal"]
    excluded_total = sum(len(samples.get(name, [])) for name in excluded)
    for name in excluded:
        count = len(samples.get(name, []))
        fraction = 100.0 * count / len(df) if len(df) else 0.0
        print(f"  {name:36s} {count:10d}  ({fraction:.3f}% of all candidates)")
    fraction = 100.0 * excluded_total / len(df) if len(df) else 0.0
    print(f"  {'TOTAL EXCLUDED':36s} {excluded_total:10d}  ({fraction:.3f}%)")

    return 1 if (unclassified or overlaps) else 0


if __name__ == "__main__":
    sys.exit(main())
