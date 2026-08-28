#!/usr/bin/env python3
"""Exercise the verdict branches of the BBbar diagnostics.

The scan's verdict is a physics conclusion -- it decides whether to merge
decay families -- and review has repeatedly found ways for the wrong one to be
issued confidently.  These tests drive ``interpret_scan()`` with constructed
rows so every branch is covered without needing the ntuples, and check the
profile and bounds helpers against likelihoods with known behaviour.

    python3 scripts/tests/test_scan_verdicts.py
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.util
import io
import pathlib
import sys
import types

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def load(name: str, stub_analysis_deps: bool = False):
    """Import one of the scripts by path, without running it."""
    if stub_analysis_deps:
        # scan_bbbar_bounds imports the analysis stack at module level; none of
        # it is reached by the pure functions under test here.
        for dependency in ("pandas", "uproot", "utilities", "bbbar_reweighting"):
            sys.modules.setdefault(dependency, types.ModuleType(dependency))
        sys.modules["utilities"].all_relevant_variables = []
    spec = importlib.util.spec_from_file_location(
        f"_test_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan = load("scan_bbbar_bounds", stub_analysis_deps=True)
eigen = load("bbbar_eigen_systematics")

BOUNDS = {"measured": (0.5, 5.0), "unmeasured_2body": (0.001, 5.0),
          "unmeasured_3body": (0.01, 1.0), "unmeasured_4body": (0.001, 1.0),
          "unmeasured_5plus": (0.001, 0.5)}

FAILURES: list[str] = []


def check(label: str, got, expected) -> None:
    ok = got == expected
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(f"{label}: expected {expected!r}, got {got!r}")


def row(scale, deviance, pinned=True, profile=None, valid=True, has_cov=True):
    """One scanned scale, in the tuple layout interpret_scan() consumes."""
    return (scale, deviance, not pinned, 0.9, {"unmeasured_2body": 0.001},
            has_cov, valid, profile)


def verdict(rows, tolerance=1.0, skip_profile=False) -> str:
    """Return the leading words of the verdict interpret_scan() prints."""
    args = argparse.Namespace(deviance_tolerance=tolerance,
                              skip_profile=skip_profile)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        scan.interpret_scan(rows, args, composite=False)
    line = next(l for l in buffer.getvalue().splitlines() if "VERDICT" in l)
    return line.split("VERDICT: ", 1)[1].split(".")[0].strip()


FLAT = {"unmeasured_2body": [0.0001, 0.0002]}


def test_verdicts() -> None:
    print("interpret_scan")

    # A repeated scale gives two converged rows whose deviance span is zero
    # without any range of bounds having been covered.
    check("duplicate scales are not a range of bounds",
          verdict([row(1.0, 100.0, profile=FLAT), row(1.0, 100.0, profile=FLAT)]),
          "inconclusive")

    # A constrained refit cannot beat the unconstrained fit; if it does, the
    # nominal fit was not the minimum and nothing can be read off it.
    check("a profile below the nominal minimum is not flatness",
          verdict([row(1.0, 100.0, profile={"unmeasured_2body": [-4.2, 0.01]}),
                   row(10.0, 99.9, profile=FLAT)]),
          "inconclusive -- the nominal fit is unreliable")
    check("a small negative delta stays within tolerance",
          verdict([row(1.0, 100.0, profile={"unmeasured_2body": [-0.3, 0.01]}),
                   row(10.0, 99.9, profile=FLAT)]),
          "flat likelihood direction")

    check("two distinct scales with flat profiles are flat",
          verdict([row(1.0, 100.0, profile=FLAT), row(10.0, 99.9, profile=FLAT)]),
          "flat likelihood direction")
    check("a rise at the far profile step alone disproves flatness",
          verdict([row(1.0, 100.0, profile={"unmeasured_2body": [0.05, 2.0]}),
                   row(10.0, 99.9, profile=FLAT)]),
          "boundary-constrained optimum, NOT a flat direction")
    check("a rise at an earlier scale alone disproves flatness",
          verdict([row(1.0, 100.0, profile={"unmeasured_2body": [3.0, 3.1]}),
                   row(10.0, 99.9, profile=FLAT)]),
          "boundary-constrained optimum, NOT a flat direction")
    check("a failed refit leaves flatness untested",
          verdict([row(1.0, 100.0, profile={"unmeasured_2body": None}),
                   row(10.0, 99.9, profile=FLAT)]),
          "inconclusive")
    check("an interior minimum ends the scan",
          verdict([row(1.0, 100.0, profile=FLAT), row(10.0, 99.9, pinned=False)]),
          "bounds were the binding constraint")
    check("a still-improving deviance is inconclusive",
          verdict([row(1.0, 100.0, profile=FLAT), row(10.0, 80.0, profile=FLAT)]),
          "inconclusive")
    check("one valid scale is no range of bounds",
          verdict([row(1.0, 100.0, profile=FLAT), row(10.0, 99.0, valid=False)]),
          "inconclusive")
    check("without a profile flatness cannot be claimed",
          verdict([row(1.0, 100.0), row(10.0, 99.9)], skip_profile=True),
          "inconclusive without a profile")
    check("nothing converged",
          verdict([row(1.0, 100.0, valid=False)]),
          "no valid fit at any scale")


def test_profile() -> None:
    """The profile must separate degeneracy from a boundary-constrained fit."""
    print("profile_pinned_inward")
    Spec = collections.namedtuple("Spec", "category lower upper")
    specs = {"a": Spec("A", 0.001, 5.0), "b": Spec("B", 0.001, 5.0)}

    def degenerate(a, b):        # only the sum is constrained
        return (a + b - 1.0) ** 2

    def boundary(a, b):          # 'a' prefers -1, outside the range
        return (a + 1.0) ** 2 + (b - 1.0) ** 2

    degenerate.errordef = boundary.errordef = 1.0

    rises = scan.profile_pinned_inward(degenerate, list(specs), specs,
                                       {"a": 0.001, "b": 0.999}, ["a"],
                                       [0.02, 0.10])["a"]
    # Not exactly zero: what is left is MIGRAD's convergence residual, orders
    # of magnitude below any usable --deviance-tolerance.
    check("a degenerate direction profiles flat", max(rises) < 1e-3, True)

    rises = scan.profile_pinned_inward(boundary, list(specs), specs,
                                       {"a": 0.001, "b": 1.0}, ["a"],
                                       [0.02, 0.10])["a"]
    # This is the case that motivates keeping every sampled point: the 2% step
    # rises by less than the default tolerance, the 10% step by more.
    check("a boundary-constrained direction rises", max(rises) > 1.0, True)
    check("its first step alone would look flat", min(rises) < 1.0, True)
    check("every sampled point is kept", len(rises), 2)


def test_combined_bounds() -> None:
    """Independent nuisance parameters vary together downstream."""
    print("check_combined_bounds")
    order = list(BOUNDS)
    central = np.array([1.0, 1.0, 0.30, 0.5, 0.2])
    shift = np.array([0.0, 0.0, 0.17, 0.0, 0.0])

    check("one direction alone stays in range",
          eigen.check_variation_bounds(order, central + shift, central - shift,
                                       BOUNDS),
          [])
    check("a single direction is not double-checked",
          eigen.check_combined_bounds(order, central, [shift], BOUNDS),
          [])
    combined = eigen.check_combined_bounds(order, central, [shift, shift], BOUNDS)
    check("two such directions together leave it", len(combined), 1)
    check("the crossing weight is named",
          "unmeasured_3body" in combined[0] if combined else False, True)
    check("the sign pattern reaching it is reported",
          "np1=-1, np2=-1" in combined[0] if combined else False, True)


def main() -> int:
    test_verdicts()
    test_profile()
    test_combined_bounds()
    print()
    if FAILURES:
        for failure in FAILURES:
            print(f"FAILED: {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
