"""Preparation and fast parameter weighting for inclusive BBbar backgrounds.

The decay classification and mode-replacement factors are invariant during a
fit.  :func:`prepare_bbbar_reweighting` computes them once; only the cheap
category lookup in :func:`apply_bbbar_weights` belongs in an optimizer loop.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

import utilities as util


BBBAR_SAMPLE_NAMES = frozenset({"bkg_combinatorial", "bkg_hadronicB_secondaryL"})
PDG_CORRECTIONS = {
    10431 * 411: 0.00105 / 0.00737857,
    10431 * 413: 0.0015 / 0.01768717,
    10431 * 421: 0.00079 / 0.00761729,
    10431 * 423: 0.0009 / 0.01706301,
}


def _prepare_side(
    df: pd.DataFrame,
    *,
    prefix: str,
    combinatorial_vars: list[str],
    neutral_cols: list[str],
    charged_cols: list[str],
    replacement_map: Mapping[int, int],
    replacement_use_pdg_corr: bool,
    cap_nbody: int | None,
    verbose: bool,
) -> None:
    """Add parameter-independent classification columns to ``df`` in place."""
    df[f"{prefix}dmID"] = df[combinatorial_vars].astype("int64").prod(axis=1).abs()
    df[f"{prefix}mask_sl"] = df[combinatorial_vars].isin(util.leptons).any(axis=1)
    neutral_n = (~df[neutral_cols].isin([-1, 22])).sum(axis=1)
    charged_n = (~df[charged_cols].isin([-1, 22])).sum(axis=1)
    df[f"{prefix}n_daughters"] = np.maximum(neutral_n, charged_n).astype(int)
    df[f"{prefix}is_measured"] = df[f"{prefix}dmID"].isin(util.measured_pdg_list)

    n_daughters = df[f"{prefix}n_daughters"].clip(lower=2)
    if cap_nbody is None:
        labels = n_daughters.astype(str) + "-body"
    else:
        labels = np.where(
            n_daughters >= cap_nbody,
            f"{cap_nbody}+-body",
            n_daughters.astype(str) + "-body",
        )
    hadronic = np.where(
        df[f"{prefix}is_measured"],
        "BBbar_measured_hadronic",
        "BBbar_unmeasured:" + labels,
    )
    df[f"{prefix}category"] = np.where(
        df[f"{prefix}mask_sl"], "BBbar_semileptonic", hadronic
    )
    mapped_correction = df[f"{prefix}dmID"].map(PDG_CORRECTIONS).fillna(1.0)
    df[f"{prefix}pdg_corr"] = np.where(
        df[f"{prefix}is_measured"], mapped_correction, 1.0
    )

    replacement_column = f"{prefix}replacement_w"
    df[replacement_column] = 1.0
    count_base = (
        df[f"{prefix}pdg_corr"]
        if replacement_use_pdg_corr
        else pd.Series(1.0, index=df.index)
    )
    new_to_old: dict[int, list[int]] = {}
    for old_dmid, new_dmid in replacement_map.items():
        new_to_old.setdefault(abs(int(new_dmid)), []).append(abs(int(old_dmid)))
    for new_dmid, old_dmids in new_to_old.items():
        old_mask = df[f"{prefix}dmID"].isin(old_dmids)
        new_mask = df[f"{prefix}dmID"] == new_dmid
        n_old = count_base[old_mask].sum()
        n_new = count_base[new_mask].sum()
        df.loc[old_mask, replacement_column] = 0.0
        if n_new <= 0:
            if verbose:
                print(f"[{prefix}] cannot transfer {old_dmids} -> {new_dmid}: n_new=0")
            continue
        factor = (n_old + n_new) / n_new
        df.loc[new_mask, replacement_column] *= factor
        if verbose:
            print(
                f"[{prefix}] replacement {old_dmids} -> {new_dmid}: "
                f"N_old={n_old:.3f}, N_new={n_new:.3f}, R={factor:.3f}"
            )


def prepare_bbbar_reweighting(
    samples: Mapping[str, pd.DataFrame],
    *,
    weight_ell_side: bool = False,
    cap_nbody: int | None = None,
    D_replacement_map: Mapping[int, int] | None = None,
    ell_replacement_map: Mapping[int, int] | None = None,
    replacement_use_pdg_corr: bool = False,
    copy: bool = True,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Classify BBbar events and calculate invariant correction factors once.

    By default the returned dictionary and DataFrames are copies.  Set
    ``copy=False`` to opt into in-place preparation for memory-heavy workflows.
    """
    prepared = {
        name: (df.copy() if copy else df)
        for name, df in samples.items()
    }
    for name, df in prepared.items():
        if name not in BBBAR_SAMPLE_NAMES:
            continue
        _prepare_side(
            df,
            prefix="D_",
            combinatorial_vars=util.combinatorial_vars_D,
            neutral_cols=util.neutral_cols_D,
            charged_cols=util.charged_cols_D,
            replacement_map=D_replacement_map or {},
            replacement_use_pdg_corr=replacement_use_pdg_corr,
            cap_nbody=cap_nbody,
            verbose=verbose,
        )
        if name == "bkg_combinatorial" and weight_ell_side:
            _prepare_side(
                df,
                prefix="ell_",
                combinatorial_vars=util.combinatorial_vars_ell,
                neutral_cols=util.neutral_cols_ell,
                charged_cols=util.charged_cols_ell,
                replacement_map=ell_replacement_map or {},
                replacement_use_pdg_corr=replacement_use_pdg_corr,
                cap_nbody=cap_nbody,
                verbose=verbose,
            )
    return prepared


def apply_bbbar_weights(
    prepared_samples: Mapping[str, pd.DataFrame],
    weight_map: Mapping[str, float],
    *,
    out_weight_col: str = "BB_weight",
    weight_ell_side: bool = False,
    copy: bool = True,
    warn_missing_weight_keys: bool = True,
) -> dict[str, pd.DataFrame]:
    """Apply category weights to already-prepared samples using vectorized maps."""
    weighted = {
        name: (df.copy() if copy else df)
        for name, df in prepared_samples.items()
    }
    warned: set[str] = set()
    for name, df in weighted.items():
        df[out_weight_col] = 1.0
        if name not in BBBAR_SAMPLE_NAMES:
            continue
        prefixes = ["D_"]
        if name == "bkg_combinatorial" and weight_ell_side:
            prefixes.append("ell_")
        for prefix in prefixes:
            category_column = f"{prefix}category"
            if category_column not in df:
                raise ValueError(
                    "BBbar samples have not been prepared; call "
                    "prepare_bbbar_reweighting() first."
                )
            missing = set(pd.unique(df[category_column])) - set(weight_map)
            new_missing = missing - warned
            if new_missing and warn_missing_weight_keys:
                warnings.warn(
                    f"Missing BBbar weights default to 1.0: {sorted(new_missing)}",
                    stacklevel=2,
                )
                warned.update(new_missing)
            manual = df[category_column].map(weight_map).fillna(1.0).astype(float)
            df[f"{prefix}manual_w"] = manual
            side_weight = (
                df[f"{prefix}pdg_corr"] * manual * df[f"{prefix}replacement_w"]
            )
            df[f"{prefix}side_weight"] = side_weight
            df[out_weight_col] *= side_weight
    return weighted
