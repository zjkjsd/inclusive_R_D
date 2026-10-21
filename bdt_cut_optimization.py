# -*- coding: utf-8 -*-
"""BDT selection optimisation with a binned figure of merit.

Chooses the four thresholds of the multiclass-BDT cut

    sig_prob > t_sig  and  fakeD_prob < t_fakeD  and
    continuum_prob < t_cont  and  combinatorial_prob < t_comb

by coordinate descent on a binned FOM computed in the 2D signal-region fit
binning (util.MM2_bins_sr x util.p_D_l_bins_sr). Per bin, with s^, b^ the MC
signal and background counts and n = s^ + b^:

    Z^2 = sum_channels sum_bins  s^ (s^ - 1) / (n - 1)    (estimator='unbiased', default)
    Z^2 = sum_channels sum_bins  s^2 / n                  (estimator='plain')

Both estimate the usual binned s^2 / (s + b); the default one does so without
the MC-statistics excess of the plain one (see the two sections below). With
eps_bkg > 0 the denominators become n - 1 + (eps b^)^2 and n + (eps b^)^2.

* s = B -> D tau nu (FOM_SIGNAL); b = the non-physics backgrounds the BDT is
  trained to suppress (FOM_BACKGROUND). Physics backgrounds (D(*,**) l/tau nu)
  enter neither s nor b.
* The FOM is filled after the full selection chain: MVA cut -> BCS (minimum
  B_D_ReChi2) -> D_M signal window. The D_M window is applied after BCS, as in
  the fit, so BCS competes candidates from the whole D_M range.
* e and mu are treated as independent channels of one fit; their Z^2 add.
* Only bins with at least min_bin_count (default 10) unweighted MC candidates
  of signal + background enter the sum (count_on='fom'; 'all' counts every
  category, like the template threshold in create_templates_new).
* All yields are raw MC (generic MC = 4 x data luminosity). Z is therefore in
  MC units; the optimum is unaffected. An eps_bkg set by hand acts on MC-unit
  yields: (eps*b_MC)^2 = (2*eps * b_data)^2 * 4, i.e. it corresponds to a
  fractional background uncertainty of 2*eps at data luminosity.
* Thresholds come from grids rounded to GRID_DECIMALS (3) decimals, so the
  working points are round numbers.

MC-statistics excess of the plain estimator
-------------------------------------------
In one bin let s, b be the EXPECTED numbers of MC signal and background
events, and s^, b^ the numbers actually found in the MC sample (independent
Poisson). The plain FOM uses f = s^2 / (s^ + b^); the true value is s^2 / (s + b).

Derivation. n = s^ + b^ is Poisson with mean mu = s + b. Given n, each of the n
events is signal with probability p = s / mu, so s^ | n is Binomial(n, p):
    E[s^ | n] = n p,   Var[s^ | n] = n p (1 - p)
    E[s^2 | n] = Var + mean^2 = n p (1 - p) + n^2 p^2
    E[s^2 / n | n] = p (1 - p) + n p^2
Averaging over n (law of total expectation, E[f] = sum_n P(n) E[f | n]; a bin
with n = 0 contributes 0 and has probability exp(-mu), and sum_n n P(n) = mu):
    E[f] = p (1 - p) (1 - exp(-mu)) + p^2 mu
and p^2 mu = s^2 / (s + b) is the true value. Hence, exactly,
    excess = E[f] - true = p (1 - p) (1 - exp(-(s + b)))
    excess / true        = (1 - p) / s * (1 - exp(-(s + b)))
The n p^2 part is unbiased; the excess is the binomial variance n p (1 - p)
of the signal count, divided by n.

Consequences
* The excess depends on the purity p, not on the size of the bin: for
  s + b >~ 5 it is p (1 - p), between 0 (pure signal or pure background) and
  its maximum 0.25 at p = 0.5. For example 0.25 for s = b, 0.09 for p = 0.1
  or 0.9, 0.14 for s = 5, b = 1.
* Relative to the bin's true contribution it is (1 - p) / s: negligible in
  bins with many signal events, large in bins with few.
* Summed over bins it adds roughly sum_i p_i (1 - p_i) to Z^2. That would be
  harmless if it were the same for every cut, but it is not: a looser cut
  populates more sparse bins, each adding its own excess with almost no true
  sensitivity behind it, so the plain FOM is pushed toward looser cuts by MC
  statistics alone.
* The threshold n >= 10 removes the sparse bins, where the excess is of the
  same size as the whole contribution, but it does not make the excess zero:
  a kept bin still carries p (1 - p) <= 0.25. That is small (<= 5 %) where s
  is large and large only in background-dominated bins with s ~ 1, whose
  contribution is small anyway. At most (S + B) / 10 bins pass, so the total
  excess is at most 0.025 (S + B).

Target function without the excess (estimator='unbiased', default)
------------------------------------------------------------------
Replace s^2 / n by s^ (s^ - 1) / (n - 1) in every bin.

Derivation. For s^ | n ~ Binomial(n, p) the second factorial moment is
    E[s^ (s^ - 1) | n] = E[s^2 | n] - E[s^ | n]
                       = n p (1 - p) + n^2 p^2 - n p = n (n - 1) p^2,
so, for every n >= 2,
    E[ s^ (s^ - 1) / (n - 1) | n ] = n p^2 .
This is exactly the unbiased part of s^2 / n (E[s^2 / n | n] = p (1 - p) + n p^2)
without the p (1 - p) term. Averaging over n as above gives p^2 mu = s^2/(s + b),
the true value, up to the n < 2 bins, which the n >= 10 threshold excludes anyway.

Why it is better
* No excess at any purity and any bin size: the quantity the optimiser sees
  is, on average, the true binned FOM, so MC statistics cannot pull the
  optimum toward looser (or tighter) cuts.
* It is the plain term minus the estimated excess, bin by bin:
      s^2 / n - s^ (s^ - 1) / (n - 1) = p^ (1 - p^) n / (n - 1),   p^ = s^ / n,
  i.e. the same FOM with the p(1 - p) term removed, not a different measure
  of sensitivity. For large n the two agree; they differ only where the
  excess matters.
* The result holds for each n separately, so selecting bins by their count
  (the n >= 10 threshold) keeps it exact. The threshold is still applied:
  the fit drops sparse bins too, and sparse bins add only noise.
* Same cost and same inputs as the plain estimator.
Limits: exact for unit weights (raw MC counts, enforced) and eps_bkg = 0;
with eps_bkg > 0 the denominator n - 1 + (eps b^)^2 is an approximation.
Checked by exact summation over Poisson probabilities in
scripts/tests/test_bdt_cut_optimization.py.

Diagnostics and alternatives
* evaluate_cut reports excess_frac = sum_kept p^ (1 - p^) / Z^2: the share of
  Z^2 that the plain estimator would add as excess (with the default
  estimator, roughly the share that was removed). Compare the two estimators
  with estimator='plain'.
* Why not a threshold on the bin purity instead of the count: the purity is
  estimated from s^ itself, so keeping only high-p^ bins keeps bins whose
  signal count fluctuated up, a new upward bias; the count threshold selects
  on n only, and given n the signal fraction is unaffected. Low-purity bins
  also carry the smallest absolute excess (p(1-p) -> 0), and the fit uses
  them, so the FOM should too.

Typical use (see the driver notebook):

    df_e  = bco.load_candidates(files_e,  'e')
    df_mu = bco.load_candidates(files_mu, 'mu')
    data  = bco.FOMInput({'e': df_e, 'mu': df_mu})
    res   = bco.run_optimization(data)
    wps   = bco.choose_working_points(data, res['best_theta'])
    bco.validation_report({'new tight': wps['tight'], 'old tight': util.lgb_tight},
                          {'run2 A': (data, 'A'), 'run2 B': (data, 'B')})

All inputs are generic MC that was NOT used in the BDT training.
"""
import contextlib
import io
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utilities as util


############################## configuration ##################################

FOM_SIGNAL = [r'$D\tau\nu$']
FOM_BACKGROUND = ['bkg_fakeD', 'bkg_continuum', 'bkg_combinatorial',
                  'bkg_hadronicB_secondaryL', 'bkg_fakeL', 'bkg_fakeTracks']
# Normalisation mode: only used for the efficiency-ratio guardrail.
NORM_COMPONENT = r'$D\ell\nu$'

FOM_BINS = [util.MM2_bins_sr, util.p_D_l_bins_sr]
FIT_VARIABLES = util.FIT_VARIABLES          # ['B0_recMissM2', 'p_D_l']
MVA_OUTPUTS = util.MVA_OUTPUTS              # ['sig_prob', 'fakeD_prob', ...]
D_MASS_SR_RANGE = (1.855, 1.885)            # same window as util.D_MASS_SR
BCS_RANK_VAR = 'B_D_ReChi2'                 # BCS keeps the minimum, as bcs='vtx'
EVENT_ID_COLUMNS = util.EVENT_ID_COLUMNS    # after apply_mva_bcs renamed them

CUT_DIRECTIONS = {'sig_prob': '>', 'fakeD_prob': '<',
                  'continuum_prob': '<', 'combinatorial_prob': '<'}
# A threshold at or beyond these values means "no cut on this output".
OPEN_CUT = {'sig_prob': 0.0, 'fakeD_prob': 1.0,
            'continuum_prob': 1.0, 'combinatorial_prob': 1.0}

# Columns kept per candidate after loading (everything else is dropped).
_KEEP_COLUMNS = MVA_OUTPUTS + FIT_VARIABLES + ['D_M', BCS_RANK_VAR] + EVENT_ID_COLUMNS


############################## cut <-> theta ##################################

def is_open(var, t):
    """True if threshold t on var removes nothing."""
    return t <= 0.0 if CUT_DIRECTIONS[var] == '>' else t >= 1.0


def cut_to_theta(cut):
    """Parse a flat 'and' cut string such as util.lgb_tight into a theta dict.

    Outputs that the string does not mention are left open. Raises if a
    condition points the wrong way (e.g. 'fakeD_prob>0.1').
    """
    theta = dict(OPEN_CUT)
    if cut is None:
        return theta
    for cond in util.split_cut_conditions(cut):
        m = re.fullmatch(r'\s*(\w+)\s*(<=|>=|<|>)\s*([-+0-9.eE]+)\s*', cond)
        if m is None or m.group(1) not in CUT_DIRECTIONS:
            raise ValueError(f'Cannot parse "{cond}" as a cut on one of {MVA_OUTPUTS}.')
        var, op, value = m.group(1), m.group(2), float(m.group(3))
        if op[0] != CUT_DIRECTIONS[var]:
            raise ValueError(f'"{cond}": expected {var}{CUT_DIRECTIONS[var]}value.')
        theta[var] = value
    return theta


def theta_to_cut(theta, digits=4):
    """Inverse of cut_to_theta; open conditions are omitted ('' if all open)."""
    parts = [f'{v}{CUT_DIRECTIONS[v]}{theta[v]:.{digits}g}'
             for v in MVA_OUTPUTS if not is_open(v, theta[v])]
    return ' and '.join(parts)


def as_theta(cut_or_theta):
    """Accept either a cut string or a theta dict."""
    if isinstance(cut_or_theta, dict):
        return {v: float(cut_or_theta.get(v, OPEN_CUT[v])) for v in MVA_OUTPUTS}
    return cut_to_theta(cut_or_theta)


############################## loading ########################################

def classify_to_codes(df, mode):
    """Category name per candidate, using util.classify_mc_dict as the source
    of truth. df must have a unique index. Candidates the classification does
    not cover (see the D_mcErrors > 512 note in utilities) get 'unclassified'.
    """
    if not df.index.is_unique:
        raise ValueError('classify_to_codes needs a unique index (reset_index first).')
    with contextlib.redirect_stdout(io.StringIO()):
        samples = util.classify_mc_dict(df, mode, template=False)
    category = pd.Series('unclassified', index=df.index, dtype=object)
    for name, sub in samples.items():
        category.loc[sub.index] = name
    return category


def load_candidates(files, mode, tree=None, extra_cut=None, step_size='1 GB',
                    weight_col=None, verbose=True):
    """Load candidate-level generic MC (all candidates, no MVA cut, no BCS).

    Reads the reconstruction ntuples chunk by chunk, applies util.offline_cut
    (+ extra_cut, e.g. util.Dst_veto_cut), adds the derived variables, attaches
    the BDT outputs with util.apply_mva_bcs(bcs=None) and the truth category,
    and keeps only the few columns the FOM needs (float32), so a full run of
    generic MC fits in memory.

    files : list of ROOT file paths. tree defaults to 'B0_<mode>'.
    """
    tree = tree or f'B0_{mode}'
    cut = util.offline_cut if extra_cut is None else f'({util.offline_cut}) & ({extra_cut})'
    keep = _KEEP_COLUMNS + ([weight_col] if weight_col else [])
    chunks = []
    import uproot
    for i, batch in enumerate(uproot.iterate(
            [f'{f}:{tree}' for f in files], cut=cut, library='pd', step_size=step_size,
            filter_name=lambda name: name in util.all_relevant_variables)):
        batch = batch.reset_index(drop=True)
        util.add_derived_variables(batch)
        with contextlib.redirect_stdout(io.StringIO()):
            # cut='ell_p>0' is always true: predictions only, no MVA cut, no BCS
            pred = util.apply_mva_bcs(batch, util.training_variables, cut='ell_p>0',
                                      library='lgbm', model='multiclass', bcs=None)
        pred = pred.reset_index(drop=True)
        category = classify_to_codes(pred, mode)
        out = pred[keep].copy()
        for col in MVA_OUTPUTS + FIT_VARIABLES + ['D_M', BCS_RANK_VAR]:
            out[col] = out[col].astype(np.float32)
        out['category'] = pd.Categorical(category.to_numpy())
        chunks.append(out)
        if verbose:
            print(f'[{mode}] chunk {i}: {len(out):,d} candidates')
    df = pd.concat(chunks, ignore_index=True)
    df['category'] = df['category'].astype('category')
    return df


############################## FOM input ######################################

class _Channel:
    """Compact, BCS-sorted arrays for one lepton channel."""

    def __init__(self, df, bins, weight_col, half_col):
        rank = df[BCS_RANK_VAR].to_numpy(dtype=float)
        # Sort by event, then rank. NaN ranks go last and are never selected,
        # as in util.add_bcs_flag. The stable sort keeps the input order among
        # equal ranks, which is how groupby().idxmin() breaks ties.
        ev = df[EVENT_ID_COLUMNS].to_numpy()
        keys = [np.where(np.isnan(rank), np.inf, rank)] + [ev[:, j] for j in range(ev.shape[1] - 1, -1, -1)]
        order = np.lexsort(keys)
        ev = ev[order]
        new_group = np.ones(len(order), dtype=bool)
        new_group[1:] = np.any(ev[1:] != ev[:-1], axis=1)
        self.group = np.cumsum(new_group) - 1
        self.n_groups = int(self.group[-1] + 1) if len(order) else 0
        self.eligible = ~np.isnan(rank[order])

        self.probs = {v: df[v].to_numpy(dtype=np.float32)[order] for v in MVA_OUTPUTS}

        x = df[FIT_VARIABLES[0]].to_numpy(dtype=float)[order]
        y = df[FIT_VARIABLES[1]].to_numpy(dtype=float)[order]
        dm = df['D_M'].to_numpy(dtype=float)[order]
        ix = np.searchsorted(bins[0], x, side='right') - 1
        iy = np.searchsorted(bins[1], y, side='right') - 1
        # np.histogram2d convention: the last bin is closed on the right
        ix[x == bins[0][-1]] = len(bins[0]) - 2
        iy[y == bins[1][-1]] = len(bins[1]) - 2
        nx, ny = len(bins[0]) - 1, len(bins[1]) - 1
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        in_sr = (D_MASS_SR_RANGE[0] < dm) & (dm < D_MASS_SR_RANGE[1])
        self.in_sr = in_sr
        self.binidx = np.where(inside & in_sr, ix * ny + iy, -1).astype(np.int64)
        self.n_bins = nx * ny

        cat = df['category'].astype(str).to_numpy()[order]
        self.cat_names = sorted(set(cat))
        lookup = {n: i for i, n in enumerate(self.cat_names)}
        self.cat = np.array([lookup[c] for c in cat], dtype=np.int16) if len(cat) else np.zeros(0, np.int16)
        self.cls = np.zeros(len(order), dtype=np.int8)
        self.cls[np.isin(cat, FOM_SIGNAL)] = 1
        self.cls[np.isin(cat, FOM_BACKGROUND)] = 2

        w = df[weight_col].to_numpy(dtype=float)[order] if weight_col else np.ones(len(order))
        self.w = w
        self.unit_weights = bool(np.all(w == 1.0))

        if half_col == 'event':
            self.half = (df['event'].to_numpy()[order] % 2).astype(np.int8)
        else:
            self.half = df[half_col].to_numpy()[order].astype(np.int8)

    def __len__(self):
        return len(self.group)


class FOMInput:
    """All candidates of all channels, prepared once for fast FOM evaluation.

    dfs        : {'e': df_e, 'mu': df_mu} from load_candidates (or any frame
                 with the same columns and a 'category' column).
    weight_col : optional per-candidate weight column (None = 1, raw MC).
    half_col   : 'event' splits by event-number parity (half A = even, B = odd);
                 or the name of a 0/1 column.
    """

    def __init__(self, dfs, bins=FOM_BINS, weight_col=None, half_col='event'):
        self.bins = bins
        self.channels = {name: _Channel(df, bins, weight_col, half_col)
                         for name, df in dfs.items()}
        self._bcs_flag_cache = {}
        self._ref_cache = {}

    def describe(self):
        rows = []
        for name, ch in self.channels.items():
            for half in (0, 1):
                m = ch.half == half
                rows.append(dict(channel=name, half='AB'[half], candidates=int(m.sum()),
                                 events=len(np.unique(ch.group[m])),
                                 signal_cands=int((m & (ch.cls == 1)).sum()),
                                 bkg_cands=int((m & (ch.cls == 2)).sum())))
        return pd.DataFrame(rows)


############################## selection + FOM ################################

def _cond(ch, var, t):
    if is_open(var, t):
        return None
    p = ch.probs[var]
    return p > t if CUT_DIRECTIONS[var] == '>' else p < t


def cut_mask(ch, theta, skip=None):
    """Boolean mask of candidates passing theta (all outputs except `skip`)."""
    mask = np.ones(len(ch), dtype=bool)
    for var in MVA_OUTPUTS:
        if var == skip:
            continue
        c = _cond(ch, var, theta[var])
        if c is not None:
            mask &= c
    return mask


def _first_per_group(ch, mask):
    """Indices of the BCS candidate among those passing mask (arrays are
    sorted by event then rank, so it is the first passing one per event)."""
    idx = np.flatnonzero(mask & ch.eligible)
    if len(idx) == 0:
        return idx
    g = ch.group[idx]
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = g[1:] != g[:-1]
    return idx[keep]


def _bcs_flag(data, name, base_theta):
    """BCS flag computed once after a fixed base cut (for bcs_then_cut)."""
    key = (name, tuple(base_theta[v] for v in MVA_OUTPUTS))
    if key not in data._bcs_flag_cache:
        ch = data.channels[name]
        flag = np.zeros(len(ch), dtype=bool)
        flag[_first_per_group(ch, cut_mask(ch, base_theta))] = True
        data._bcs_flag_cache[key] = flag
    return data._bcs_flag_cache[key]


def _half_mask(ch, half):
    if half in (None, 'all'):
        return None
    return ch.half == {'A': 0, 'B': 1}[half]


def select(data, name, theta, half='A', bcs_order='cut_then_bcs', bcs_base=None, mask=None):
    """Indices of the candidates kept by MVA cut + BCS in one channel.

    bcs_order:
      'cut_then_bcs' : MVA cut, then BCS among the survivors (nominal).
      'bcs_then_cut' : BCS after the fixed cut bcs_base (None = no MVA cut,
                       util.lgb_loose = the current ntuple chain), then theta.
    mask: optional precomputed cut_mask(ch, theta) to save time in scans.
    """
    ch = data.channels[name]
    m = cut_mask(ch, theta) if mask is None else mask
    h = _half_mask(ch, half)
    if h is not None:
        m = m & h
    if bcs_order == 'cut_then_bcs':
        return _first_per_group(ch, m)
    if bcs_order == 'bcs_then_cut':
        return np.flatnonzero(m & _bcs_flag(data, name, as_theta(bcs_base)))
    raise ValueError(f'Unknown bcs_order {bcs_order!r}.')


# Bins with fewer MC candidates than this are left out of Z^2. In such bins the
# MC-statistics excess p(1-p)(1-exp(-(s+b))) is as large as the bin's true
# contribution s^2/(s+b); see "MC-statistics excess" in the module docstring.
MIN_BIN_COUNT = 10
COUNT_ON = ('fom', 'all')
ESTIMATORS = ('unbiased', 'plain')    # first = default


def _channel_result(ch, sel, eps_bkg, min_bin_count, count_on, estimator='unbiased', weights=None):
    """Z^2, S, B, the number of summed bins and the estimated excess for one channel.

    estimator: 'unbiased' (default) sums s(s-1)/(s+b-1+(eps b)^2), which has no
    MC-statistics excess; 'plain' sums s^2/(s+b+(eps b)^2) (module docstring).

    weights: optional extra per-selected-candidate weights (bootstrap); they
    also enter the bin counts used for the threshold.
    """
    w = ch.w[sel] if weights is None else ch.w[sel] * weights
    bi = ch.binidx[sel]
    cls = ch.cls[sel]
    in_fit = bi >= 0
    sig = in_fit & (cls == 1)
    bkg = in_fit & (cls == 2)
    s = np.bincount(bi[sig], weights=w[sig], minlength=ch.n_bins)
    b = np.bincount(bi[bkg], weights=w[bkg], minlength=ch.n_bins)
    if count_on not in COUNT_ON:
        raise ValueError(f'count_on must be one of {COUNT_ON}.')
    counted = (sig | bkg) if count_on == 'fom' else in_fit
    unit = np.ones(len(sel)) if weights is None else np.asarray(weights, dtype=float)
    n = np.bincount(bi[counted], weights=unit[counted], minlength=ch.n_bins)
    if estimator not in ESTIMATORS:
        raise ValueError(f'estimator must be one of {ESTIMATORS}.')
    if estimator == 'unbiased' and not ch.unit_weights:
        raise ValueError("estimator='unbiased' needs unit weights (raw MC counts).")
    denom = s + b + (eps_bkg * b) ** 2
    keep = (n >= min_bin_count) & (denom > 0)
    if estimator == 'plain':
        z2 = float(np.sum(s[keep] ** 2 / denom[keep]))
    else:
        # s(s-1)/(n-1): bins with s + b < 2 contribute 0 (s(s-1) = 0 there anyway)
        keep &= (s + b) >= 2
        z2 = float(np.sum(s[keep] * (s[keep] - 1.0) / (denom[keep] - 1.0)))
    # Estimated MC-statistics excess of the PLAIN estimator (module docstring):
    # p(1-p) per kept bin, with p estimated by s/(s+b). Diagnostic only; with
    # estimator='unbiased' it is the amount that was removed.
    sb = s[keep] + b[keep]
    p_hat = np.divide(s[keep], sb, out=np.zeros_like(sb), where=sb > 0)
    excess = float(np.sum(p_hat * (1.0 - p_hat)))
    return z2, s.sum(), b.sum(), int(keep.sum()), excess


def evaluate_cut(data, theta, half='A', bcs_order='cut_then_bcs', bcs_base=None,
                 eps_bkg=0.0, min_bin_count=MIN_BIN_COUNT, count_on='fom',
                 estimator='unbiased', yields=False, _masks=None):
    """Binned FOM of one working point.

    Returns a dict with Z, Z_<channel>, n_bins (bins passing the threshold),
    S, B (summed over all fit bins of the D_M signal window), excess_frac
    (estimated MC-statistics excess / Z^2, see the module docstring) and, if
    yields=True, the SR yield of every truth category ('yield_<category>',
    summed over channels).
    """
    theta = as_theta(theta)
    out = dict(theta)
    z2_tot, s_tot, b_tot, n_bins, excess_tot = 0.0, 0.0, 0.0, 0, 0.0
    cat_yields = {}
    for name, ch in data.channels.items():
        sel = select(data, name, theta, half, bcs_order, bcs_base,
                     mask=None if _masks is None else _masks[name])
        z2, s, b, nb, exc = _channel_result(ch, sel, eps_bkg, min_bin_count, count_on, estimator)
        out[f'Z_{name}'] = np.sqrt(z2)
        z2_tot += z2
        n_bins += nb
        excess_tot += exc
        s_tot += s
        b_tot += b
        if yields:
            sr = ch.in_sr[sel]
            y = np.bincount(ch.cat[sel][sr], weights=ch.w[sel][sr], minlength=len(ch.cat_names))
            for i, cname in enumerate(ch.cat_names):
                cat_yields[cname] = cat_yields.get(cname, 0.0) + y[i]
    out.update(Z=np.sqrt(z2_tot), n_bins=n_bins, S=s_tot, B=b_tot,
               excess_frac=excess_tot / z2_tot if z2_tot > 0 else np.nan)
    if yields:
        out.update({f'yield_{k}': v for k, v in cat_yields.items()})
    return out


def efficiencies(data, theta, half='A', bcs_order='cut_then_bcs', bcs_base=None):
    """Per-category SR efficiency of theta relative to no MVA cut (same BCS
    procedure, same half), plus the eps(D tau nu)/eps(D l nu) ratio."""
    key = (half, bcs_order, None if bcs_base is None else theta_to_cut(as_theta(bcs_base)))
    if key not in data._ref_cache:
        data._ref_cache[key] = evaluate_cut(data, OPEN_CUT, half, bcs_order, bcs_base, yields=True)
    ref = data._ref_cache[key]
    cur = evaluate_cut(data, theta, half, bcs_order, bcs_base, yields=True)
    eff = {k.replace('yield_', 'eff_'): cur[k] / ref[k] if ref.get(k) else np.nan
           for k in cur if k.startswith('yield_')}
    sig, norm = f'eff_{FOM_SIGNAL[0]}', f'eff_{NORM_COMPONENT}'
    if sig in eff and norm in eff:
        eff['eff_ratio_sig_norm'] = eff[sig] / eff[norm]
    return eff


############################## scans ##########################################

GRID_DECIMALS = 3


def make_grid(data, var, n=35, eff_min=0.10, half='A', decimals=GRID_DECIMALS):
    """Thresholds at quantiles of var for candidate-level D tau nu in the SR.

    The grid is uniform in single-cut signal efficiency, from 1 (open) down to
    eff_min, so every step removes a similar amount of signal. Thresholds are
    rounded to `decimals` (points that coincide after rounding are merged), so
    every working point found by the descent is a round number.
    """
    vals = []
    for ch in data.channels.values():
        m = (ch.cls == 1) & ch.in_sr
        h = _half_mask(ch, half)
        vals.append(ch.probs[var][m if h is None else m & h])
    vals = np.concatenate(vals)
    effs = np.linspace(1.0, eff_min, n)
    q = 1.0 - effs if CUT_DIRECTIONS[var] == '>' else effs
    grid = np.quantile(vals, q)
    grid = np.append(grid, OPEN_CUT[var])
    return np.unique(np.round(grid, decimals))


def make_grids(data, n=35, eff_min=0.10, half='A', decimals=GRID_DECIMALS):
    return {v: make_grid(data, v, n, eff_min, half, decimals) for v in MVA_OUTPUTS}


def scan_1d(data, var, theta, grid, **eval_kw):
    """Z (and S, B) as a function of one threshold, the others fixed."""
    theta = as_theta(theta)
    base = {name: cut_mask(ch, theta, skip=var) for name, ch in data.channels.items()}
    rows = []
    for t in grid:
        th = dict(theta, **{var: float(t)})
        masks = {}
        for name, ch in data.channels.items():
            c = _cond(ch, var, t)
            masks[name] = base[name] if c is None else base[name] & c
        r = evaluate_cut(data, th, _masks=masks, **eval_kw)
        r['scan_var'] = var
        r['threshold'] = float(t)
        rows.append(r)
    return pd.DataFrame(rows)


def _looseness(var, t):
    """Larger = looser, for tie-breaking."""
    return -t if CUT_DIRECTIONS[var] == '>' else t


def _best_row(scan, var, rel_tie=1e-6):
    """Highest Z; among (numerically) equal Z the loosest threshold."""
    zmax = scan['Z'].max()
    tied = scan[scan['Z'] >= zmax * (1 - rel_tie)]
    return tied.loc[tied['threshold'].map(lambda t: _looseness(var, t)).idxmax()]


############################## coordinate descent #############################

def coordinate_descent(data, theta0, grids, order=None, max_sweeps=8, rel_tol=1e-3,
                       label='', verbose=True, **eval_kw):
    """Cycle 1D scans over the four outputs until Z stops improving.

    Stops when a full sweep changes no threshold, or improves Z by less than
    rel_tol (relative). Returns (theta, path DataFrame).
    """
    order = order or MVA_OUTPUTS
    theta = as_theta(theta0)
    start = evaluate_cut(data, theta, **eval_kw)
    path = [dict(start, start=label, sweep=0, step='start')]
    z = start['Z']
    if verbose:
        print(f'[{label}] start Z={z:.4f}  {theta_to_cut(theta) or "(no cut)"}')
    for sweep in range(1, max_sweeps + 1):
        z_sweep_start, moved = z, False
        for var in order:
            grid = np.unique(np.append(grids[var], theta[var]))
            best = _best_row(scan_1d(data, var, theta, grid, **eval_kw), var)
            if best['threshold'] != theta[var]:
                moved = True
            theta[var] = float(best['threshold'])
            z = best['Z']
            path.append(dict(best.drop(['scan_var', 'threshold']).to_dict(),
                             start=label, sweep=sweep, step=var))
        if verbose:
            print(f'[{label}] sweep {sweep}: Z={z:.4f}  {theta_to_cut(theta) or "(no cut)"}')
        if not moved or (z - z_sweep_start) < rel_tol * max(z_sweep_start, 1e-12):
            break
    return theta, pd.DataFrame(path)


def _neighbours(theta, grids):
    """theta with each coordinate moved by -1, 0, +1 grid steps (3^4 - 1)."""
    idx = {}
    for v in MVA_OUTPUTS:
        g = np.unique(np.append(grids[v], theta[v]))
        idx[v] = (g, int(np.searchsorted(g, theta[v])))
    for steps in np.ndindex(*(3,) * len(MVA_OUTPUTS)):
        if all(s == 1 for s in steps):
            continue
        th = {}
        for v, s in zip(MVA_OUTPUTS, steps):
            g, i = idx[v]
            th[v] = float(g[min(max(i + s - 1, 0), len(g) - 1)])
        yield th


def local_grid_check(data, theta, grids, **eval_kw):
    """Evaluate the 80 neighbours of theta; catches improvements along
    diagonals, which coordinate descent cannot see."""
    rows = [evaluate_cut(data, th, **eval_kw) for th in _neighbours(as_theta(theta), grids)]
    return pd.DataFrame(rows).sort_values('Z', ascending=False, ignore_index=True)


def run_optimization(data, starts=None, grids=None, max_restarts=3, verbose=True, **eval_kw):
    """Coordinate descent from several starting points + local grid check.

    starts: {label: cut string or theta}; default open, util.lgb_tight,
    util.lgb_loose. If the local check finds a better neighbour, the descent
    is restarted from it (at most max_restarts times).
    Returns dict(summary, paths, thetas, best_theta, grids).
    """
    starts = starts or {'open': OPEN_CUT, 'lgb_tight': util.lgb_tight, 'lgb_loose': util.lgb_loose}
    grids = grids or make_grids(data, half=eval_kw.get('half', 'A'))
    paths, thetas, summary = [], {}, []
    for label, th0 in starts.items():
        theta, n_restart = as_theta(th0), 0
        while True:
            theta, path = coordinate_descent(data, theta, grids, label=label,
                                             verbose=verbose, **eval_kw)
            path['restart'] = n_restart
            paths.append(path)
            z = path['Z'].iloc[-1]
            check = local_grid_check(data, theta, grids, **eval_kw)
            if check['Z'].iloc[0] > z * (1 + 1e-6) and n_restart < max_restarts:
                theta, n_restart = {v: check[v].iloc[0] for v in MVA_OUTPUTS}, n_restart + 1
                if verbose:
                    print(f'[{label}] neighbour beats the descent result; restarting')
                continue
            break
        thetas[label] = theta
        summary.append(dict(start=label, Z=z, restarts=n_restart, cut=theta_to_cut(theta), **theta))
    summary = pd.DataFrame(summary).sort_values('Z', ascending=False, ignore_index=True)
    best = thetas[summary['start'].iloc[0]]
    return dict(summary=summary, paths=pd.concat(paths, ignore_index=True),
                thetas=thetas, best_theta=best, grids=grids)


############################## uncertainty + working points ##################

def bootstrap_z(data, theta, n_boot=200, seed=0, **eval_kw):
    """Bootstrap the MC-statistical spread of Z at theta.

    After BCS each selected candidate is one event, so Poisson(1) weights per
    selected candidate resample events. Returns (Z nominal, std of Z, all Z).
    """
    theta = as_theta(theta)
    half = eval_kw.get('half', 'A')
    bcs_order = eval_kw.get('bcs_order', 'cut_then_bcs')
    bcs_base = eval_kw.get('bcs_base')
    eps_bkg = eval_kw.get('eps_bkg', 0.0)
    min_bin_count = eval_kw.get('min_bin_count', MIN_BIN_COUNT)
    count_on = eval_kw.get('count_on', 'fom')
    estimator = eval_kw.get('estimator', 'unbiased')
    rng = np.random.default_rng(seed)
    sels = {name: select(data, name, theta, half, bcs_order, bcs_base) for name in data.channels}
    zs = np.empty(n_boot)
    for k in range(n_boot):
        z2 = 0.0
        for name, ch in data.channels.items():
            pw = rng.poisson(1.0, size=len(sels[name]))
            z2 += _channel_result(ch, sels[name], eps_bkg, min_bin_count, count_on,
                                  estimator, weights=pw)[0]
        zs[k] = np.sqrt(z2)
    z0 = evaluate_cut(data, theta, **eval_kw)['Z']
    return z0, float(np.std(zs, ddof=1)), zs


def greedy_loosen(data, theta, grids, z_floor, **eval_kw):
    """Loosen theta one grid step at a time while Z stays >= z_floor.

    At each step every output is tried one step looser; the move with the
    largest signal gain per unit of Z lost is taken (a move that does not
    lower Z is taken first). Returns (theta, path DataFrame).
    """
    theta = as_theta(theta)
    cur = evaluate_cut(data, theta, **eval_kw)
    path = [dict(cur, step='start')]
    while True:
        best, best_score = None, -np.inf
        for v in MVA_OUTPUTS:
            g = np.unique(np.append(grids[v], theta[v]))
            i = int(np.searchsorted(g, theta[v]))
            j = i - 1 if CUT_DIRECTIONS[v] == '>' else i + 1
            if j < 0 or j >= len(g):
                continue
            th = dict(theta, **{v: float(g[j])})
            r = evaluate_cut(data, th, **eval_kw)
            if r['Z'] < z_floor:
                continue
            gain, loss = r['S'] - cur['S'], cur['Z'] - r['Z']
            score = np.inf if loss <= 0 else gain / loss
            if score > best_score:
                best, best_score = (v, th, r), score
        if best is None:
            return theta, pd.DataFrame(path)
        v, theta, cur = best
        path.append(dict(cur, step=v))


def redundancy_table(data, theta, **eval_kw):
    """For each condition of theta: fraction of selected events it removes on
    top of the other three. ~0 means the condition is inactive (the outputs
    sum to one, so e.g. a high sig_prob threshold can make others redundant)."""
    theta = as_theta(theta)
    half = eval_kw.get('half', 'A')
    bcs_order = eval_kw.get('bcs_order', 'cut_then_bcs')
    bcs_base = eval_kw.get('bcs_base')
    n_full = sum(len(select(data, n, theta, half, bcs_order, bcs_base)) for n in data.channels)
    rows = []
    for v in MVA_OUTPUTS:
        if is_open(v, theta[v]):
            continue
        n_wo = sum(len(select(data, n, dict(theta, **{v: OPEN_CUT[v]}), half, bcs_order, bcs_base))
                   for n in data.channels)
        z_wo = evaluate_cut(data, dict(theta, **{v: OPEN_CUT[v]}), **eval_kw)['Z']
        rows.append(dict(condition=f'{v}{CUT_DIRECTIONS[v]}{theta[v]:.4g}',
                         frac_removed=1 - n_full / n_wo if n_wo else np.nan, Z_without=z_wo))
    return pd.DataFrame(rows)


def choose_working_points(data, theta_star, grids=None, k_sigma=1.0, loose_loss=0.05,
                          n_boot=200, **eval_kw):
    """Tight and loose working points from the FOM optimum.

    tight: loosest point reachable by greedy loosening with Z >= Z* - k_sigma*sigma_Z
           (the plateau rule; sigma_Z from bootstrap at theta_star).
    loose: same, with Z >= (1 - loose_loss) * Z*.
    """
    grids = grids or make_grids(data, half=eval_kw.get('half', 'A'))
    z_star, sigma, _ = bootstrap_z(data, theta_star, n_boot=n_boot, **eval_kw)
    tight, tight_path = greedy_loosen(data, theta_star, grids, z_star - k_sigma * sigma, **eval_kw)
    loose_floor = (1 - loose_loss) * z_star
    if tight_path['Z'].iloc[-1] < loose_floor:
        # The plateau already costs more than loose_loss: MC statistics cannot
        # resolve the two working points, so they coincide.
        print(f'WARNING: k_sigma*sigma_Z = {k_sigma * sigma:.3g} exceeds the loose budget '
              f'{loose_loss:.0%} of Z* = {loose_loss * z_star:.3g}; loose = tight.')
        loose, loose_path = dict(tight), tight_path.iloc[[-1]]
    else:
        loose, loose_path = greedy_loosen(data, tight, grids, loose_floor, **eval_kw)
    return dict(optimum=as_theta(theta_star), tight=tight, loose=loose,
                z_star=z_star, sigma_z=sigma,
                tight_path=tight_path, loose_path=loose_path,
                redundancy_tight=redundancy_table(data, tight, **eval_kw))


############################## validation #####################################

def validation_report(cuts, datasets, bcs_orders=('cut_then_bcs',), bcs_base=None,
                      eps_bkg=0.0, min_bin_count=MIN_BIN_COUNT, count_on='fom',
                      estimator='unbiased'):
    """One row per (cut, dataset, BCS order).

    cuts     : {name: cut string or theta}
    datasets : {label: (FOMInput, half)}, e.g. {'run2 A': (d, 'A'),
               'run2 B': (d, 'B'), 'run1 mu': (d1, 'all')}
    """
    rows = []
    for cname, cut in cuts.items():
        theta = as_theta(cut)
        for dlabel, (data, half) in datasets.items():
            for order in bcs_orders:
                kw = dict(half=half, bcs_order=order, bcs_base=bcs_base)
                r = evaluate_cut(data, theta, eps_bkg=eps_bkg, min_bin_count=min_bin_count,
                                 count_on=count_on, estimator=estimator, **kw)
                eff = efficiencies(data, theta, **kw)
                rows.append(dict(cut_name=cname, dataset=dlabel, bcs_order=order,
                                 cut=theta_to_cut(theta),
                                 **{k: r[k] for k in r if k.startswith('Z')},
                                 n_bins=r['n_bins'], excess_frac=r['excess_frac'],
                                 S=r['S'], B=r['B'], **eff))
    return pd.DataFrame(rows)


############################## per-category efficiencies #####################

# Display order of the truth categories (anything else is appended at the end).
PHYSICS_CATEGORIES = [r'$D\ell\nu$', r'$D^\ast\tau\nu$', r'$D^\ast\ell\nu$',
                      r'$D^{\ast\ast}\tau\nu$', r'$D^{\ast\ast}\ell\nu$_narrow',
                      r'$D^{\ast\ast}\ell\nu$_broad', r'$D\ell\nu$_gap']
CATEGORY_ORDER = FOM_SIGNAL + FOM_BACKGROUND + PHYSICS_CATEGORIES
EFF_REGIONS = ('sr', 'fit', 'all')


def _category_group(cat):
    if cat in FOM_SIGNAL:
        return 'signal'
    if cat in FOM_BACKGROUND:
        return 'FOM background'
    if cat in PHYSICS_CATEGORIES:
        return 'physics'
    return 'other'


def _region_mask(ch, idx, region):
    if region == 'sr':
        return ch.in_sr[idx]
    if region == 'fit':
        return ch.binidx[idx] >= 0
    if region == 'all':
        return np.ones(len(idx), dtype=bool)
    raise ValueError(f'region must be one of {EFF_REGIONS}.')


def _kept_candidates(data, name, theta, half, level, bcs_order, bcs_base):
    """Indices kept by theta: after BCS (level='event') or before (level='candidate')."""
    ch = data.channels[name]
    if level == 'event':
        return select(data, name, theta, half, bcs_order, bcs_base)
    if level == 'candidate':
        m = cut_mask(ch, theta)
        h = _half_mask(ch, half)
        return np.flatnonzero(m if h is None else m & h)
    raise ValueError("level must be 'event' or 'candidate'.")


def _category_yields(ch, idx, region):
    keep = _region_mask(ch, idx, region)
    y = np.bincount(ch.cat[idx][keep], weights=ch.w[idx][keep], minlength=len(ch.cat_names))
    return dict(zip(ch.cat_names, y))


def efficiency_table(data, cuts, half='all', level='event', region='sr',
                     bcs_order='cut_then_bcs', bcs_base=None):
    """Efficiency of each BDT cut for every MC truth category.

    efficiency = N(category | cut) / N(category | no MVA cut), both counted
    in `region` after the same selection chain, per channel and for e+mu.

    cuts   : {name: cut string or theta}, e.g. {'new tight': wps['tight'],
             'lgb_tight': util.lgb_tight}
    half   : 'all' (default), 'A' or 'B'
    level  : 'event'     - after BCS, i.e. the yields that enter the fit (default).
                           BCS picks the best candidate among those that pass, so
                           an event can change category when the cut removes its
                           previous best candidate; N(after) is then not a strict
                           subset of N(before), a category can (rarely) exceed 1,
                           and the binomial error is only approximate.
             'candidate' - before BCS: the pure MVA-cut efficiency per candidate,
                           a strict subset, so the binomial error is exact.
    region : 'sr' (D_M signal window, default), 'fit' (D_M window and inside the
             fit binning) or 'all'.

    Returns a long DataFrame with one row per (cut, channel, category):
    group, n_before, n_after, eff, eff_err (binomial, sqrt(eff(1-eff)/n_before)).
    Two summary rows per channel, 'FOM signal (total)' and 'FOM background
    (total)', sum the categories entering s and b.
    """
    if region not in EFF_REGIONS:
        raise ValueError(f'region must be one of {EFF_REGIONS}.')
    rows = []
    for cname, cut in cuts.items():
        theta = as_theta(cut)
        totals_before, totals_after = {}, {}
        per_channel = {}
        for name, ch in data.channels.items():
            before = _category_yields(ch, _kept_candidates(data, name, OPEN_CUT, half, level,
                                                           bcs_order, bcs_base), region)
            after = _category_yields(ch, _kept_candidates(data, name, theta, half, level,
                                                          bcs_order, bcs_base), region)
            per_channel[name] = (before, after)
            for k in before:
                totals_before[k] = totals_before.get(k, 0.0) + before[k]
                totals_after[k] = totals_after.get(k, 0.0) + after.get(k, 0.0)
        if len(data.channels) > 1:
            per_channel['e+mu' if set(data.channels) == {'e', 'mu'} else 'all channels'] = \
                (totals_before, totals_after)
        for chan, (before, after) in per_channel.items():
            summary = {'FOM signal (total)': FOM_SIGNAL, 'FOM background (total)': FOM_BACKGROUND}
            entries = [(c, before.get(c, 0.0), after.get(c, 0.0)) for c in before]
            entries += [(label, sum(before.get(c, 0.0) for c in members),
                         sum(after.get(c, 0.0) for c in members))
                        for label, members in summary.items()]
            for cat, nb, na in entries:
                eff = na / nb if nb > 0 else np.nan
                err = np.sqrt(eff * (1 - eff) / nb) if nb > 0 and 0 <= eff <= 1 else np.nan
                group = 'summary' if cat in summary else _category_group(cat)
                rows.append(dict(cut_name=cname, cut=theta_to_cut(theta), channel=chan,
                                 category=cat, group=group, n_before=nb, n_after=na,
                                 eff=eff, eff_err=err))
    table = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(CATEGORY_ORDER + ['FOM signal (total)',
                                                          'FOM background (total)'])}
    table['_order'] = table['category'].map(lambda c: order.get(c, len(order)))
    table['_cut'] = table['cut_name'].map({c: i for i, c in enumerate(cuts)})
    chans = list(dict.fromkeys(table['channel']))
    table['_chan'] = table['channel'].map({c: i for i, c in enumerate(chans)})
    table = table.sort_values(['_cut', '_chan', '_order', 'category'], kind='stable')
    return table.drop(columns=['_order', '_cut', '_chan']).reset_index(drop=True)


def efficiency_summary(table, channel='e+mu', percent=True, decimals=1):
    """Wide view of efficiency_table: one row per category, one column per cut,
    entries 'eff +- err' (in % by default), plus the number of MC events
    before the cut."""
    t = table[table['channel'] == channel]
    scale = 100.0 if percent else 1.0
    fmt = lambda r: (f"{scale * r['eff']:.{decimals}f} ± {scale * r['eff_err']:.{decimals}f}"
                     if np.isfinite(r['eff_err']) else f"{scale * r['eff']:.{decimals}f}")
    t = t.assign(value=t.apply(fmt, axis=1))
    cut_order = list(dict.fromkeys(t['cut_name']))
    cat_order = list(dict.fromkeys(t['category']))
    wide = t.pivot(index='category', columns='cut_name', values='value')
    wide = wide.reindex(index=cat_order, columns=cut_order)
    n0 = t.drop_duplicates('category').set_index('category')['n_before'].reindex(cat_order)
    group = t.drop_duplicates('category').set_index('category')['group'].reindex(cat_order)
    wide.insert(0, 'N before (MC)', n0.round().astype('Int64'))
    wide.insert(0, 'group', group)
    wide.columns.name = f'efficiency [{"%" if percent else "fraction"}], {channel}'
    return wide


# categorical slots 1-4 of the dataviz reference palette (light), each with its
# own marker so identity never relies on colour alone
_CUT_STYLES = [('#2a78d6', 'o'), ('#eb6834', 's'), ('#1baf7a', 'D'), ('#eda100', '^')]


def plot_efficiencies(table, channel='e+mu', log=True, save=None):
    """Dot plot of the per-category efficiencies: one row per category, one
    marker per cut (up to four cuts), binomial error bars, log x by default
    because efficiencies span orders of magnitude. On a log axis a zero
    efficiency (no MC event passes) is drawn as a hollow marker at the left
    edge, labelled '0'."""
    t = table[table['channel'] == channel]
    cuts = list(dict.fromkeys(t['cut_name']))
    if len(cuts) > len(_CUT_STYLES):
        raise ValueError(f'At most {len(_CUT_STYLES)} cuts per plot; split them.')
    cats = list(dict.fromkeys(t['category']))
    y = {c: i for i, c in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(cats) + 1.6))
    offsets = np.linspace(-0.22, 0.22, len(cuts)) if len(cuts) > 1 else [0.0]
    positive = t.loc[t['eff'] > 0, 'eff']
    x_floor = positive.min() / 3 if log and len(positive) else 0.0
    zeros = []
    for (cut, off), (color, marker) in zip(zip(cuts, offsets), _CUT_STYLES):
        d = t[t['cut_name'] == cut]
        ys = d['category'].map(y) + off
        nz = d['eff'] > 0 if log else np.ones(len(d), dtype=bool)
        ax.errorbar(d['eff'][nz], ys[nz], xerr=d['eff_err'][nz].fillna(0), fmt=marker, ms=6,
                    color=color, ecolor=color, elinewidth=1, capsize=0, label=cut)
        if log and (~nz).any():
            ax.plot(np.full((~nz).sum(), x_floor), ys[~nz], marker, ms=6, mfc='none',
                    color=color)
            zeros += list(ys[~nz])
    groups = t.drop_duplicates('category').set_index('category')['group']
    for i, c in enumerate(cats[1:], start=1):          # thin rule between groups
        if groups[c] != groups[cats[i - 1]]:
            ax.axhline(i - 0.5, color='#c3c2b7', lw=0.8)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    if log:
        ax.set_xscale('log')
        if zeros:
            ax.set_xlim(left=x_floor / 1.6)
            for yz in zeros:
                ax.annotate('0', (x_floor, yz), xytext=(-9, 0), textcoords='offset points',
                            ha='right', va='center', fontsize=8, color='#5f5e5a')
    ax.set_xlabel(f'efficiency w.r.t. no BDT cut ({channel})')
    ax.grid(axis='x', alpha=0.3, which='both')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(cuts), frameon=False)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


############################## plots ##########################################

def n_minus_1_scans(data, theta, grids, **eval_kw):
    """N-1 scans around theta: {output: scan DataFrame}."""
    return {v: scan_1d(data, v, theta, np.unique(np.append(grids[v], as_theta(theta)[v])), **eval_kw)
            for v in MVA_OUTPUTS}


def plot_n_minus_1(scans, marks=None, sigma_z=None, z_ref=None, save=None):
    """Z vs threshold for each output (2x2), with the signal and background
    yields relative to the open point on a twin axis.

    marks: {label: theta or cut string} drawn as vertical lines.
    sigma_z, z_ref: draw the z_ref - sigma_z band (plateau rule).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    styles = ['-', '--', ':', '-.']
    for ax, var in zip(axes.flat, MVA_OUTPUTS):
        sc = scans[var].sort_values('threshold')
        ax.plot(sc['threshold'], sc['Z'], 'k.-', label='Z')
        if sigma_z is not None and z_ref is not None:
            ax.axhspan(z_ref - sigma_z, z_ref, color='gray', alpha=0.25,
                       label=r'$Z^\ast - 1\sigma$')
        ax2 = ax.twinx()
        open_row = sc.loc[sc['threshold'].map(lambda t: _looseness(var, t)).idxmax()]
        ax2.plot(sc['threshold'], sc['S'] / open_row['S'], color='tab:blue', lw=1, label='S / S(open)')
        ax2.plot(sc['threshold'], sc['B'] / open_row['B'], color='tab:red', lw=1, label='B / B(open)')
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel('relative yield')
        for (label, th), ls in zip((marks or {}).items(), styles):
            ax.axvline(as_theta(th)[var], color='tab:green', ls=ls, label=label)
        ax.set_xlabel(f'{var} threshold ({var}{CUT_DIRECTIONS[var]}t)')
        ax.set_ylabel('binned FOM Z')
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower left')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


def plot_descent_paths(paths, save=None):
    """Z after every 1D scan, one line per starting point."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, restart), p in paths.groupby(['start', 'restart'], sort=False):
        ax.plot(np.arange(len(p)), p['Z'], 'o-', ms=3,
                label=label if restart == 0 else f'{label} (restart {restart})')
    ax.set_xlabel('step (1D scans)')
    ax.set_ylabel('binned FOM Z')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig