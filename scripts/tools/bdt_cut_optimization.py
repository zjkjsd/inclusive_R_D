# -*- coding: utf-8 -*-
"""BDT selection optimization with a binned figure of merit.

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
* Luminosity: FOMInput(..., lumi_scale=L_data/L_MC) (0.25 for the generic MC)
  multiplies every MC weight, so all yields, Z and sigma_S/S are at DATA
  luminosity. For the statistical FOM alone this only rescales Z (by
  sqrt(lumi_scale)) and leaves the optimum unchanged; it matters as soon as a
  term that does not scale with luminosity enters (eps_bkg, control regions):
  the statistical term falls as 1/L, the systematic ones do not, so raw MC
  (4 x data) would understate the statistical uncertainty by a factor 2 and
  push the optimum toward tight cuts. The bin threshold always counts raw MC
  events, and the excess-free estimator is exact under a constant weight scale.
* Every evaluation reports both Z (maximized internally) and the relative
  uncertainty of the signal yield sigma_S/S = 1/Z (statistical only, or with
  the control-region terms).
* Thresholds come from grids rounded to GRID_DECIMALS (2) decimals, so the
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
  sensitivity behind it, so without a bin threshold the plain FOM is pushed
  toward looser cuts by MC statistics alone. With the threshold the direction
  depends on the purity of the kept bins: p(1-p) grows with p for p < 0.5, so
  in background-dominated bins a tighter cut raises the excess and the plain
  FOM is pulled tighter (seen on the run2 MC). Either way it is a bias, which
  the default estimator removes.
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
* No excess at any purity and any bin size: the quantity the optimizer sees
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
Weighted MC (corrections, FOMInput(..., weight_col=MC_WEIGHT)): the
estimator generalizes to (s^2 - W2_s) / (n - W2_n / n), W2 = sum of squared
weights, n = weighted s + b. It is s(s-1)/(n-1) for unit weights, exact for
equal weights, and toys with weights scattered around 1 (spread up to 30 %)
show no excess (scripts/tests). The bin threshold always counts raw MC events.
Limits: exact for eps_bkg = 0; with eps_bkg > 0 the denominator
n - W2_n/n + (eps b^)^2 is an approximation.
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

Target with control-region systematics (control_regions=[...])
--------------------------------------------------------------
Data/MC mismodeling measured in control regions turns into an uncertainty on
the signal-region background, and the target becomes the expected relative
uncertainty of the D tau nu yield, sigma_S / S (to be minimized; Z = 1/(sigma_S/S)
is what the optimizer maximizes, so all tools work unchanged).

Each control region (ControlRegion) compares TOTAL data with TOTAL MC after the
same BDT cut and BCS, in its own coarse binning (ControlRegion.bins, set once per
region by auto_cr_bins), and gives two fractions
(excess_variance, the same numbers mismodeling_label prints on residual plots):
    n     = max(|R - 1|, R_err)  normalization; R = data/MC ratio (on all data).
            The measured offset already contains the statistical scatter
            (E[(R-1)^2] = (R_true-1)^2 + R_err^2), so it is not added again;
            R_err is a floor: the normalization is not known better than it is
            measured. On one half: see ControlRegion.measure. norm_estimator=
            'quadrature' gives the former sqrt((R-1)^2 + R_err^2);
            include_offset=False keeps only R_err.
    delta = shape mismodeling (excess variance after scaling MC by R; 0 if the
            measured delta^2 is negative)
A fraction only costs as much as there are signal-region events of the same
kind, so each region declares which signal-region background it describes
(applies_to, evaluated on the signal-region MC truth):
    D_mass_sidebands -> category == 'bkg_fakeD'
    offres           -> B0_isContinuumEvent == 1   (continuum and qq fake D)
    combinatorial    -> placeholder with hand-set n, delta (ControlRegion.fixed)
Single-bin form, with B_k the signal-region background described by region k:
    (sigma_S/S)^2 = (S + B)/S^2 + sum_k (n_k^2 + delta_k^2) (B_k / S)^2
i.e. statistical term plus, per region, its fraction times the background-to-
signal ratio of the events it describes. Binned: relative_uncertainty() builds
the Fisher matrix of nu_i = mu s_i + b_i + sum_k alpha_k b_k,i with Gaussian
priors of width n_k on alpha_k and per-bin variance s_i + b_i + sum_k
(delta_k b_k,i)^2; sigma_S/S = sqrt((F^-1)_mumu). The Fisher matrix is the
curvature of the likelihood of the Asimov data set (data = expectation), i.e.
the uncertainty an Asimov fit would report, for a model linear in its
parameters. Without control regions it is exactly the statistical FOM, so the
estimator choice ('unbiased': s(s-1) and b_k b_l - b_kl in place of s^2 and
b_k b_l) and the bin threshold carry over. A normalization term hurts only as
far as the background's shape across bins resembles the signal's; the Fisher
matrix accounts for that, the single-bin form does not.
Breakdown: rel_unc_stat, rel_unc_norm, rel_unc_shape (quadrature differences)
and per region its impact sqrt(total^2 - total_without_region^2).

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
# Normalization mode: only used for the efficiency-ratio guardrail.
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
# MC-truth columns kept when present (used to define which signal-region events a
# control region describes, e.g. 'B0_isContinuumEvent == 1' for off-resonance).
TRUTH_FLAG_COLUMNS = ['B0_isContinuumEvent']


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


############################## MC corrections #################################
#
# A correction is a per-candidate MC weight computed while loading, from the full
# set of branches (before load_candidates drops them). Each one is stored as
# column 'w_<name>', and their product as MC_WEIGHT. Without corrections
# MC_WEIGHT is 1, so FOMInput(..., weight_col=MC_WEIGHT) is always safe.
#
# To add one: register_correction('name', func, branches=[...]), where
#     func(df, mode, run, **options) -> array of weights, one per row of df
# df has every branch of util.all_relevant_variables (+ `branches`), the derived
# variables and the BDT outputs; mode is 'e' or 'mu'; run is 'run1' or 'run2'.

MC_WEIGHT = 'mc_weight'
CORRECTIONS = {}


def register_correction(name, func, branches=()):
    """Make a correction available to load_candidates(corrections=[name])."""
    CORRECTIONS[name] = dict(func=func, branches=tuple(branches))


def _pid_weight(df, mode, run, MC='MC16'):
    """PID efficiency and fake-rate corrections of the lepton, K and both pions
    (util.apply_pid_corrections)."""
    out = util.apply_pid_corrections(df=df, MC=MC, run=run, channel=mode,
                                     corr_col_name='_pid_weight_tmp')
    return out['_pid_weight_tmp'].to_numpy()


register_correction('pid', _pid_weight)


def apply_corrections(df, mode, run, corrections=(), options=None):
    """Weight columns of the requested corrections and their product.

    corrections : names registered in CORRECTIONS, e.g. ['pid'] (default: none)
    options     : {name: dict of keyword arguments}, e.g. {'pid': {'MC': 'MC16'}}
    Returns a DataFrame with 'w_<name>' per correction and MC_WEIGHT (their
    product; 1 when there are no corrections).
    """
    options = options or {}
    out = pd.DataFrame(index=df.index)
    total = np.ones(len(df))
    for name in corrections:
        if name not in CORRECTIONS:
            raise ValueError(f'Unknown correction {name!r}; registered: {sorted(CORRECTIONS)}.')
        if run is None:
            raise ValueError(f'Correction {name!r} needs run= ("run1" or "run2").')
        w = np.asarray(CORRECTIONS[name]['func'](df, mode, run, **options.get(name, {})), float)
        if w.shape != (len(df),):
            raise ValueError(f'Correction {name!r} returned {w.shape}, expected ({len(df)},).')
        out[f'w_{name}'] = w
        total = total * w
    out[MC_WEIGHT] = total
    return out


# Off-resonance samples are recorded ~60 MeV below the Y(4S), but the BDT was
# trained on on-resonance MC. Its beam-energy-dependent inputs, the ROE-side
# Mbc and DeltaE, are therefore recomputed as at the on-resonance energy before
# the offline cut and the BDT (load_candidates(..., ecms_target=ONRES_ECMS)),
# so that the BDT (and the offline cut) act on off-resonance events as on
# on-resonance ones. The other training variables do not depend on the beam
# energy (normalized shape moments, D variables, angles, significances). The fit
# variables (M2_miss, p_D + p_l) are left as measured: data and MC of the
# off-resonance region are at the same energy, so their comparison is direct.
ONRES_ECMS = 10.58                                    # GeV
BEAM_ENERGY_VARS = ('B0_roeMbc_my_mask', 'B0_roeDeltae_my_mask')
_BEAM_ENERGY_BRANCHES = ('Ecms', 'B0_CMS_roeP_my_mask', 'B0_CMS_roeE_my_mask')


def correct_beam_energy(df, ecms_target=ONRES_ECMS):
    """Recompute the ROE-side Mbc and DeltaE as at the CMS energy ecms_target, in place.

    The ROE energy and momentum in the CMS (B0_CMS_roeE/P_my_mask) are scaled by
    ecms_target / Ecms (the event's own CMS energy), and the beam energy is
    ecms_target / 2:
        Mbc    = sqrt((ecms_target/2)^2 - (p* ecms_target/Ecms)^2)   (NaN if negative)
        DeltaE = E* ecms_target/Ecms - ecms_target/2
    i.e. Mbc and DeltaE scaled by ecms_target / Ecms. The original values are
    kept as B0_roeMbc_ori and B0_roeDeltae_ori.

    Returns a summary: n = candidates whose values recomputed at the event's
    own Ecms could be compared with the ntuple, n_off = those differing by more
    than 1 MeV in Mbc or DeltaE (should be 0: the formula matches the ntuple
    definition); n_shift, sum_dMbc, sum_dDeltaE = sums of the applied changes
    (corrected - original), for the mean shift.
    """
    ecms = df['Ecms'].to_numpy(dtype=float)
    p = df['B0_CMS_roeP_my_mask'].to_numpy(dtype=float)
    e = df['B0_CMS_roeE_my_mask'].to_numpy(dtype=float)
    k = ecms_target / ecms

    def mbc(ebeam, pp):
        m2 = ebeam ** 2 - pp ** 2
        return np.where(m2 >= 0, np.sqrt(np.abs(m2)), np.nan)

    mbc_orig = df['B0_roeMbc_my_mask'].to_numpy(dtype=float)
    de_orig = df['B0_roeDeltae_my_mask'].to_numpy(dtype=float)
    with np.errstate(invalid='ignore'):
        d_mbc = np.abs(mbc(ecms / 2, p) - mbc_orig)
        d_de = np.abs((e - ecms / 2) - de_orig)
    comparable = np.isfinite(d_mbc) & np.isfinite(d_de)
    new_mbc = mbc(ecms_target / 2, p * k)
    new_de = e * k - ecms_target / 2
    shift = np.isfinite(new_mbc) & np.isfinite(mbc_orig)
    check = dict(n=int(comparable.sum()),
                 n_off=int(((d_mbc > 1e-3) | (d_de > 1e-3))[comparable].sum()),
                 n_shift=int(shift.sum()),
                 sum_dMbc=float((new_mbc - mbc_orig)[shift].sum()),
                 sum_dDeltaE=float((new_de - de_orig)[shift].sum()))
    df['B0_roeMbc_ori'] = mbc_orig
    df['B0_roeDeltae_ori'] = de_orig
    df['B0_roeMbc_my_mask'] = new_mbc
    df['B0_roeDeltae_my_mask'] = new_de
    return check


def _split_cut(cut, variables):
    """(terms of an '&'-joined cut without the variables, all terms), as cut strings."""
    terms = [t.strip() for t in cut.split('&') if t.strip()]
    rest = [t for t in terms if not any(v in t for v in variables)]
    return ' & '.join(rest) if rest else None, ' & '.join(terms)


def load_candidates(files, mode, tree=None, extra_cut=None, step_size='1 GB',
                    is_data=False, run=None, corrections=(), correction_options=None,
                    ecms_target=None, verbose=True):
    """Load candidate-level generic MC or data (all candidates, no MVA cut, no BCS).

    Reads the reconstruction ntuples chunk by chunk, applies util.offline_cut
    (+ extra_cut, e.g. util.Dst_veto_cut), adds the derived variables, attaches
    the BDT outputs with util.apply_mva_bcs(bcs=None) and the truth category,
    and keeps only the few columns the FOM needs (float32), so a full run of
    generic MC fits in memory.

    files       : list of ROOT file paths. tree defaults to 'B0_<mode>'.
    is_data     : True for real data (on- or off-resonance): no truth
                  classification, every candidate gets category 'data', no weights.
    run         : 'run1' or 'run2' (needed by run-dependent corrections; load
                  run1 and run2 MC in separate calls).
    corrections : MC weights to attach, by name (see CORRECTIONS; default none,
                  e.g. ['pid']). Columns 'w_<name>' and MC_WEIGHT (product) are
                  kept; MC_WEIGHT = 1 without corrections.
    correction_options : {name: kwargs} passed to the correction functions.
    ecms_target : None (default), or the CMS energy to which the beam-energy
                  dependent BDT inputs are corrected (ONRES_ECMS for the
                  off-resonance data and MC, see correct_beam_energy). The
                  offline cut on them is then applied after the correction.
    MC also keeps TRUTH_FLAG_COLUMNS (B0_isContinuumEvent).
    """
    tree = tree or f'B0_{mode}'
    cut = util.offline_cut if extra_cut is None else f'({util.offline_cut}) & ({extra_cut})'
    corrections = list(corrections or [])
    if is_data and corrections:
        raise ValueError('Corrections are for MC only (is_data=True with corrections).')
    keep = list(_KEEP_COLUMNS)
    if not is_data:
        keep += TRUTH_FLAG_COLUMNS + [f'w_{c}' for c in corrections] + [MC_WEIGHT]
    branches = set(util.all_relevant_variables)
    for c in corrections:
        branches |= set(CORRECTIONS.get(c, {}).get('branches', ()))
    read_cut, post_cut = cut, None
    if ecms_target is not None:
        branches |= set(_BEAM_ENERGY_BRANCHES) | set(BEAM_ENERGY_VARS)
        # cut on the beam-energy dependent variables only after their correction
        read_cut, post_cut = _split_cut(cut, BEAM_ENERGY_VARS)
    chunks = []
    summary = dict(n=0, n_off=0, n_shift=0, sum_dMbc=0.0, sum_dDeltaE=0.0,
                   n_read=0, n_pass=0, n_pass_ori=0)
    import uproot
    for i, batch in enumerate(uproot.iterate(
            [f'{f}:{tree}' for f in files], cut=read_cut, library='pd', step_size=step_size,
            filter_name=lambda name: name in branches)):
        batch = batch.reset_index(drop=True)
        if ecms_target is not None:
            chk = correct_beam_energy(batch, ecms_target)
            for key, val in chk.items():
                summary[key] += val
            # how many would pass the offline cut with the original values (for the summary only)
            cut_ori = post_cut.replace('B0_roeMbc_my_mask', 'B0_roeMbc_ori') \
                              .replace('B0_roeDeltae_my_mask', 'B0_roeDeltae_ori')
            summary['n_read'] += len(batch)
            summary['n_pass_ori'] += int(batch.eval(cut_ori).sum())
            batch = batch.query(post_cut).reset_index(drop=True)
            summary['n_pass'] += len(batch)
            if len(batch) == 0:
                continue
        util.add_derived_variables(batch)
        with contextlib.redirect_stdout(io.StringIO()):
            # cut='ell_p>0' is always true: predictions only, no MVA cut, no BCS
            pred = util.apply_mva_bcs(batch, util.training_variables, cut='ell_p>0',
                                      library='lgbm', model='multiclass', bcs=None)
        pred = pred.reset_index(drop=True)
        if is_data:
            category = pd.Series('data', index=pred.index)
        else:
            pred = pd.concat([pred, apply_corrections(pred, mode, run, corrections,
                                                      correction_options)], axis=1)
            category = classify_to_codes(pred, mode)
        out = pred[[c for c in keep if c in pred.columns]].copy()
        for col in MVA_OUTPUTS + FIT_VARIABLES + ['D_M', BCS_RANK_VAR]:
            out[col] = out[col].astype(np.float32)
        out['category'] = pd.Categorical(category.to_numpy())
        chunks.append(out)
        if verbose:
            print(f'[{mode}] chunk {i}: {len(out):,d} candidates')
    if ecms_target is not None:
        n, n_off, ns = summary['n'], summary['n_off'], max(summary['n_shift'], 1)
        if verbose:
            print(f"[{mode}] beam-energy correction to {ecms_target} GeV applied to all "
                  f"{summary['n_read']:,d} candidates read: mean shift Mbc "
                  f"{1e3 * summary['sum_dMbc'] / ns:+.1f} MeV, DeltaE "
                  f"{1e3 * summary['sum_dDeltaE'] / ns:+.1f} MeV; offline cut passed by "
                  f"{summary['n_pass']:,d} (with the uncorrected values it would be "
                  f"{summary['n_pass_ori']:,d})")
            print(f"[{mode}]   formula check: {'OK' if n_off == 0 else 'NOT OK'}, {n_off} of {n:,d} "
                  f"candidates differ from the ntuple Mbc / DeltaE (recomputed at their own Ecms) "
                  f"by > 1 MeV")
        if n and n_off > 1e-3 * n:
            import warnings
            warnings.warn(f'{n_off} of {n} candidates: Mbc / DeltaE recomputed at the event Ecms '
                          'differ from the ntuple by > 1 MeV; check correct_beam_energy.')
    df = pd.concat(chunks, ignore_index=True)
    df['category'] = df['category'].astype('category')
    df.attrs['corrections'] = corrections
    df.attrs['ecms_target'] = ecms_target
    if ecms_target is not None:
        df.attrs['beam_energy_summary'] = summary
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
        # raw values, for control regions with their own D_M window and binning
        self.x, self.y, self.dm = x.astype(np.float32), y.astype(np.float32), dm.astype(np.float32)
        self.truth = {c: df[c].to_numpy()[order] for c in TRUTH_FLAG_COLUMNS if c in df.columns}

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
    weight_col : optional per-candidate weight column (None = 1), e.g. MC_WEIGHT.
    lumi_scale : L_data / L_MC, multiplies every weight (0.25 for the generic
                 MC; 1 for data). Yields, Z and sigma_S/S are then at data
                 luminosity, which the control-region target needs (see the
                 module docstring). Bin thresholds still count raw events.
    half_col   : 'event' splits by event-number parity (half A = even, B = odd);
                 or the name of a 0/1 column.
    """

    def __init__(self, dfs, bins=FOM_BINS, weight_col=None, lumi_scale=1.0, half_col='event'):
        self.bins = bins
        self.lumi_scale = float(lumi_scale)
        self.channels = {name: _Channel(df, bins, weight_col, half_col)
                         for name, df in dfs.items()}
        for ch in self.channels.values():
            ch.w = ch.w * self.lumi_scale
            ch.unit_weights = bool(np.all(ch.w == 1.0))
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


# One half (A or B, event-number parity) holds half of the MC and half of the
# data. MC weights are scaled to the FULL data luminosity (lumi_scale), so the
# yields of one half correspond to half of it; HALF_LUMI_FACTOR scales them back
# up, so that yields, Z and sigma_S/S evaluated on one half are those expected
# for the full data set (the statistical part shrinks by sqrt(2), the
# control-region fractions do not depend on the luminosity). MC-statistical
# fluctuations stay those of the half sample. Set it to 1.0 for the old
# behaviour (half-luminosity yields).
HALF_LUMI_FACTOR = 2.0


def _half_factor(half):
    """Yield scale of a half: HALF_LUMI_FACTOR for 'A' / 'B', 1 for 'all' / None."""
    return 1.0 if half in (None, 'all') else HALF_LUMI_FACTOR


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


def _unbiased_terms(s, b, w2_s, w2_n, eps_bkg):
    """Numerator and denominator of the excess-free estimator per bin.

    Raw counts:        s(s-1) / (n - 1),                n = s + b
    Weighted events:   (s^2 - W2_s) / (n - W2_n / n),   W2 = sum of squared weights
    The weighted form is s^2 minus its variance estimate over n times
    (1 - 1/n_eff), n_eff = n^2 / W2_n; for unit weights W2 = counts and it is
    exactly the raw-count form, for equal weights it is exact, and for PID-like
    weights scattered around 1 toys show no excess (see the module docstring).
    """
    s, b = np.asarray(s, float), np.asarray(b, float)
    w2_s, w2_n = np.asarray(w2_s, float), np.asarray(w2_n, float)
    n = s + b
    num = s * s - w2_s
    den = np.divide(n * n - w2_n, n, out=np.zeros(n.shape), where=n > 0) + (eps_bkg * b) ** 2
    return num, den


def _channel_result(ch, sel, eps_bkg, min_bin_count, count_on, estimator='unbiased', weights=None,
                    scale=1.0):
    """Z^2, S, B, the number of summed bins and the estimated excess for one channel.

    estimator: 'unbiased' (default) sums the excess-free (s^2 - W2_s)/(n - W2_n/n)
    (s(s-1)/(n-1) for raw counts); 'plain' sums s^2/(s+b+(eps b)^2) (module
    docstring).

    weights: optional extra per-selected-candidate multiplicities (bootstrap):
    an event drawn k times counts as k copies, also in the bin threshold.
    scale: factor on the MC weights (_half_factor for one half); the bin
    threshold still counts raw MC events.
    """
    mult = np.ones(len(sel)) if weights is None else np.asarray(weights, dtype=float)
    w = ch.w[sel] * scale * mult
    w2 = (ch.w[sel] * scale) ** 2 * mult        # k copies of an event with weight w
    bi = ch.binidx[sel]
    cls = ch.cls[sel]
    in_fit = bi >= 0
    sig = in_fit & (cls == 1)
    bkg = in_fit & (cls == 2)
    hist = lambda m, ww: np.bincount(bi[m], weights=ww[m], minlength=ch.n_bins).astype(float)
    s, b = hist(sig, w), hist(bkg, w)
    if count_on not in COUNT_ON:
        raise ValueError(f'count_on must be one of {COUNT_ON}.')
    counted = (sig | bkg) if count_on == 'fom' else in_fit
    n = hist(counted, mult)                     # raw MC counts for the threshold
    if estimator not in ESTIMATORS:
        raise ValueError(f'estimator must be one of {ESTIMATORS}.')
    denom = s + b + (eps_bkg * b) ** 2
    keep = (n >= min_bin_count) & (denom > 0)
    if estimator == 'plain':
        z2 = float(np.sum(s[keep] ** 2 / denom[keep]))
    else:
        keep &= hist(sig | bkg, mult) >= 2      # at least two events for the estimator
        num, den = _unbiased_terms(s, b, hist(sig, w2), hist(sig | bkg, w2), eps_bkg)
        keep &= den > 0
        z2 = float(np.sum(num[keep] / den[keep]))
    # Estimated MC-statistics excess of the PLAIN estimator (module docstring):
    # p(1-p) per kept bin, with p estimated by s/(s+b). Diagnostic only; with
    # estimator='unbiased' it is the amount that was removed.
    # The excess per bin is p(1-p) per raw event, i.e. p(1-p) * w for events of
    # weight w; with mixed weights the effective weight is W2_n / n.
    sb = s[keep] + b[keep]
    p_hat = np.divide(s[keep], sb, out=np.zeros_like(sb), where=sb > 0)
    w_eff = np.divide(hist(sig | bkg, w2)[keep], sb, out=np.zeros_like(sb), where=sb > 0)
    excess = float(np.sum(p_hat * (1.0 - p_hat) * w_eff))
    return z2, s.sum(), b.sum(), int(keep.sum()), excess


def evaluate_cut(data, theta, half='A', bcs_order='cut_then_bcs', bcs_base=None,
                 eps_bkg=0.0, min_bin_count=MIN_BIN_COUNT, count_on='fom',
                 estimator='unbiased', control_regions=None, yields=False, _masks=None):
    """Binned FOM of one working point.

    control_regions: None (default) -> Z is the statistical FOM. A list of
    ControlRegion -> Z = 1 / (sigma_S / S) from relative_uncertainty(), which
    adds each region's normalization and shape mismodeling; the breakdown
    (rel_unc*, cr_*) is added to the output and Z_stat keeps the statistical FOM.

    Returns a dict with Z, rel_unc (= sigma_S/S = 1/Z), Z_<channel>, n_bins (bins passing the threshold),
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
        z2, s, b, nb, exc = _channel_result(ch, sel, eps_bkg, min_bin_count, count_on, estimator,
                                            scale=_half_factor(half))
        out[f'Z_{name}'] = np.sqrt(z2)
        z2_tot += z2
        n_bins += nb
        excess_tot += exc
        s_tot += s
        b_tot += b
        if yields:
            sr = ch.in_sr[sel]
            y = np.bincount(ch.cat[sel][sr], weights=ch.w[sel][sr] * _half_factor(half),
                            minlength=len(ch.cat_names))
            for i, cname in enumerate(ch.cat_names):
                cat_yields[cname] = cat_yields.get(cname, 0.0) + y[i]
    out.update(Z=np.sqrt(z2_tot), n_bins=n_bins, S=s_tot, B=b_tot,
               excess_frac=excess_tot / z2_tot if z2_tot > 0 else np.nan)
    # relative uncertainty of the signal yield, statistical only (= 1/Z)
    out['rel_unc'] = out['rel_unc_stat'] = 1.0 / out['Z'] if out['Z'] > 0 else np.inf
    if control_regions:
        ru = relative_uncertainty(data, theta, control_regions, half=half, bcs_order=bcs_order,
                                  bcs_base=bcs_base, eps_bkg=eps_bkg, min_bin_count=min_bin_count,
                                  count_on=count_on, estimator=estimator)
        out.update(ru)
        out['Z_stat'] = out['Z']
        out['Z'] = 1.0 / ru['rel_unc'] if ru['rel_unc'] > 0 else 0.0
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


############################## data/MC mismodeling ###########################

excess_variance = util.excess_variance
mismodeling_label = util.mismodeling_label


############################## control regions ################################

# D_M windows as lists of (low, high) intervals
D_MASS_SR_WINDOW = [D_MASS_SR_RANGE]                       # util.D_MASS_SR
D_MASS_SB_WINDOW = [(-np.inf, 1.85), (1.9, np.inf)]        # util.D_MASS_SB
# Comparison binning per control region (coarse). The M2_miss and p_D + p_l
# spans differ between regions, so by default every region takes its range from
# its own events (bins='auto', auto_cr_bins): the 0.1 % and 99.9 % quantiles, so
# that a few far outliers cannot stretch the axis, and events beyond the range
# are folded into the first / last bin (fold_overflow=True), so that no event is
# lost. Each region keeps its binning (ControlRegion.bins) for everything it
# computes: R, delta, chi2/ndf, histograms and overflow fractions. The fixed
# binnings below remain available (bins=CR_BINS, ...).
CR_BINS = [util.MM2_bins_sb, util.p_D_l_bins_sb]           # D_M sidebands (= fit SB channel)
OFFRES_BINS = [np.linspace(-5, 10, 21), util.p_D_l_bins_sb]
CR_AUTO_NBINS = 20
# The automatic range is taken from the BCS candidate per event without a BDT
# cut (OPEN_CUT; D_M window after BCS, as in the histograms), i.e. one candidate
# per event, as the region's histograms contain at an open cut. It is fixed per
# region, so every working point is compared in the same bins; whether a tight
# cut pushes events beyond it shows in ControlRegion.overflow_fraction() (and in
# the title of the notebook's control-region plots).
# Optionally, several reference cuts can be given (ref_cuts=[...]); the union of
# their ranges is then used.
CR_AUTO_REF_CUTS = (OPEN_CUT,)
CR_AUTO_MIN_EVENTS = 200        # with several ref_cuts: one with fewer events (data + MC) is skipped


def auto_cr_bins(inputs, dm_window, quantiles=(0.001, 0.999), steps=(0.5, 0.1),
                 n=(CR_AUTO_NBINS, CR_AUTO_NBINS), ref_cuts=CR_AUTO_REF_CUTS,
                 ranges=(None, None), min_events=CR_AUTO_MIN_EVENTS, min_tail=5):
    """Edges in both fit variables covering the events of a region.

    By default (ref_cuts=CR_AUTO_REF_CUTS = (OPEN_CUT,)) the BCS candidate per
    event without a BDT cut is taken (BCS -> D_M window, as in ControlRegion's
    histograms), from data and MC together (unweighted). The range of each
    variable is given by the quantiles (so far outliers do not stretch it),
    rounded outward to steps = (M2_miss, p_D + p_l), n bins each.

    ref_cuts : reference cut(s), theta dicts or cut strings, a single one or a
               list. With several, the union of their ranges is used; cuts with
               fewer than min_events events are skipped (if all are, OPEN_CUT is
               used).

    In a small sample a 0.1 % tail is less than one event, so the quantiles are
    moved inward until at least min_tail events lie beyond each (a single
    outlier can then not set the edge). Computed once per region, so the binning
    is the same for every working point; events beyond it are folded into the
    edge bins by ControlRegion (fold_overflow=True), and
    ControlRegion.overflow_fraction() reports how many.

    ranges : optional fixed (low, high) per axis, e.g. (None, (0.0, 6.5)) keeps
             the automatic M2_miss range and fixes p_D + p_l to [0, 6.5].
    """
    if isinstance(ref_cuts, (dict, str)):
        ref_cuts = [ref_cuts]           # a single cut, e.g. ref_cuts=OPEN_CUT
    ref_cuts = list(ref_cuts) if ref_cuts else [OPEN_CUT]

    def values(theta):
        xs, ys = [], []
        for inp in inputs:
            for name, ch in inp.channels.items():
                sel = select(inp, name, theta, 'all', 'cut_then_bcs')
                inw = np.zeros(len(sel), dtype=bool)
                for lo, hi in dm_window:
                    inw |= (ch.dm[sel] > lo) & (ch.dm[sel] < hi)
                x, y = ch.x[sel][inw], ch.y[sel][inw]
                ok = np.isfinite(x) & np.isfinite(y)
                xs.append(x[ok])
                ys.append(y[ok])
        return np.concatenate(xs), np.concatenate(ys)

    samples = [values(as_theta(c)) for c in ref_cuts]
    used = [v for v in samples if len(v[0]) >= min_events]
    if not used:
        used = [values(OPEN_CUT)]
    edges = []
    for k, (step, nb, fallback) in enumerate(zip(steps, n, CR_BINS)):
        fixed = ranges[k] if ranges is not None else None
        if fixed is not None:
            lo, hi = map(float, fixed)
        else:
            qs = []
            for v in used:
                nv = len(v[k])
                if nv == 0:
                    continue
                q_lo = max(quantiles[0], min(min_tail / nv, 0.5))
                q_hi = min(quantiles[1], max(1.0 - min_tail / nv, 0.5))
                qs.append(np.quantile(v[k], (q_lo, q_hi)))
            if not qs:
                edges.append(np.asarray(fallback, float))
                continue
            lo, hi = min(q[0] for q in qs), max(q[1] for q in qs)
            lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
        if hi <= lo:
            hi = lo + step
        edges.append(np.linspace(lo, hi, nb + 1))
    return edges


class ControlRegion:
    """One control region: total data vs total MC after the same BDT cut.

    For every working point theta it measures (measure()):
        R, R_err : data/MC normalization ratio of the region and its stat. error
        delta    : shape mismodeling (excess variance after scaling MC by R),
                   clipped at 0 when the measured delta^2 is negative
        n        : normalization term, max(|R-1|, R_err) on all data
                   (norm_estimator='debiased', default; see measure for one
                   half), sqrt((R-1)^2 + R_err^2) with 'quadrature', R_err
                   alone with include_offset=False
    and it says which signal-region events these fractions apply to
    (applies_to, a query on the signal-region MC; only events in the
    statistical-term background, FOM_BACKGROUND, are counted). In the target the
    region contributes n * B_CR (normalization, correlated across bins) and
    delta * b_CR,i (shape, per bin), where B_CR is the signal-region yield of
    those events.

    data, mc   : FOMInput of the region's data and MC (whole events, all
                 candidates; the D_M window is applied after BCS, as in the fit).
                 They may be the same objects as the signal-region input (the D_M
                 sideband is part of the on-resonance samples).
    mc_scale   : extra factor on the MC weights. Default 1: the MC FOMInput is
                 already at data luminosity (FOMInput(..., lumi_scale=0.25)).
                 Only for an MC input built without lumi_scale pass L_data/L_MC.
    dm_window  : list of (low, high) D_M intervals defining the region.
    bins       : 'auto' (default; auto_cr_bins: M2_miss and p_D + p_l ranges from
                 the region's own data and MC, BCS candidate per event without a
                 BDT cut, 0.1 % - 99.9 % quantiles) or a pair of edge arrays (e.g.
                 CR_BINS, the fit's sideband-channel binning). Set once here and
                 used for every quantity of this region (self.bins).
    norm_estimator : 'debiased' (default): n = max(|R-1|, R_err) on all data
                 (see measure); 'quadrature': n = sqrt((R-1)^2 + R_err^2).
    split_halves : True (default): when the target is evaluated on half A or B,
                 the region compares data and MC of that half only (event-number
                 parity), so optimization (A) and validation (B) use independent
                 control-region data too; R_err is scaled to the full data set
                 (measure). False: always all data and MC.
    auto_kw    : options for auto_cr_bins, e.g. dict(ranges=(None, (0, 6.5))) to fix
                 the p_D + p_l range, or dict(ref_cuts=[...]) for other reference cuts.
    min_shape_bins : the shape (delta) needs at least this many bins with
                 >= min_count expected events; otherwise it is measured on the
                 M2_miss projection, and if that is too sparse too, delta = 0
                 (measure()['shape_dim'] says which: '2d', '1d', 'none').
    fold_overflow : True (default): events beyond the binning are counted in the
                 first / last bin of each axis, so every event of the region
                 enters R and delta; False: they are left out.
    channels   : 'combined' (sum e and mu before comparing; default) or
                 'separate' is not implemented yet.
    Use ControlRegion.fixed() for a region without samples (placeholder).
    """

    def __init__(self, name, data, mc, applies_to, mc_scale=1.0, dm_window=D_MASS_SR_WINDOW,
                 bins='auto', fold_overflow=True, include_offset=True, min_count=10,
                 min_shape_bins=10, auto_kw=None, split_halves=True, norm_estimator='debiased'):
        self.name = name
        self.data, self.mc = data, mc
        self.applies_to = applies_to
        self.mc_scale = float(mc_scale)
        self.dm_window = dm_window
        if isinstance(bins, str):
            if bins != 'auto':
                raise ValueError("bins must be a pair of edge arrays or 'auto'.")
            bins = auto_cr_bins([data, mc], dm_window, **(auto_kw or {})) if data is not None else CR_BINS
        self.bins = [np.asarray(b, float) for b in bins]
        self.fold_overflow = bool(fold_overflow)
        self.include_offset = include_offset
        self.min_count = min_count
        self.min_shape_bins = int(min_shape_bins)
        self.split_halves = bool(split_halves)
        if norm_estimator not in ('debiased', 'quadrature'):
            raise ValueError("norm_estimator must be 'debiased' or 'quadrature'.")
        self.norm_estimator = norm_estimator
        self._fixed = None
        self._cache = {}
        self._sr_masks = {}

    @classmethod
    def fixed(cls, name, norm, shape, applies_to):
        """Placeholder region with hand-set fractions (e.g. combinatorial for now)."""
        cr = cls(name, None, None, applies_to)
        cr._fixed = dict(R=np.nan, R_err=np.nan, n=float(norm), delta=float(shape),
                         delta2=float(shape) ** 2, delta2_err=np.nan, chi2=np.nan, ndf=0,
                         n_data=np.nan, n_mc=np.nan, shape_dim='fixed')
        return cr

    # -- which signal-region events the region describes
    def sr_mask(self, ch):
        """Bool per candidate of a signal-region channel: matches applies_to."""
        key = id(ch)
        if key not in self._sr_masks:
            frame = pd.DataFrame(dict(category=np.array(ch.cat_names, dtype=object)[ch.cat], **ch.truth))
            try:
                mask = frame.eval(self.applies_to).to_numpy(dtype=bool)
            except Exception as err:
                raise ValueError(f'{self.name}: cannot evaluate applies_to "{self.applies_to}" '
                                 f'on the signal-region MC (columns {list(frame.columns)}). '
                                 'B0_isContinuumEvent is kept by load_candidates for MC.') from err
            self._sr_masks[key] = mask
        return self._sr_masks[key]

    # -- data/MC comparison at one working point
    def _cr_bins(self, ch):
        """Flat control-region bin per candidate (-1 outside the D_M window, or
        outside the binning when fold_overflow=False), computed once per channel."""
        key = ('bins', id(ch))
        if key not in self._sr_masks:
            bx, by = self.bins
            ix = np.searchsorted(bx, ch.x, side='right') - 1
            iy = np.searchsorted(by, ch.y, side='right') - 1
            ix[ch.x == bx[-1]] = len(bx) - 2
            iy[ch.y == by[-1]] = len(by) - 2
            nx, ny = len(bx) - 1, len(by) - 1
            # which side of the binning each candidate lies beyond (before folding):
            # bit 0/1 = M2_miss below/above, bit 2/3 = p_D + p_l below/above
            self._sr_masks[('side', id(ch))] = ((ix < 0) | 2 * (ix >= nx)
                                                | 4 * (iy < 0) | 8 * (iy >= ny)).astype(np.int8)
            if self.fold_overflow:
                # events beyond the range go to the first / last bin (data and MC
                # alike), so no event of the region is lost to the binning
                ix, iy = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(ch.x) & np.isfinite(ch.y)
            inw = np.zeros(len(ch), dtype=bool)
            for lo, hi in self.dm_window:
                inw |= (ch.dm > lo) & (ch.dm < hi)
            self._sr_masks[key] = np.where(ok & inw, ix * ny + iy, -1).astype(np.int64)
        return self._sr_masks[key]

    def _region_half(self, half):
        """Half of the region's data and MC to use: the signal-region half if
        split_halves, else always 'all'."""
        return half if (self.split_halves and half in ('A', 'B')) else 'all'

    def _hist(self, inp, theta, bcs_order, bcs_base, half='all'):
        nb = (len(self.bins[0]) - 1) * (len(self.bins[1]) - 1)
        h, hw2 = np.zeros(nb), np.zeros(nb)
        for name, ch in inp.channels.items():
            sel = select(inp, name, theta, half, bcs_order, bcs_base)
            bi = self._cr_bins(ch)[sel]
            ok = bi >= 0
            w = ch.w[sel][ok]
            h += np.bincount(bi[ok], weights=w, minlength=nb)
            hw2 += np.bincount(bi[ok], weights=w ** 2, minlength=nb)
        shape = (len(self.bins[0]) - 1, len(self.bins[1]) - 1)
        return h.reshape(shape), hw2.reshape(shape)

    def measure(self, theta, bcs_order='cut_then_bcs', bcs_base=None, half='all'):
        """R, R_err, n, delta, chi2/ndf, ... at one working point.

        half: 'all' (default) uses all data and MC of the region. 'A' / 'B'
        (passed by relative_uncertainty with the signal-region half) uses only
        the events of that half in data AND MC (event-number parity), if
        split_halves=True: optimizing on A and validating on B then uses
        independent data in the control regions as well. Both sides hold half
        of the luminosity, so R and delta need no correction; R_err, the
        precision of the half, is scaled by 1/sqrt(HALF_LUMI_FACTOR) to that
        expected for the full data set (as the signal-region yields are);
        'R_err_half' keeps the measured one. n_data and n_mc are the counts in
        the half.

        Normalization term (norm_estimator='debiased', default):
            n^2 = max((R-1)^2 - R_err_half^2, 0) + R_err^2
        The measured (R-1)^2 contains the scatter of the sample it is measured
        on, R_err_half^2 (= R_err^2 on all data), which is subtracted (as the
        excess variance does for delta); the precision expected for the full
        data, R_err^2, is then counted once. On all data this is
        n = max(|R-1|, R_err). 'quadrature': n^2 = (R-1)^2 + R_err^2 (the
        scatter counted twice on average)."""
        if self._fixed is not None:
            return dict(self._fixed)
        theta = as_theta(theta)
        half = self._region_half(half)
        key = (tuple(theta[v] for v in MVA_OUTPUTS), bcs_order,
               None if bcs_base is None else theta_to_cut(as_theta(bcs_base)), half)
        if key not in self._cache:
            d, _ = self._hist(self.data, theta, bcs_order, bcs_base, half)
            m, mw2 = self._hist(self.mc, theta, bcs_order, bcs_base, half)
            m, mw2 = m * self.mc_scale, mw2 * self.mc_scale ** 2
            r = self._shape_measurement(d, m, mw2)
            r_err_half = r['R_err']
            r_err = r_err_half / np.sqrt(_half_factor(half))
            off = (r['R'] - 1.0) if (self.include_offset and np.isfinite(r['R'])) else 0.0
            if not np.isfinite(r_err):
                n = 1.0
            elif self.norm_estimator == 'quadrature':
                n = np.sqrt(off ** 2 + r_err ** 2)
            else:
                # (R-1)^2 measured on this sample contains its scatter r_err_half^2:
                # subtract it, then add the precision expected for the full data once
                n = np.sqrt(max(off ** 2 - r_err_half ** 2, 0.0) + r_err ** 2) if off else r_err
            delta = r['delta'] if np.isfinite(r['delta']) else 0.0
            self._cache[key] = dict(r, R_err=r_err, R_err_half=r_err_half, half=half,
                                    n=n, delta=delta, n_data=d.sum(), n_mc=m.sum())
        return dict(self._cache[key])

    def _shape_measurement(self, d, m, mw2):
        """excess_variance on the 2D histograms; if fewer than min_shape_bins
        (default 10; ndf = 9, so chi2 itself fluctuates by ~ +-47 %) bins are
        usable (too sparse, e.g. off-resonance after a tight cut), on
        the M2_miss projection instead; if that is too sparse as well, no shape
        measurement (delta = 0). 'shape_dim' records which: '2d', '1d', 'none'.
        The normalization R is the same in all cases (totals)."""
        kw = dict(normalize=True, min_count=self.min_count)
        r = excess_variance(d, m, model_err=np.sqrt(mw2), **kw)
        if np.isfinite(r['delta2']) and r['ndf'] + 1 >= self.min_shape_bins:
            return dict(r, shape_dim='2d')
        r1 = excess_variance(d.sum(axis=1), m.sum(axis=1), model_err=np.sqrt(mw2.sum(axis=1)), **kw)
        if np.isfinite(r1['delta2']) and r1['ndf'] + 1 >= self.min_shape_bins:
            return dict(r1, shape_dim='1d')
        return dict(r1, shape_dim='none')

    def overflow_fraction(self, theta, bcs_order='cut_then_bcs', bcs_base=None):
        """Fraction of the region's events beyond the binning on each side, for data
        and MC (weighted): {'data': {'M2_miss low': f, 'M2_miss high': f,
        'p_D_l low': f, 'p_D_l high': f}, 'mc': {...}}. With fold_overflow=True
        these events sit in the edge bins; a few % or more means the range is too
        narrow for this working point (see auto_cr_bins)."""
        theta = as_theta(theta)
        sides = (('M2_miss low', 1), ('M2_miss high', 2), ('p_D_l low', 4), ('p_D_l high', 8))
        out = {}
        for label, inp in (('data', self.data), ('mc', self.mc)):
            tot, beyond = 0.0, np.zeros(len(sides))
            for name, ch in inp.channels.items():
                sel = select(inp, name, theta, 'all', bcs_order, bcs_base)
                self._cr_bins(ch)         # fills the per-candidate side flags
                inw = np.zeros(len(sel), dtype=bool)
                for lo, hi in self.dm_window:
                    inw |= (ch.dm[sel] > lo) & (ch.dm[sel] < hi)
                w = ch.w[sel]
                side = self._sr_masks[('side', id(ch))][sel]
                fin = np.isfinite(ch.x[sel]) & np.isfinite(ch.y[sel])
                region = inw & fin        # every event of the region, counted or not
                tot += w[region].sum()
                for k, (_, bit) in enumerate(sides):
                    beyond[k] += w[region & ((side & bit) > 0)].sum()
            out[label] = {s: (beyond[k] / tot if tot > 0 else np.nan) for k, (s, _) in enumerate(sides)}
        return out

    def histograms(self, theta, bcs_order='cut_then_bcs', bcs_base=None, half='all'):
        """(data, MC scaled to data, MC error) histograms, e.g. for residual plots
        (half: 'all' by default; 'A' / 'B' as in measure)."""
        half = self._region_half(half)
        d, _ = self._hist(self.data, as_theta(theta), bcs_order, bcs_base, half)
        m, mw2 = self._hist(self.mc, as_theta(theta), bcs_order, bcs_base, half)
        return d, m * self.mc_scale, np.sqrt(mw2) * self.mc_scale


def d_mass_sideband_cr(data_onres, mc_onres, mc_scale=1.0, **kw):
    """D_M sidebands of the on-resonance samples; describes the signal-region
    fake D from BB events (bkg_fakeD with B0_isContinuumEvent == 0). Fake D from
    qq events are described by the off-resonance region, with all other qq
    events (partition by event origin, so no signal-region event is counted by
    both regions). The sidebands' own events include ~7 % qq fake D."""
    return ControlRegion('D_mass_sidebands', data_onres, mc_onres,
                         applies_to="category == 'bkg_fakeD' and B0_isContinuumEvent == 0",
                         mc_scale=mc_scale,
                         dm_window=D_MASS_SB_WINDOW, **kw)


def offres_cr(data_offres, mc_offres, mc_scale=1.0, dm_window=D_MASS_SR_WINDOW,
              bins='auto', **kw):
    """Off-resonance sample (D_M signal window by default); describes all
    signal-region qq events (B0_isContinuumEvent == 1: continuum background with
    a real D and qq fake D), as its own events are; fake D from BB events are
    described by the D_M sidebands (partition by event origin, so no
    signal-region event is counted by both regions). Its M2_miss
    extends lower than on resonance; the automatic binning follows it
    (OFFRES_BINS = [-5, 10] as a fixed alternative)."""
    return ControlRegion('offres', data_offres, mc_offres,
                         applies_to='B0_isContinuumEvent == 1', mc_scale=mc_scale,
                         dm_window=dm_window, bins=bins, **kw)


def combinatorial_placeholder(norm=0.0, shape=0.0):
    """BDT combinatorial sideband, not measured yet: hand-set fractions."""
    return ControlRegion.fixed('combinatorial', norm, shape,
                               applies_to="category == 'bkg_combinatorial'")


def ratio_slices(cr, base_cut, var, edges=None, half='all', bcs_order='cut_then_bcs', bcs_base=None):
    """Data/MC ratio of a control region in slices of one BDT output.

    Does the region's data/MC ratio depend on the output that is cut on? If it
    does, the MC models that output poorly in the region, and the target's
    optimum sits where the MC happens to agree with the data.

    Events are selected as everywhere (MVA cut -> BCS -> D_M window), with the
    other outputs at base_cut and `var` cut at the loosest edge (i.e. open over
    the scanned range); each event goes into the slice of its selected
    candidate's `var` value. The slices are disjoint and add up to the region at
    the loosest cut.

    edges : slice edges of var (default 0 to 1 in steps of 0.05; trailing slices
            without data and MC are dropped).
    half  : 'all' (default), or 'A' / 'B' (with cr.split_halves).

    Returns a DataFrame per slice: lo, hi, n_data, n_mc (scaled to data),
    mc_err, R = n_data / n_mc, R_err, pull = (R - 1) / R_err and
    pull_shape = (R - R_total) / R_err. R_err uses the data count expected from
    the region's total ratio (R_total * n_mc, Pearson, as excess_variance) and
    the MC statistics, so empty slices still get an error. df.attrs holds
    R_total (all slices together), chi2_shape and ndf_shape (constancy of R
    over slices with n_mc >= cr.min_count) and the cut used.
    """
    if cr.data is None:
        raise ValueError(f'{cr.name}: no samples (placeholder region).')
    edges = np.round(np.arange(0.0, 1.0001, 0.05), 2) if edges is None else np.asarray(edges, float)
    th = as_theta(base_cut)
    th[var] = float(edges[-1] if CUT_DIRECTIONS[var] == '<' else edges[0])
    half = cr._region_half(half)
    nb = len(edges) - 1
    tot = {}
    for label, inp in (('data', cr.data), ('mc', cr.mc)):
        h, h2 = np.zeros(nb), np.zeros(nb)
        for name, ch in inp.channels.items():
            sel = select(inp, name, th, half, bcs_order, bcs_base)
            sel = sel[cr._cr_bins(ch)[sel] >= 0]            # the region's events, as in its histograms
            k = np.searchsorted(edges, ch.probs[var][sel], side='right') - 1
            k[ch.probs[var][sel] == edges[-1]] = nb - 1     # last slice closed on the right
            ok = (k >= 0) & (k < nb)
            w = ch.w[sel][ok]
            h += np.bincount(k[ok], weights=w, minlength=nb)
            h2 += np.bincount(k[ok], weights=w ** 2, minlength=nb)
        tot[label] = (h, h2)
    d = tot['data'][0]
    m, m_err = tot['mc'][0] * cr.mc_scale, np.sqrt(tot['mc'][1]) * cr.mc_scale
    r_tot = d.sum() / m.sum() if m.sum() > 0 else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.where(m > 0, d / m, np.nan)
        # data error from the expected count r_tot * m (Pearson), plus the MC error
        R_err = np.where(m > 0, np.sqrt(r_tot * m + (r_tot * m_err) ** 2) / m, np.nan)
    out = pd.DataFrame(dict(lo=edges[:-1], hi=edges[1:], n_data=d, n_mc=m, mc_err=m_err,
                            R=R, R_err=R_err, pull=(R - 1.0) / R_err, pull_shape=(R - r_tot) / R_err))
    filled = np.flatnonzero((d > 0) | (m > 0))
    out = out.iloc[:filled[-1] + 1] if len(filled) else out.iloc[:0]
    used = (out['n_mc'] >= cr.min_count) & np.isfinite(out['pull_shape'])
    out.attrs.update(region=cr.name, var=var, cut=theta_to_cut(th), half=half, R_total=r_tot,
                     chi2_shape=float((out.loc[used, 'pull_shape'] ** 2).sum()),
                     ndf_shape=int(max(used.sum() - 1, 0)))
    return out


def plot_ratio_slices(tables, save=None):
    """Data and MC per slice (top) and their ratio (bottom), one column per
    ratio_slices table. tables: {label: DataFrame} or a list of DataFrames."""
    if isinstance(tables, pd.DataFrame):
        tables = [tables]
    if not isinstance(tables, dict):
        tables = {f"{t.attrs.get('region', '')}: {t.attrs.get('var', '')}": t for t in tables}
    n = len(tables)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 5.6), squeeze=False,
                             gridspec_kw=dict(height_ratios=[2, 1.3]), sharex='col')
    for j, (label, t) in enumerate(tables.items()):
        ax, axr = axes[0, j], axes[1, j]
        x = 0.5 * (t['lo'] + t['hi'])
        edges = np.r_[t['lo'].to_numpy(), t['hi'].to_numpy()[-1:]]
        ax.stairs(t['n_mc'], edges, color='C0', label='MC (scaled to data lumi)')
        ax.bar(x, 2 * t['mc_err'], width=t['hi'] - t['lo'], bottom=t['n_mc'] - t['mc_err'],
               color='C0', alpha=0.25, linewidth=0)
        ax.errorbar(x, t['n_data'], yerr=np.sqrt(t['n_data']), fmt='.', color='k', label='data')
        ax.set_ylabel('events per slice')
        ax.set_title(f"{label}\n{t.attrs.get('cut', '')}", fontsize=9)
        ax.legend(fontsize=8)
        a = t.attrs
        axr.errorbar(x, t['R'], yerr=t['R_err'], fmt='.', color='k',
                     label=(f"$R_\\mathrm{{total}}$ = {a.get('R_total', np.nan):.3f}, "
                            f"$\\chi^2$/ndf (R const.) = {a.get('chi2_shape', np.nan):.0f}/{a.get('ndf_shape', 0)}"))
        axr.axhline(1.0, color='gray', ls='--', lw=1)
        axr.axhline(a.get('R_total', np.nan), color='C1', ls=':', lw=1)
        axr.set_ylabel('data / MC')
        axr.set_xlabel(f"{a.get('var', '')} of the selected candidate")
        axr.legend(fontsize=8)
    fig.suptitle('data/MC per slice of one BDT output (others at the base cut)', fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


############################## target with control regions ###################

def _fisher_inputs(data, theta, crs, half, bcs_order, bcs_base, eps_bkg, min_bin_count,
                   count_on, estimator, weights=None):
    """Per-bin arrays of the kept signal-region bins, all channels concatenated:
    s, b (weighted yields), n (raw counts), b_cr[k] (background of CR k),
    w2_sh[k][l] (sum of squared weights of background in both k and l),
    w2_s, w2_n (sum of squared weights of signal, of signal + background)."""
    K = len(crs)
    S, B, N, W2S, W2N = [], [], [], [], []
    BCR, SH = [[] for _ in crs], [[[] for _ in crs] for _ in crs]
    for name, ch in data.channels.items():
        sel = select(data, name, theta, half, bcs_order, bcs_base)
        mult = np.ones(len(sel)) if weights is None else np.asarray(weights[name], float)
        wh = ch.w[sel] * _half_factor(half)         # yields at full data luminosity
        w, w2 = wh * mult, wh ** 2 * mult
        bi, cls = ch.binidx[sel], ch.cls[sel]
        in_fit = bi >= 0
        sig, bkg = in_fit & (cls == 1), in_fit & (cls == 2)
        hist = lambda m, ww: np.bincount(bi[m], weights=ww[m], minlength=ch.n_bins).astype(float)
        s, b = hist(sig, w), hist(bkg, w)
        counted = (sig | bkg) if count_on == 'fom' else in_fit
        n = hist(counted, mult)
        denom = s + b + (eps_bkg * b) ** 2
        keep = (n >= min_bin_count) & (denom > 0)
        if estimator == 'unbiased':
            keep &= hist(sig | bkg, mult) >= 2
        masks = [bkg & cr.sr_mask(ch)[sel] for cr in crs]
        S.append(s[keep]); B.append(b[keep]); N.append(n[keep])
        W2S.append(hist(sig, w2)[keep]); W2N.append(hist(sig | bkg, w2)[keep])
        for k, mk in enumerate(masks):
            BCR[k].append(hist(mk, w)[keep])
            for l, ml in enumerate(masks):
                SH[k][l].append(hist(mk & ml, w2)[keep])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    return (cat(S), cat(B), cat(N), [cat(x) for x in BCR],
            [[cat(x) for x in row] for row in SH], cat(W2S), cat(W2N))


def _sigma_mu(s, b, b_cr, b_sh, norms, deltas, eps_bkg, estimator, use_norm, use_shape,
              w2_s=None, w2_n=None):
    """sqrt((F^-1)_mumu) for nu_i = mu s_i + b_i + sum_k alpha_k b_cr,k,i.

    use_norm[k]  : alpha_k is a nuisance with Gaussian prior width norms[k]
                   (otherwise fixed at 0, i.e. the normalization is known).
    use_shape[k] : add (deltas[k] * b_cr,k,i)^2 to the variance of every bin.
    b_sh[k][l]   : sum of squared weights of background in both k and l
                   (= event counts for raw MC).
    w2_s, w2_n   : sum of squared weights of signal, of signal + background
                   (default: raw counts, s and s + b).
    estimator='unbiased' uses the excess-free forms: s^2 -> s^2 - W2_s,
    b_k b_l -> b_k b_l - W2_kl, variance s + b -> n - W2_n / n (for raw counts
    s(s-1), b_k b_l - b_kl and n - 1); 'plain' uses s^2, b_k b_l and s + b.
    """
    if len(s) == 0:
        return np.inf
    unb = estimator == 'unbiased'
    w2_s = s if w2_s is None else w2_s
    w2_n = s + b if w2_n is None else w2_n
    if unb:
        ss, V = _unbiased_terms(s, b, w2_s, w2_n, eps_bkg)
    else:
        ss, V = s ** 2, s + b + (eps_bkg * b) ** 2
    for k in range(len(b_cr)):
        if use_shape[k]:
            V = V + (deltas[k] * b_cr[k]) ** 2
    ok = V > 0
    nuis = [k for k in range(len(b_cr)) if use_norm[k] and norms[k] > 0]
    F = np.zeros((1 + len(nuis), 1 + len(nuis)))
    F[0, 0] = np.sum(ss[ok] / V[ok])
    for a, k in enumerate(nuis, start=1):
        F[0, a] = F[a, 0] = np.sum(s[ok] * b_cr[k][ok] / V[ok])
        for c, l in enumerate(nuis, start=1):
            prod = b_cr[k] * b_cr[l] - (b_sh[k][l] if unb else 0.0)
            F[a, c] = np.sum(prod[ok] / V[ok])
        F[a, a] += 1.0 / norms[k] ** 2
    if F[0, 0] <= 0:
        return np.inf
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return np.inf
    return float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.inf


def relative_uncertainty(data, theta, control_regions, half='A', bcs_order='cut_then_bcs',
                         bcs_base=None, eps_bkg=0.0, min_bin_count=MIN_BIN_COUNT,
                         count_on='fom', estimator='unbiased', weights=None):
    """Expected relative uncertainty of the D tau nu yield, sigma_S / S, with the
    control-region systematics, and its breakdown.

    Model (signal-region fit bins, e and mu together): nu_i = mu s_i + b_i, with
    b = the statistical-term background (FOM_BACKGROUND). Each control region k
    adds a normalization nuisance alpha_k with prior width n_k acting on b_cr,k
    (the background events it describes, applies_to) and a per-bin shape term
    (delta_k b_cr,k,i)^2. sigma_S/S = sigma(mu) = sqrt((F^-1)_mumu), F the Fisher
    matrix (the curvature of the Asimov likelihood). In one bin this is
        (sigma_S/S)^2 = (S + B)/S^2 + sum_k (n_k^2 + delta_k^2) (B_cr,k / S)^2 .
    Without control regions it equals 1/Z of the statistical FOM.

    Returns dict with rel_unc, rel_unc_stat, rel_unc_norm, rel_unc_shape
    (quadrature differences: stat -> +norm -> +shape) and per region
    cr_<name>_{R, R_err, n, delta, B, impact}; impact = sqrt(total^2 - total
    without that region^2).
    """
    theta = as_theta(theta)
    crs = list(control_regions)
    meas = [cr.measure(theta, bcs_order, bcs_base, half=half) for cr in crs]
    norms = [m['n'] for m in meas]
    deltas = [m['delta'] for m in meas]
    s, b, n, b_cr, b_sh, w2_s, w2_n = _fisher_inputs(data, theta, crs, half, bcs_order, bcs_base,
                                                     eps_bkg, min_bin_count, count_on, estimator,
                                                     weights)
    K = len(crs)
    sig = lambda un, us: _sigma_mu(s, b, b_cr, b_sh, norms, deltas, eps_bkg, estimator, un, us,
                                   w2_s, w2_n)
    none, all_ = [False] * K, [True] * K
    stat = sig(none, none)
    with_norm = sig(all_, none)
    total = sig(all_, all_)
    q = lambda hi, lo: float(np.sqrt(max(hi ** 2 - lo ** 2, 0.0))) if np.isfinite(hi) else np.inf
    out = dict(rel_unc=total, rel_unc_stat=stat, rel_unc_norm=q(with_norm, stat),
               rel_unc_shape=q(total, with_norm))
    for k, (cr, m) in enumerate(zip(crs, meas)):
        without = [j != k for j in range(K)]
        out.update({f'cr_{cr.name}_R': m['R'], f'cr_{cr.name}_R_err': m['R_err'],
                    f'cr_{cr.name}_n': m['n'], f'cr_{cr.name}_delta': m['delta'],
                    f'cr_{cr.name}_shape_dim': m.get('shape_dim'),
                    f'cr_{cr.name}_B': float(b_cr[k].sum()),
                    f'cr_{cr.name}_impact': q(total, sig(without, without))})
    return out


############################## scans ##########################################

GRID_DECIMALS = 2


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
        summary.append(dict(start=label, Z=z, rel_unc=1.0 / z if z > 0 else np.inf,
                            restarts=n_restart, cut=theta_to_cut(theta), **theta))
    summary = pd.DataFrame(summary).sort_values('Z', ascending=False, ignore_index=True)
    best = thetas[summary['start'].iloc[0]]
    return dict(summary=summary, paths=pd.concat(paths, ignore_index=True),
                thetas=thetas, best_theta=best, grids=grids)


############################## Optuna (alternative optimizers) ################
#
# Cross-check of the coordinate descent, and the statistics-vs-systematics
# trade-off. Thresholds are sampled as indices into the same grids, so every
# trial is a round working point and the results feed the existing tools
# (choose_working_points, validation_report, plots) unchanged.

def _optuna():
    try:
        import optuna
    except ImportError as err:
        raise ImportError('run_optuna needs optuna (pip install optuna; CMA-ES also needs cmaes).') from err
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _grid_lists(data, grids, half):
    grids = grids or make_grids(data, half=half)
    return {v: np.unique(np.asarray(grids[v], float)) for v in MVA_OUTPUTS}


def _theta_from_trial(trial, glist):
    return {v: float(glist[v][trial.suggest_int(v, 0, len(glist[v]) - 1)]) for v in MVA_OUTPUTS}


def _grid_index(glist, theta):
    """Nearest grid index of each threshold (to enqueue a starting point)."""
    theta = as_theta(theta)
    return {v: int(np.argmin(np.abs(glist[v] - theta[v]))) for v in MVA_OUTPUTS}


def _make_sampler(optuna, sampler, seed, multi=False):
    if sampler == 'tpe':
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    if sampler == 'cmaes' and not multi:
        return optuna.samplers.CmaEsSampler(seed=seed)
    if sampler == 'nsga2' and multi:
        return optuna.samplers.NSGAIISampler(seed=seed)
    raise ValueError(f"sampler {sampler!r} not available here: single objective 'tpe' or 'cmaes', "
                     "two objectives 'nsga2' or 'tpe'.")


def run_optuna(data, n_trials=500, sampler='tpe', grids=None, starts=None, seed=0, top_k=10,
               validate_half='B', n_boot=50, verbose=True, **eval_kw):
    """Maximize Z (same target as run_optimization) with Optuna.

    sampler : 'tpe' (default, multivariate TPE) or 'cmaes' (needs the cmaes package)
    starts  : working points tried first (default open, lgb_tight, lgb_loose)
    Selection: the single best trial is mostly the luckiest MC fluctuation, so
    the top_k distinct trials are re-evaluated on validate_half (default 'B',
    independent of the optimization half) and bootstrapped on the optimization
    half; best_theta is the one with the highest Z on validate_half. Set
    validate_half=None to take the best trial on the optimization half.
    **eval_kw: as for evaluate_cut (half, control_regions, estimator, ...).
    Returns dict(summary (top_k table), best_theta, grids, trials, study).
    """
    optuna = _optuna()
    half = eval_kw.get('half', 'A')
    glist = _grid_lists(data, grids, half)
    starts = starts if starts is not None else {'open': OPEN_CUT, 'lgb_tight': util.lgb_tight,
                                                'lgb_loose': util.lgb_loose}
    cache = {}

    def evaluate(theta):
        key = tuple(theta[v] for v in MVA_OUTPUTS)
        if key not in cache:
            cache[key] = evaluate_cut(data, theta, **eval_kw)
        return cache[key]

    def objective(trial):
        r = evaluate(_theta_from_trial(trial, glist))
        trial.set_user_attr('rel_unc', float(r['rel_unc']))
        return float(r['Z'])

    study = optuna.create_study(direction='maximize', sampler=_make_sampler(optuna, sampler, seed))
    for th in starts.values():
        study.enqueue_trial(_grid_index(glist, th))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials = _trials_frame(study, glist)
    top = (trials.sort_values('Z', ascending=False)
           .drop_duplicates(subset=list(MVA_OUTPUTS)).head(top_k))
    rows = []
    for _, t in top.iterrows():
        theta = {v: t[v] for v in MVA_OUTPUTS}
        row = dict(cut=theta_to_cut(theta), **theta, Z_opt=t['Z'], rel_unc_opt=t['rel_unc'])
        _, row['sigma_Z_opt'], _ = bootstrap_z(data, theta, n_boot=n_boot, **eval_kw)
        if validate_half is not None:
            rv = evaluate_cut(data, theta, **dict(eval_kw, half=validate_half))
            row.update(Z_val=rv['Z'], rel_unc_val=rv['rel_unc'])
        rows.append(row)
    summary = pd.DataFrame(rows)
    key = 'Z_val' if validate_half is not None else 'Z_opt'
    summary = summary.sort_values(key, ascending=False, ignore_index=True)
    best = {v: float(summary[v].iloc[0]) for v in MVA_OUTPUTS}
    if verbose:
        print(f'[optuna {sampler}] {n_trials} trials, {len(cache)} distinct working points; '
              f'chosen by {key}: {theta_to_cut(best)}  Z_opt={summary["Z_opt"].iloc[0]:.4f}'
              + (f'  Z_val={summary["Z_val"].iloc[0]:.4f}' if validate_half else ''))
    return dict(summary=summary, best_theta=best, grids={v: glist[v] for v in MVA_OUTPUTS},
                trials=trials, study=study)


def _trials_frame(study, glist):
    rows = []
    for t in study.trials:
        if t.state.name != 'COMPLETE':
            continue
        row = {v: float(glist[v][t.params[v]]) for v in MVA_OUTPUTS}
        row['number'] = t.number
        if t.values is not None and len(t.values) == 1:
            row['Z'] = t.values[0]
        row.update(t.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def run_optuna_pareto(data, control_regions, n_trials=500, sampler='nsga2', grids=None,
                      starts=None, seed=0, verbose=True, **eval_kw):
    """Statistics-vs-systematics trade-off: minimize the statistical part and the
    control-region part of sigma_S/S at the same time (two objectives).

    The result is the Pareto front: working points where neither part can be
    lowered without raising the other. The sigma_S/S optimum lies on it (total^2
    = stat^2 + syst^2 for every point, syst^2 = norm^2 + shape^2). The search
    spreads its trials along the whole front, so its minimum-total point is
    usually a little worse than run_optimization / run_optuna: use this for the
    trade-off picture, those for the working point. Pass the known optima in
    `starts` (e.g. {'CD': res_cr['best_theta']}) so the front includes them.
    sampler: 'nsga2' (default) or 'tpe'.
    Returns dict(front, all (every evaluated point), best_theta (minimum total on
    the front), grids, study). Columns: thresholds, cut, rel_unc_stat,
    rel_unc_syst, rel_unc (total), Z.
    """
    if not control_regions:
        raise ValueError('The trade-off needs control_regions (otherwise there is no systematic part).')
    optuna = _optuna()
    half = eval_kw.get('half', 'A')
    glist = _grid_lists(data, grids, half)
    starts = starts if starts is not None else {'open': OPEN_CUT, 'lgb_tight': util.lgb_tight,
                                                'lgb_loose': util.lgb_loose}
    cache = {}

    def objective(trial):
        theta = _theta_from_trial(trial, glist)
        key = tuple(theta[v] for v in MVA_OUTPUTS)
        if key not in cache:
            cache[key] = evaluate_cut(data, theta, control_regions=control_regions, **eval_kw)
        r = cache[key]
        syst = float(np.hypot(r['rel_unc_norm'], r['rel_unc_shape']))
        stat = float(r['rel_unc_stat'])
        if not (np.isfinite(stat) and np.isfinite(syst)):
            stat = syst = 1e6                       # no bin passes: worst possible
        return stat, syst

    study = optuna.create_study(directions=['minimize', 'minimize'],
                                sampler=_make_sampler(optuna, sampler, seed, multi=True))
    for th in starts.values():
        study.enqueue_trial(_grid_index(glist, th))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    def frame(trials):
        rows = []
        for t in trials:
            theta = {v: float(glist[v][t.params[v]]) for v in MVA_OUTPUTS}
            stat, syst = t.values
            if stat >= 1e6:
                continue
            rows.append(dict(**theta, cut=theta_to_cut(theta), rel_unc_stat=stat, rel_unc_syst=syst,
                             rel_unc=float(np.hypot(stat, syst)), Z=1.0 / float(np.hypot(stat, syst))))
        return pd.DataFrame(rows).drop_duplicates(subset=list(MVA_OUTPUTS)).reset_index(drop=True)

    done = [t for t in study.trials if t.state.name == 'COMPLETE']
    front = frame(study.best_trials)
    if front.empty:
        raise RuntimeError('run_optuna_pareto: no trial gave a finite sigma_S/S (no fit bin with '
                           f'>= min_bin_count MC events at any sampled cut). Increase n_trials or check '
                           'the input sample.')
    front = front.sort_values('rel_unc_stat', ignore_index=True)
    allpts = frame(done)
    i = int(front['rel_unc'].idxmin())
    best = {v: float(front[v].iloc[i]) for v in MVA_OUTPUTS}
    if verbose:
        print(f'[optuna pareto {sampler}] {n_trials} trials, {len(front)} points on the front; '
              f'minimum sigma_S/S {100 * front["rel_unc"].iloc[i]:.2f}% at {theta_to_cut(best)}')
    return dict(front=front, all=allpts, best_theta=best,
                grids={v: glist[v] for v in MVA_OUTPUTS}, study=study)


def plot_pareto(result, data=None, marks=None, control_regions=None, save=None, **eval_kw):
    """Statistical vs systematic part of sigma_S/S: every evaluated working point
    (grey), the Pareto front (line), circles of constant total sigma_S/S
    (dashed), the minimum-total point, and optional working points `marks`
    ({label: cut}, evaluated with data / control_regions / eval_kw)."""
    front, allpts = result['front'], result['all']
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(100 * allpts['rel_unc_stat'], 100 * allpts['rel_unc_syst'], '.', color='#c3c2b7', ms=4,
            label='evaluated working points', zorder=1)
    ax.plot(100 * front['rel_unc_stat'], 100 * front['rel_unc_syst'], 'o-', color='#2a78d6', ms=5,
            lw=2, label='Pareto front', zorder=2)
    i = int(front['rel_unc'].idxmin())
    best_total = 100 * front['rel_unc'].iloc[i]
    # zoom on the region around the optimum (the far end of the front is not interesting)
    lim = min(2.5 * best_total,
              100 * max(front['rel_unc_stat'].max(), front['rel_unc_syst'].max()) * 1.1)
    phi = np.linspace(0, np.pi / 2, 100)
    for k, r in enumerate(best_total * np.array([1.0, 1.1, 1.25, 1.5])):
        ax.plot(r * np.cos(phi), r * np.sin(phi), color='#5f5e5a', ls='--', lw=0.8, alpha=0.6,
                label=r'constant $\sigma_S/S$' if k == 0 else None)
        ax.annotate(f'{r:.1f}%', (r * np.cos(0.8), r * np.sin(0.8)), fontsize=7, color='#5f5e5a')
    ax.plot(100 * front['rel_unc_stat'].iloc[i], 100 * front['rel_unc_syst'].iloc[i], '*',
            color='#eb6834', ms=22, mec='white', label=f'minimum total on front ({best_total:.2f}%)',
            zorder=3)                              # behind the marks, larger: both stay visible
    if marks:
        if data is None or not control_regions:
            raise ValueError('marks need data and control_regions to be evaluated.')
        for (label, cut), (color, marker) in zip(marks.items(), _CUT_STYLES[2:] + _CUT_STYLES[:2]):
            r = evaluate_cut(data, cut, control_regions=control_regions, **eval_kw)
            x, yv = 100 * r['rel_unc_stat'], 100 * np.hypot(r['rel_unc_norm'], r['rel_unc_shape'])
            if not (np.isfinite(x) and np.isfinite(yv)):
                ax.plot([], [], marker, color=color, ms=9, label=f'{label} (no bin passes; not shown)')
                continue
            tag = '' if max(x, yv) <= lim else ' (outside the plot)'
            ax.plot(x, yv, marker, color=color, ms=9, mec='white', mew=1.5,
                    label=f'{label}: {np.hypot(x, yv):.2f}%{tag}', zorder=6, clip_on=True)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(r'statistical part of $\sigma_S/S$ [%]')
    ax.set_ylabel(r'systematic part (control regions) of $\sigma_S/S$ [%]')
    ax.grid(alpha=0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    # the points and the front sit at the upper right (both parts are well above 0),
    # so the lower left corner is empty
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


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
    crs = eval_kw.get('control_regions')
    rng = np.random.default_rng(seed)
    sels = {name: select(data, name, theta, half, bcs_order, bcs_base) for name in data.channels}
    zs = np.empty(n_boot)
    for k in range(n_boot):
        pw = {name: rng.poisson(1.0, size=len(sels[name])) for name in data.channels}
        if crs:
            # signal-region MC resampled; control-region fractions kept at their values
            ru = relative_uncertainty(data, theta, crs, half=half, bcs_order=bcs_order,
                                      bcs_base=bcs_base, eps_bkg=eps_bkg,
                                      min_bin_count=min_bin_count, count_on=count_on,
                                      estimator=estimator, weights=pw)
            zs[k] = 1.0 / ru['rel_unc'] if ru['rel_unc'] > 0 else 0.0
            continue
        z2 = 0.0
        for name, ch in data.channels.items():
            z2 += _channel_result(ch, sels[name], eps_bkg, min_bin_count, count_on,
                                  estimator, weights=pw[name], scale=_half_factor(half))[0]
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
                      estimator='unbiased', control_regions=None):
    """One row per (cut, dataset, BCS order).

    control_regions: as in evaluate_cut; adds the sigma_S/S breakdown and the
    per-region numbers (rel_unc*, cr_*) to every row.

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
                                 count_on=count_on, estimator=estimator,
                                 control_regions=control_regions, **kw)
                eff = efficiencies(data, theta, **kw)
                rows.append(dict(cut_name=cname, dataset=dlabel, bcs_order=order,
                                 cut=theta_to_cut(theta),
                                 **{k: r[k] for k in r if k.startswith('Z')},
                                 n_bins=r['n_bins'], excess_frac=r['excess_frac'],
                                 S=r['S'], B=r['B'],
                                 **{k: r[k] for k in r if k.startswith(('rel_unc', 'cr_'))},
                                 **eff))
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


def _category_yields(ch, idx, region, scale=1.0):
    """{category: weighted yield}, {category: sum of w^2}, {category: raw MC count}."""
    keep = _region_mask(ch, idx, region)
    cat, w = ch.cat[idx][keep], ch.w[idx][keep] * scale
    nc = len(ch.cat_names)
    y = np.bincount(cat, weights=w, minlength=nc)
    y2 = np.bincount(cat, weights=w ** 2, minlength=nc)
    n = np.bincount(cat, minlength=nc)
    return (dict(zip(ch.cat_names, y)), dict(zip(ch.cat_names, y2)),
            dict(zip(ch.cat_names, n.astype(float))))


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
    group, n_before, n_after (yields at the luminosity of the input, i.e. data
    luminosity with lumi_scale; for half 'A' / 'B' scaled by HALF_LUMI_FACTOR
    to the full data luminosity), n_mc_before (raw MC events), eff, eff_err
    (binomial, sqrt(eff(1-eff)/N_eff), N_eff = (sum w)^2 / sum w^2 = the raw MC
    count for equal weights).
    Two summary rows per channel, 'FOM signal (total)' and 'FOM background
    (total)', sum the categories entering s and b.
    """
    if region not in EFF_REGIONS:
        raise ValueError(f'region must be one of {EFF_REGIONS}.')
    rows = []
    for cname, cut in cuts.items():
        theta = as_theta(cut)
        tot_b, tot_b2, tot_bn, tot_a = {}, {}, {}, {}
        per_channel = {}
        for name, ch in data.channels.items():
            b, b2, bn = _category_yields(ch, _kept_candidates(data, name, OPEN_CUT, half, level,
                                                              bcs_order, bcs_base), region,
                                         _half_factor(half))
            a, _, _ = _category_yields(ch, _kept_candidates(data, name, theta, half, level,
                                                            bcs_order, bcs_base), region,
                                       _half_factor(half))
            per_channel[name] = (b, b2, bn, a)
            for k in b:
                tot_b[k] = tot_b.get(k, 0.0) + b[k]
                tot_b2[k] = tot_b2.get(k, 0.0) + b2[k]
                tot_bn[k] = tot_bn.get(k, 0.0) + bn[k]
                tot_a[k] = tot_a.get(k, 0.0) + a.get(k, 0.0)
        if len(data.channels) > 1:
            per_channel['e+mu' if set(data.channels) == {'e', 'mu'} else 'all channels'] = \
                (tot_b, tot_b2, tot_bn, tot_a)
        for chan, (b, b2, bn, a) in per_channel.items():
            summary = {'FOM signal (total)': FOM_SIGNAL, 'FOM background (total)': FOM_BACKGROUND}
            groups = [(c, [c]) for c in b] + list(summary.items())
            for cat, members in groups:
                nb = sum(b.get(c, 0.0) for c in members)
                nb2 = sum(b2.get(c, 0.0) for c in members)
                nmc = sum(bn.get(c, 0.0) for c in members)
                na = sum(a.get(c, 0.0) for c in members)
                eff = na / nb if nb > 0 else np.nan
                # binomial error with the effective number of MC events (sum w)^2 / sum w^2:
                # the MC statistics, not the luminosity-scaled yield, set the precision
                n_eff = nb ** 2 / nb2 if nb2 > 0 else 0.0
                err = np.sqrt(eff * (1 - eff) / n_eff) if n_eff > 0 and 0 <= eff <= 1 else np.nan
                group = 'summary' if cat in summary else _category_group(cat)
                rows.append(dict(cut_name=cname, cut=theta_to_cut(theta), channel=chan,
                                 category=cat, group=group, n_before=nb, n_after=na,
                                 n_mc_before=nmc, eff=eff, eff_err=err))
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
    entries 'eff +- err' (in % by default), plus the yield before the cut (at
    the input's luminosity) and the number of raw MC events behind it."""
    t = table[table['channel'] == channel]
    scale = 100.0 if percent else 1.0
    fmt = lambda r: (f"{scale * r['eff']:.{decimals}f} ± {scale * r['eff_err']:.{decimals}f}"
                     if np.isfinite(r['eff_err']) else f"{scale * r['eff']:.{decimals}f}")
    t = t.assign(value=t.apply(fmt, axis=1))
    cut_order = list(dict.fromkeys(t['cut_name']))
    cat_order = list(dict.fromkeys(t['category']))
    wide = t.pivot(index='category', columns='cut_name', values='value')
    wide = wide.reindex(index=cat_order, columns=cut_order)
    first = t.drop_duplicates('category').set_index('category')
    n0 = first['n_before'].reindex(cat_order)
    group = first['group'].reindex(cat_order)
    wide.insert(0, 'MC events before', first['n_mc_before'].reindex(cat_order).round().astype('Int64'))
    wide.insert(0, 'N before (data lumi)', n0.round().astype('Int64'))
    wide.insert(0, 'group', group)
    wide.columns.name = f'efficiency [{"%" if percent else "fraction"}], {channel}'
    return wide


# categorical slots 1-4 of the dataviz reference palette (light), each with its
# own marker so identity never relies on color alone
_CUT_STYLES = [('#2a78d6', 'o'), ('#eb6834', 's'), ('#1baf7a', 'D'), ('#eda100', '^')]


def plot_efficiencies(table, channel='e+mu', log=True, save=None):
    """Dot plot of the per-category efficiencies: one row per category, one
    marker per cut (up to four cuts), binomial error bars, log x by default
    because efficiencies span orders of magnitude. On a log axis a zero
    efficiency (no MC event passes) is drawn as a hollow marker at the left
    edge, labeled '0'."""
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


############################## ROC ############################################

def _area(y, x):
    """Trapezoidal integral of y over x (np.trapz/np.trapezoid differ by numpy version)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def roc_curve(data, score='sig_prob', background=None, half='all', level='candidate',
              region='sr', n_points=200, bcs_order='cut_then_bcs', bcs_base=None):
    """ROC of one BDT output: signal (FOM_SIGNAL) vs `background` efficiency
    as the threshold on `score` is scanned, the other outputs left open.

    background : list of categories (default FOM_BACKGROUND, combined).
    level      : 'candidate' - every candidate, no BCS (the BDT's own ROC; exact,
                               all thresholds);
                 'event'     - after BCS, evaluated at n_points thresholds.
    Efficiencies are relative to no BDT cut, counted in `region` ('sr', 'fit',
    'all'), so the working points from roc_working_points sit on the same axes.
    Returns (DataFrame threshold / eff_sig / eff_bkg, AUC). AUC is the area
    under eff_sig vs eff_bkg (1 = perfect, 0.5 = random), for level='candidate'
    only. At event level it is None: BCS can move an event into another
    category when the cut removes its best candidate, so event-level
    efficiencies need not be monotonic or <= 1, and an area is not meaningful.
    """
    background = list(background or FOM_BACKGROUND)
    if CUT_DIRECTIONS[score] != '>':
        raise ValueError('roc_curve scans a "score > t" output, e.g. sig_prob.')
    if level == 'candidate':
        vals, is_sig, is_bkg, w = [], [], [], []
        for ch in data.channels.values():
            idx = np.arange(len(ch))
            h = _half_mask(ch, half)
            if h is not None:
                idx = idx[h]
            idx = idx[_region_mask(ch, idx, region)]
            names = np.array(ch.cat_names, dtype=object)[ch.cat[idx]]
            vals.append(ch.probs[score][idx])
            is_sig.append(np.isin(names, FOM_SIGNAL))
            is_bkg.append(np.isin(names, background))
            w.append(ch.w[idx])
        vals, is_sig, is_bkg, w = map(np.concatenate, (vals, is_sig, is_bkg, w))
        order = np.argsort(-vals, kind='stable')
        vals, ws, wb = vals[order], (w * is_sig)[order], (w * is_bkg)[order]
        cs, cb = np.cumsum(ws), np.cumsum(wb)
        last = np.r_[vals[1:] != vals[:-1], True]           # one point per distinct value
        eff_s = np.r_[0.0, cs[last] / cs[-1]]
        eff_b = np.r_[0.0, cb[last] / cb[-1]]
        thr = np.r_[1.0, vals[last]]
        auc = _area(eff_s, eff_b)
        if len(thr) > n_points:                              # thin out for plotting
            keep = np.unique(np.r_[0, np.linspace(0, len(thr) - 1, n_points).astype(int),
                                   len(thr) - 1])
            thr, eff_s, eff_b = thr[keep], eff_s[keep], eff_b[keep]
        return pd.DataFrame(dict(threshold=thr, eff_sig=eff_s, eff_bkg=eff_b)), auc
    if level != 'event':
        raise ValueError("level must be 'candidate' or 'event'.")
    vals = np.concatenate([ch.probs[score][(ch.cls == 1) & ch.in_sr] for ch in data.channels.values()])
    thresholds = np.unique(np.r_[0.0, np.quantile(vals, np.linspace(0, 0.999, n_points)), 1.0])
    rows = []
    for t in thresholds:
        tab = efficiency_table(data, {'t': {score: float(t)}}, half=half, level='event',
                               region=region, bcs_order=bcs_order, bcs_base=bcs_base)
        rows.append(_roc_point(tab, background) | dict(threshold=float(t)))
    roc = pd.DataFrame(rows).sort_values('eff_bkg', ignore_index=True)[['threshold', 'eff_sig', 'eff_bkg']]
    return roc, None


def _roc_point(table, background):
    """(eff_sig, eff_bkg) from an efficiency_table of a single cut, all channels summed."""
    chan = table['channel'].iloc[-1] if 'e+mu' not in set(table['channel']) else 'e+mu'
    t = table[(table['channel'] == chan) & (table['group'] != 'summary')]
    n0 = lambda cats: t.loc[t['category'].isin(cats), 'n_before'].sum()
    n1 = lambda cats: t.loc[t['category'].isin(cats), 'n_after'].sum()
    return dict(eff_sig=n1(FOM_SIGNAL) / n0(FOM_SIGNAL),
                eff_bkg=n1(background) / n0(background) if n0(background) else np.nan)


def roc_working_points(data, cuts, background=None, half='all', level='candidate', region='sr',
                       bcs_order='cut_then_bcs', bcs_base=None):
    """(eff_sig, eff_bkg) of each 4-output cut, on the same axes as roc_curve."""
    background = list(background or FOM_BACKGROUND)
    rows = []
    for name, cut in cuts.items():
        tab = efficiency_table(data, {name: cut}, half=half, level=level, region=region,
                               bcs_order=bcs_order, bcs_base=bcs_base)
        rows.append(dict(cut_name=name, cut=theta_to_cut(as_theta(cut)), **_roc_point(tab, background)))
    return pd.DataFrame(rows)


def plot_roc(data, cuts, by_class=False, half='all', level='candidate', region='sr',
             log=True, save=None, **kw):
    """ROC of sig_prob with the working points of `cuts` (up to four) marked.

    by_class=False: one panel, signal vs all FOM backgrounds combined.
    by_class=True : one panel per FOM background category (2 x 3).
    A working point below the curve rejects more background at the same signal
    efficiency than a cut on sig_prob alone, i.e. the cuts on the background
    outputs add information. Background efficiency on a log axis by default.
    """
    if len(cuts) > len(_CUT_STYLES):
        raise ValueError(f'At most {len(_CUT_STYLES)} working points; split them.')
    panels = [list(FOM_BACKGROUND)] if not by_class else [[c] for c in FOM_BACKGROUND]
    ncol = 1 if not by_class else 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 if ncol == 1 else 13, 5 if nrow == 1 else 7.5),
                             squeeze=False)
    common = dict(half=half, level=level, region=region)
    # score / n_points only shape the curve; the BCS options apply to both
    curve_kw = dict(common, **kw)
    wp_kw = dict(common, **{k: v for k, v in kw.items() if k in ('bcs_order', 'bcs_base')})
    for ax, bkg in zip(axes.flat, panels):
        roc, auc = roc_curve(data, background=bkg, **curve_kw)
        ax.plot(roc['eff_sig'], roc['eff_bkg'], color='#5f5e5a', lw=2,
                label='sig_prob scan' + (f' (AUC {auc:.3f})' if auc is not None else ''))
        wp = roc_working_points(data, cuts, background=bkg, **wp_kw)
        positive = np.r_[roc['eff_bkg'][roc['eff_bkg'] > 0], wp['eff_bkg'][wp['eff_bkg'] > 0]]
        y_floor = positive.min() / 3 if log and len(positive) else 0.0
        for (_, r), (color, marker) in zip(wp.iterrows(), _CUT_STYLES):
            if log and not r['eff_bkg'] > 0:      # no background passes: hollow marker at the floor
                ax.plot(r['eff_sig'], y_floor, marker, ms=8, mfc='none', color=color,
                        clip_on=False, label=f"{r['cut_name']} (0 bkg passes)")
                continue
            ax.plot(r['eff_sig'], r['eff_bkg'], marker, ms=8, color=color, mec='white', mew=1.5,
                    label=r['cut_name'])
        if log:
            ax.set_yscale('log')
            ax.set_ylim(bottom=y_floor / 1.5, top=1.5)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(f'signal efficiency ({FOM_SIGNAL[0]})')
        ax.set_ylabel('background efficiency')
        ax.set_title(bkg[0] if len(bkg) == 1 else 'all FOM backgrounds', fontsize=10)
        ax.grid(alpha=0.3, which='both')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, loc='upper left', frameon=False)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f'BDT ROC ({level} level, {region.upper()}, half={half})', fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


############################## plots ##########################################

def n_minus_1_scans(data, theta, grids, **eval_kw):
    """N-1 scans around theta: {output: scan DataFrame}."""
    return {v: scan_1d(data, v, theta, np.unique(np.append(grids[v], as_theta(theta)[v])), **eval_kw)
            for v in MVA_OUTPUTS}


def plot_n_minus_1(scans, marks=None, sigma_z=None, z_ref=None, y='Z', save=None):
    """Target vs threshold for each output (2x2), with the signal and background
    yields relative to the open point on a twin axis.

    y      : 'Z' (default, higher is better) or 'rel_unc' (sigma_S/S in %, lower
             is better; = 1/Z, with the control-region terms if the scans were
             made with control_regions).
    marks  : {label: theta or cut string} drawn as vertical lines.
    sigma_z, z_ref: draw the plateau band, Z* - sigma_Z .. Z* (for y='rel_unc'
             the same band converted: 1/Z* .. 1/(Z* - sigma_Z)).
    """
    if y not in ('Z', 'rel_unc'):
        raise ValueError("y must be 'Z' or 'rel_unc'.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    styles = ['-', '--', ':', '-.']
    for ax, var in zip(axes.flat, MVA_OUTPUTS):
        sc = scans[var].sort_values('threshold')
        if y == 'Z':
            ax.plot(sc['threshold'], sc['Z'], 'k.-', label='Z')
        else:
            ru = 100 * sc['rel_unc'].replace(np.inf, np.nan)   # no bin passes -> not drawn
            ax.plot(sc['threshold'], ru, 'k.-', label=r'$\sigma_S/S$')
        if sigma_z is not None and z_ref is not None:
            if y == 'Z':
                ax.axhspan(z_ref - sigma_z, z_ref, color='gray', alpha=0.25,
                           label=r'$Z^\ast - 1\sigma$')
            elif z_ref - sigma_z > 0:
                ax.axhspan(100 / z_ref, 100 / (z_ref - sigma_z), color='gray', alpha=0.25,
                           label=r'best $+ 1\sigma$')
        ax2 = ax.twinx()
        open_row = sc.loc[sc['threshold'].map(lambda t: _looseness(var, t)).idxmax()]
        ax2.plot(sc['threshold'], sc['S'] / open_row['S'], color='tab:blue', lw=1, label='S / S(open)')
        ax2.plot(sc['threshold'], sc['B'] / open_row['B'], color='tab:red', lw=1, label='B / B(open)')
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel('relative yield')
        for (label, th), ls in zip((marks or {}).items(), styles):
            ax.axvline(as_theta(th)[var], color='tab:green', ls=ls, label=label)
        ax.set_xlabel(f'{var} threshold ({var}{CUT_DIRECTIONS[var]}t)')
        ax.set_ylabel('binned FOM Z' if y == 'Z' else r'$\sigma_S/S$ [%] (lower is better)')
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower left' if y == 'Z' else 'upper left')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# per-region lines: categorical slots 4-8 of the reference palette, one dash pattern each
_CR_STYLES = [('#eda100', (0, (6, 2))), ('#e87ba4', (0, (2, 2))), ('#6250d6', (0, (6, 2, 2, 2))),
              ('#e34948', (0, (1, 1))), ('#008300', (0, (8, 3)))]


def _cr_names(frame):
    """Control-region names present in an evaluate_cut / scan result."""
    return [c[3:-len('_impact')] for c in frame.columns
            if c.startswith('cr_') and c.endswith('_impact')]


def plot_uncertainty_breakdown(scans, marks=None, per_cr=False, save=None):
    """sigma_S/S and its parts along the N-1 scans of
    n_minus_1_scans(..., control_regions=crs). Lower is better.

    Default lines: total, statistical, normalization (all control regions),
    shape (all control regions); the parts add in quadrature to the total.
    per_cr=True adds one line per control region: its impact,
    sqrt(total^2 - total_without_that_region^2), i.e. how much sigma_S/S
    drops if that region's normalization and shape terms are removed. Impacts
    of different regions need not add up exactly in quadrature, because the
    regions can describe the same signal-region events (e.g. qq fake D).
    The active control regions are listed in the figure title.
    """
    parts = [('rel_unc', 'total', '#2b2a27', '-'), ('rel_unc_stat', 'statistical', '#2a78d6', '-'),
             ('rel_unc_norm', 'normalization (all CRs)', '#eb6834', '--'),
             ('rel_unc_shape', 'shape (all CRs)', '#1baf7a', ':')]
    first = scans[MVA_OUTPUTS[0]]
    if 'rel_unc_norm' not in first:
        raise ValueError('The scans have no control-region terms: pass control_regions to n_minus_1_scans.')
    names = _cr_names(first)
    if per_cr and len(names) > len(_CR_STYLES):
        raise ValueError(f'At most {len(_CR_STYLES)} control regions per plot with per_cr=True.')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, var in zip(axes.flat, MVA_OUTPUTS):
        sc = scans[var].sort_values('threshold')
        for col, label, color, ls in parts:
            ax.plot(sc['threshold'], 100 * sc[col], color=color, ls=ls, lw=2, label=label)
        if per_cr:
            for name, (color, dash) in zip(names, _CR_STYLES):
                ax.plot(sc['threshold'], 100 * sc[f'cr_{name}_impact'], color=color, ls=dash,
                        lw=1.5, label=f'{name} (impact)')
        for (label, th), mls in zip((marks or {}).items(), ['-', '--', ':', '-.']):
            ax.axvline(as_theta(th)[var], color='#c3c2b7', ls=mls, lw=1, label=label)
        ax.set_xlabel(f'{var} threshold ({var}{CUT_DIRECTIONS[var]}t)')
        ax.set_ylabel(r'$\sigma_S/S$ [%]')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    # one shared legend below the panels (the entries are the same in every panel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle('Uncertainty breakdown along N-1 scans; control regions: '
                 + (', '.join(names) if names else 'none'), fontsize=11)
    fig.tight_layout(rect=(0, 0.07 if len(labels) <= 8 else 0.1, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, frameon=False)
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