#!/usr/bin/env python3
"""Check bdt_cut_optimization against a direct pandas implementation.

The module replaces "query(cut) -> groupby().idxmin() -> D_M window ->
histogram2d" by pre-sorted numpy arrays and bincount, for speed. These tests
build synthetic multi-candidate events (no ntuples needed) and require that
both routes give the same selected candidates and the same Z, for both BCS
orders, both halves and with NaN BCS ranks present. They then run the full
optimization on a toy where the right answer is known in direction.

    python3 scripts/tests/test_bdt_cut_optimization.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent / 'tools'
sys.path.insert(0, str(TOOLS_DIR))

import utilities as util            # noqa: E402
import bdt_cut_optimization as bco  # noqa: E402

CATS = bco.FOM_SIGNAL + bco.FOM_BACKGROUND + [r'$D\ell\nu$', r'$D^\ast\tau\nu$']
# which output each category tends to score high in
PREFERRED = {r'$D\tau\nu$': 0, 'bkg_fakeD': 1, 'bkg_fakeL': 1, 'bkg_fakeTracks': 1,
             'bkg_continuum': 2, 'bkg_combinatorial': 3, 'bkg_hadronicB_secondaryL': 3,
             r'$D\ell\nu$': 0, r'$D^\ast\tau\nu$': 0}


def toy_channel(n_events, seed):
    rng = np.random.default_rng(seed)
    n_cand = rng.integers(1, 5, size=n_events)
    ev = np.repeat(np.arange(n_events), n_cand)
    n = len(ev)
    cat = rng.choice(CATS, size=n, p=[0.08, 0.35, 0.12, 0.12, 0.08, 0.05, 0.03, 0.12, 0.05])
    alpha = np.ones((n, 4))
    alpha[np.arange(n), [PREFERRED[c] for c in cat]] = 4.0
    probs = np.array([rng.dirichlet(a) for a in alpha])
    sig = cat == r'$D\tau\nu$'
    df = pd.DataFrame({
        'sig_prob': probs[:, 0], 'fakeD_prob': probs[:, 1],
        'continuum_prob': probs[:, 2], 'combinatorial_prob': probs[:, 3],
        'B0_recMissM2': np.where(sig, rng.normal(3, 2, n), rng.normal(1, 2.5, n)),
        'p_D_l': rng.uniform(0.3, 5.0, n),
        'D_M': rng.uniform(1.83, 1.91, n),
        'B_D_ReChi2': rng.exponential(2, n),
        '__experiment__': 26, '__run__': ev // 1000, '__event__': ev, '__production__': 1,
        'category': cat,
        # qq origin: all continuum, 30 % of the fake D
        'B0_isContinuumEvent': ((cat == 'bkg_continuum')
                                | ((cat == 'bkg_fakeD') & (rng.random(n) < 0.3))).astype(int),
    })
    df.loc[rng.random(n) < 0.02, 'B_D_ReChi2'] = np.nan
    # shuffle so the module's sorting is actually exercised
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def reference(df, theta, half, bcs_order, bcs_base=None, min_count=10, count_on='fom',
              bins=bco.FOM_BINS, estimator='plain'):
    """The slow, obvious implementation."""
    def apply(d, th):
        cut = bco.theta_to_cut(th)
        return d.query(cut) if cut else d

    def bcs(d):
        d = d[d['B_D_ReChi2'].notna()]
        return d.loc[d.groupby(util.EVENT_ID_COLUMNS)['B_D_ReChi2'].idxmin()]

    d = df if half == 'all' else df[df['__event__'] % 2 == {'A': 0, 'B': 1}[half]]
    if bcs_order == 'cut_then_bcs':
        best = bcs(apply(d, theta))
    else:
        best = apply(bcs(apply(d, bco.as_theta(bcs_base))), theta)
    sr = best.query(util.D_MASS_SR)
    def hist(d):
        return np.histogram2d(*(d[v] for v in bco.FIT_VARIABLES), bins=bins)[0]
    s = hist(sr[sr.category.isin(bco.FOM_SIGNAL)])
    b = hist(sr[sr.category.isin(bco.FOM_BACKGROUND)])
    n = s + b if count_on == 'fom' else hist(sr)
    ok = (n >= min_count) & (s + b > 0)
    if estimator == 'plain':
        z2 = np.sum(s[ok] ** 2 / (s[ok] + b[ok]))
    else:
        ok &= s + b >= 2
        z2 = np.sum(s[ok] * (s[ok] - 1) / (s[ok] + b[ok] - 1))
    return np.sqrt(z2), best


def check(name, cond):
    print(f'{"PASS" if cond else "FAIL"}  {name}')
    return cond


def test_beam_energy_correction(tmpdir):
    """Off-resonance loading: Mbc / DeltaE recomputed at the on-resonance energy
    before the offline cut and the BDT (apply_mva_bcs stubbed: no model file)."""
    import uproot
    ok = True
    rng = np.random.default_rng(5)
    n = 4000
    ecms = rng.normal(10.52, 0.002, n)
    p_roe = rng.uniform(0.0, 1.0, n)
    mbc_on = rng.uniform(4.9, 5.29, n)                       # target on-resonance Mbc
    # build the off-resonance kinematics: E* and p* such that, scaled to 10.58, they give mbc_on
    k = bco.ONRES_ECMS / ecms
    p_star = np.sqrt(np.clip((bco.ONRES_ECMS / 2) ** 2 - mbc_on ** 2, 0, None)) / k
    e_star = ecms / 2 + rng.normal(0, 0.3, n)
    tree = {'__experiment__': np.full(n, 26, np.int32), '__run__': np.ones(n, np.int32),
            '__event__': np.arange(n, dtype=np.int64), '__production__': np.ones(n, np.int32),
            'Ecms': ecms, 'B0_CMS_roeP_my_mask': p_star, 'B0_CMS_roeE_my_mask': e_star,
            'B0_roeMbc_my_mask': np.sqrt((ecms / 2) ** 2 - p_star ** 2),
            'B0_roeDeltae_my_mask': e_star - ecms / 2,
            'B0_dr': np.full(n, 0.01), 'ell_p': np.full(n, 1.5), 'B0_vtxReChi2': rng.uniform(0, 5, n),
            'D_vtxReChi2': rng.uniform(0, 5, n), 'D_CMS_p': rng.uniform(0.5, 2.5, n),
            'ell_CMS_p': rng.uniform(0.5, 2.5, n), 'D_M': np.full(n, 1.87),
            'B0_recMissM2': rng.normal(0, 2, n)}
    path = f'{tmpdir}/offres_toy.root'
    with uproot.recreate(path) as f:              # a TTree (as the ntuples), not an RNTuple
        f.mktree('B0_e', {b: np.asarray(v).dtype for b, v in tree.items()})
        f['B0_e'].extend(tree)
    seen = {}

    def stub(df, features, cut, **kw):
        seen['mbc'] = df['B0_roeMbc_my_mask'].to_numpy().copy()
        seen['de'] = df['B0_roeDeltae_my_mask'].to_numpy().copy()
        out = df.copy()                 # util.apply_mva_bcs keeps the basf2 event-ID names
        for v in bco.MVA_OUTPUTS:
            out[v] = 0.25
        return out

    orig = util.apply_mva_bcs
    util.apply_mva_bcs = stub
    try:
        raw = bco.load_candidates([path], 'e', is_data=True, verbose=False)
        mbc_raw = seen['mbc']
        cor = bco.load_candidates([path], 'e', is_data=True, ecms_target=bco.ONRES_ECMS, verbose=False)
        mbc_cor, de_cor = seen['mbc'], seen['de']
    finally:
        util.apply_mva_bcs = orig
    de_on = e_star * k - bco.ONRES_ECMS / 2                   # target on-resonance DeltaE
    pass_on = (mbc_on > 5) & (de_on > -4) & (de_on < 1)       # the offline cut at 10.58 GeV
    ok &= check(f'beam energy: the BDT sees Mbc at 10.58 GeV ({len(mbc_cor)} candidates)',
                len(mbc_cor) == pass_on.sum()
                and np.allclose(np.sort(mbc_cor), np.sort(mbc_on[pass_on]), atol=1e-9))
    ok &= check('beam energy: DeltaE at 10.58 GeV, offline cut applied after the correction',
                len(cor) == pass_on.sum() and np.allclose(np.sort(de_cor), np.sort(de_on[pass_on])))
    ok &= check(f'beam energy: without the correction the cut and the BDT see the off-res Mbc '
                f'({len(raw)} vs {len(cor)} candidates pass)',
                len(raw) < len(cor) and np.all(mbc_raw > 5)
                and raw.attrs['ecms_target'] is None and cor.attrs['ecms_target'] == bco.ONRES_ECMS)
    sm = cor.attrs['beam_energy_summary']
    ok &= check(f"beam energy: summary counts (read {sm['n_read']}, pass {sm['n_pass']}, "
                f"uncorrected {sm['n_pass_ori']}, mean dMbc {1e3 * sm['sum_dMbc'] / sm['n_shift']:+.1f} MeV)",
                sm['n_pass'] == len(cor) and sm['n_pass_ori'] == len(raw) and sm['n_off'] == 0
                and sm['sum_dMbc'] > 0)
    df_t = pd.DataFrame(tree)
    chk = bco.correct_beam_energy(df_t)
    ok &= check('beam energy: formula reproduces the ntuple Mbc / DeltaE at the event Ecms',
                chk['n_off'] == 0 and chk['n'] > 0
                and np.allclose(df_t['B0_roeMbc_ori'], tree['B0_roeMbc_my_mask']))
    return ok


def main():
    dfs = {'e': toy_channel(6000, 1), 'mu': toy_channel(5000, 2)}
    data = bco.FOMInput(dfs)
    # coarse binning so that the 10-event threshold removes some bins, not all
    coarse = [np.linspace(-2.5, 10, 11), np.linspace(0.4, 4.8, 9)]
    data_coarse = bco.FOMInput(dfs, bins=coarse)
    ok = True

    ok &= check('cut string round trip (lgb_tight)',
                bco.cut_to_theta(bco.theta_to_cut(bco.cut_to_theta(util.lgb_tight)))
                == bco.cut_to_theta(util.lgb_tight))

    thetas = {'open': bco.OPEN_CUT, 'lgb_tight': util.lgb_tight, 'lgb_loose': util.lgb_loose,
              'sig only': 'sig_prob>0.35'}
    for label, th in thetas.items():
        for half in ('A', 'B', 'all'):
            for order, base in (('cut_then_bcs', None), ('bcs_then_cut', None),
                                ('bcs_then_cut', util.lgb_loose)):
                for d, bins, tag in ((data, bco.FOM_BINS, 'fit'), (data_coarse, coarse, 'coarse')):
                    for min_count, count_on, est in ((1, 'fom', 'plain'), (10, 'fom', 'plain'),
                                                     (10, 'all', 'plain'), (10, 'fom', 'unbiased'),
                                                     (10, 'all', 'unbiased')):
                        z_fast = bco.evaluate_cut(d, th, half=half, bcs_order=order, bcs_base=base,
                                                  min_bin_count=min_count, count_on=count_on,
                                                  estimator=est)['Z']
                        # one half is scaled to the full luminosity: Z^2 x HALF_LUMI_FACTOR
                        z_ref = np.sqrt(sum(reference(dfs[c], bco.as_theta(th), half, order, base,
                                                      min_count, count_on, bins, est)[0] ** 2
                                            for c in dfs) * bco._half_factor(half))
                        ok &= check(f'Z {label:9s} half={half:3s} {order:12s} base={bool(base)!s:5s} '
                                    f'{tag:6s} n>={min_count:<2d} on={count_on:3s} {est:8s}: '
                                    f'{z_fast:.6f} vs {z_ref:.6f}',
                                    np.isclose(z_fast, z_ref, rtol=1e-9, atol=1e-9))

    # same selected candidates, not only the same Z
    th = bco.as_theta(util.lgb_loose)
    ch = data.channels['e']
    sel = bco.select(data, 'e', th, half='all')
    _, best = reference(dfs['e'], th, 'all', 'cut_then_bcs')
    fast_keys = set(zip(ch.group[sel], ch.probs['sig_prob'][sel]))
    ok &= check('same number of BCS candidates', len(sel) == len(best))
    ok &= check('same BCS candidates (sig_prob per event)',
                sorted(np.float32(best.sort_values('__event__')['sig_prob']))
                == sorted(p for _, p in fast_keys))

    # the full optimization runs and does not end below its starting points
    data = data_coarse
    res = bco.run_optimization(data, grids=bco.make_grids(data, n=12), verbose=False)
    z_best = res['summary']['Z'].iloc[0]
    for label, th in {'open': bco.OPEN_CUT, 'lgb_tight': util.lgb_tight,
                      'lgb_loose': util.lgb_loose}.items():
        ok &= check(f'optimum ({z_best:.3f}) >= start {label}',
                    z_best >= bco.evaluate_cut(data, th)['Z'] - 1e-12)
    ok &= check('optimum cuts on sig_prob', res['best_theta']['sig_prob'] > 0)

    wps = bco.choose_working_points(data, res['best_theta'], grids=res['grids'], n_boot=30)
    z_t = bco.evaluate_cut(data, wps['tight'])['Z']
    z_l = bco.evaluate_cut(data, wps['loose'])['Z']
    ok &= check('tight within 1 sigma of optimum', z_t >= wps['z_star'] - wps['sigma_z'] - 1e-12)
    ok &= check('loose within 5% of optimum, or equal to tight',
                z_l >= min(0.95 * wps['z_star'], z_t) - 1e-12)
    ok &= check('loose keeps at least as much signal as tight',
                bco.evaluate_cut(data, wps['loose'])['S'] >= bco.evaluate_cut(data, wps['tight'])['S'])

    rep = bco.validation_report({'tight': wps['tight'], 'old': util.lgb_tight},
                                {'A': (data, 'A'), 'B': (data, 'B')},
                                bcs_orders=('cut_then_bcs', 'bcs_then_cut'))
    ok &= check('validation report has 8 rows and an efficiency ratio',
                len(rep) == 8 and 'eff_ratio_sig_norm' in rep)

    # the MC-statistics excess quoted in the module docstring, by exact summation
    from scipy.stats import poisson
    for s, b in [(0.5, 0.5), (1, 9), (5, 1), (10, 10), (2, 8)]:
        k = np.arange(0, 80)
        sh, bh = np.meshgrid(k, k, indexing='ij')
        n = sh + bh
        f = np.divide(sh ** 2, n, out=np.zeros(n.shape), where=n > 0)
        exact = (np.outer(poisson.pmf(k, s), poisson.pmf(k, b)) * f).sum() - s * s / (s + b)
        p = s / (s + b)
        ok &= check(f'excess s={s:<3g} b={b:<3g}: exact {exact:.6f} vs p(1-p)(1-e^-mu) '
                    f'{p * (1 - p) * (1 - np.exp(-(s + b))):.6f}',
                    np.isclose(exact, p * (1 - p) * (1 - np.exp(-(s + b))), atol=1e-9))
    # s(s-1)/(n-1) has no excess, also with the n >= 10 selection
    for s, b in [(2, 2), (5, 5), (10, 10), (1, 9), (2, 8)]:
        k = np.arange(0, 120)
        sh, bh = np.meshgrid(k, k, indexing='ij')
        n = sh + bh
        P = np.outer(poisson.pmf(k, s), poisson.pmf(k, b)) * (n >= 10)
        target = (P * n).sum() * (s / (s + b)) ** 2
        unb = (P * np.divide(sh * (sh - 1), n - 1, out=np.zeros(n.shape), where=n > 1)).sum()
        ok &= check(f'unbiased estimator s={s:<3g} b={b:<3g}: {unb:.8f} vs {target:.8f}',
                    np.isclose(unb, target, rtol=1e-10, atol=1e-12))

    # working points are round numbers
    for name in ('tight', 'loose'):
        vals = [wps[name][v] for v in bco.MVA_OUTPUTS]
        ok &= check(f'{name} thresholds have <= {bco.GRID_DECIMALS} decimals: {vals}',
                    all(np.isclose(v, round(v, bco.GRID_DECIMALS)) for v in vals))

    # per-category efficiencies
    full = bco.FOMInput(dfs)
    cuts = {'lgb_tight': util.lgb_tight, 'lgb_loose': util.lgb_loose}
    cand = bco.efficiency_table(full, cuts, level='candidate')
    for cname, cut in cuts.items():
        for chan, df in dfs.items():
            sr = df.query(util.D_MASS_SR)
            passed = sr.query(cut)
            for cat in (r'$D\tau\nu$', 'bkg_fakeD', r'$D\ell\nu$'):
                expect = (passed.category == cat).sum() / (sr.category == cat).sum()
                got = cand.query('cut_name == @cname and channel == @chan and category == @cat')['eff'].item()
                ok &= check(f'candidate eff {cname:9s} {chan:2s} {cat:10s}: {got:.6f} vs {expect:.6f}',
                            np.isclose(got, expect))
    ev = bco.efficiency_table(full, cuts, level='event', region='fit')
    for cname, cut in cuts.items():
        r = bco.evaluate_cut(full, cut, half='all')
        sig = ev.query('cut_name == @cname and channel == "e+mu" and category == "FOM signal (total)"')
        bkg = ev.query('cut_name == @cname and channel == "e+mu" and category == "FOM background (total)"')
        ok &= check(f'event-level totals {cname} match S, B of the FOM',
                    np.isclose(sig['n_after'].item(), r['S']) and np.isclose(bkg['n_after'].item(), r['B']))
    ok &= check('efficiency_summary keeps the cut order',
                list(bco.efficiency_summary(ev).columns[3:]) == list(cuts))

    # ROC: AUC equals the Mann-Whitney probability P(score_sig > score_bkg)
    roc, auc = bco.roc_curve(full)
    allc = pd.concat(dfs.values()).query(util.D_MASS_SR)
    xs = allc.loc[allc.category.isin(bco.FOM_SIGNAL), 'sig_prob'].to_numpy()
    xb = np.sort(allc.loc[allc.category.isin(bco.FOM_BACKGROUND), 'sig_prob'].to_numpy())
    mw = (np.searchsorted(xb, xs, side='left') + 0.5 * (np.searchsorted(xb, xs, side='right')
                                                        - np.searchsorted(xb, xs, side='left'))).mean() / len(xb)
    ok &= check(f'ROC AUC {auc:.6f} vs Mann-Whitney {mw:.6f}', np.isclose(auc, mw, atol=1e-9))
    wp = bco.roc_working_points(full, {'sig only': 'sig_prob>0.35'})
    exact, _ = bco.roc_curve(full, n_points=10 ** 9)
    on = exact[exact.threshold > 0.35].iloc[-1]
    ok &= check('a sig_prob-only working point lies on the ROC curve',
                np.isclose(wp['eff_sig'].item(), on.eff_sig) and np.isclose(wp['eff_bkg'].item(), on.eff_bkg))
    ok &= check('no AUC at event level', bco.roc_curve(full, level='event', n_points=10)[1] is None)
    import matplotlib
    matplotlib.use('Agg')
    for lvl in ('candidate', 'event'):    # the notebook calls, with n_points passed through
        try:
            bco.plot_roc(full, cuts, half='all', level=lvl, n_points=10)
            bco.plot_roc(full, cuts, half='all', level=lvl, n_points=10, by_class=True)
            ok &= check(f'plot_roc runs at {lvl} level with n_points', True)
        except Exception as err:
            ok &= check(f'plot_roc runs at {lvl} level with n_points ({err})', False)

    # ---------------- control regions and the sigma_S/S target ----------------
    # single bin: Fisher result = S / sqrt(S + B + (n B_cr)^2 + (delta B_cr)^2)
    S1, B1, Bc, nn, dd = 50.0, 200.0, 120.0, 0.07, 0.04
    one = lambda x: np.array([x])
    sig1 = bco._sigma_mu(one(S1), one(B1), [one(Bc)], [[one(Bc)]], [nn], [dd], 0.0, 'plain',
                         [True], [True])
    ok &= check(f'single-bin Fisher {sig1:.6f} vs formula '
                f'{np.sqrt(S1 + B1 + (nn * Bc) ** 2 + (dd * Bc) ** 2) / S1:.6f}',
                np.isclose(sig1, np.sqrt(S1 + B1 + (nn * Bc) ** 2 + (dd * Bc) ** 2) / S1))
    # two disjoint regions in one bin add in quadrature
    sig2 = bco._sigma_mu(one(S1), one(B1), [one(80.0), one(40.0)],
                         [[one(80.0), one(0.0)], [one(0.0), one(40.0)]], [0.05, 0.1], [0.0, 0.0],
                         0.0, 'plain', [True, True], [False, False])
    ok &= check('single-bin Fisher, two disjoint regions',
                np.isclose(sig2, np.sqrt(S1 + B1 + (0.05 * 80) ** 2 + (0.1 * 40) ** 2) / S1))

    # no control regions / zero fractions -> exactly the statistical FOM
    # (coarse binning: the toy is too small for the fit binning with n >= 10)
    tgt = bco.FOMInput(dfs, bins=coarse)
    z_stat = bco.evaluate_cut(tgt, util.lgb_loose, half='all')['Z']
    ru0 = bco.relative_uncertainty(tgt, util.lgb_loose, [], half='all')
    assert np.isfinite(z_stat) and z_stat > 0
    ok &= check(f'sigma_S/S without CRs = 1/Z_stat ({ru0["rel_unc"]:.6f} vs {1 / z_stat:.6f})',
                np.isclose(ru0['rel_unc'], 1 / z_stat))
    zero = [bco.ControlRegion.fixed('zero', 0.0, 0.0, "category == 'bkg_fakeD'")]
    ok &= check('zero-size control region changes nothing',
                np.isclose(bco.evaluate_cut(tgt, util.lgb_loose, half='all', control_regions=zero)['Z'],
                           z_stat))

    # measurement: data = MC itself -> R = 1/mc_scale, no shape mismodeling
    as_data = {c: df.assign(category='data') for c, df in dfs.items()}
    data_in = bco.FOMInput(as_data)
    cr = bco.d_mass_sideband_cr(data_in, full, mc_scale=1.25)
    m = cr.measure(util.lgb_loose)
    ok &= check(f'CR ratio with data = MC, scale 1.25: R = {m["R"]:.4f}', np.isclose(m['R'], 0.8))
    ok &= check(f'CR shape with data = MC: delta = {m["delta"]:.4f}', m['delta'] == 0.0)
    ok &= check(f"CR norm term (debiased): n = max(|R-1|, R_err) = {m['n']:.4f}",
                np.isclose(m['n'], max(0.2, m['R_err'])))
    cr_q = bco.d_mass_sideband_cr(data_in, full, mc_scale=1.25, norm_estimator='quadrature')
    ok &= check('CR norm term (quadrature): sqrt((R-1)^2 + R_err^2)',
                np.isclose(cr_q.measure(util.lgb_loose)['n'], np.sqrt(0.2 ** 2 + m['R_err'] ** 2)))
    # an offset smaller than the precision: n = R_err (not below it, not sqrt(2) R_err)
    cr_s = bco.d_mass_sideband_cr(data_in, full, mc_scale=1.0 / (1.0 + 0.3 * m['R_err'] / 0.8))
    m_s = cr_s.measure(util.lgb_loose)
    ok &= check(f"CR norm term with |R-1| < R_err: n = R_err ({abs(m_s['R'] - 1):.4f} < {m_s['R_err']:.4f})",
                abs(m_s['R'] - 1) < m_s['R_err'] and np.isclose(m_s['n'], m_s['R_err']))
    # on one half: the half's scatter is subtracted, the full-data precision added once
    m_h = cr.measure(util.lgb_loose, half='A')
    ok &= check('CR norm term on a half: n^2 = max((R-1)^2 - R_err_half^2, 0) + R_err^2',
                np.isclose(m_h['n'] ** 2, max((m_h['R'] - 1) ** 2 - m_h['R_err_half'] ** 2, 0)
                           + m_h['R_err'] ** 2))
    cr_nooff = bco.d_mass_sideband_cr(data_in, full, mc_scale=1.25, include_offset=False)
    ok &= check('include_offset=False keeps only R_err',
                np.isclose(cr_nooff.measure(util.lgb_loose)['n'], m['R_err']))
    # which signal-region events a region describes
    ch_e = full.channels['e']
    off = bco.offres_cr(data_in, full, mc_scale=1.0)
    cat_e = np.asarray(ch_e.cat_names)[ch_e.cat]
    qq_e = ch_e.truth['B0_isContinuumEvent'] == 1
    ok &= check('offres applies to all qq events (B0_isContinuumEvent == 1)',
                np.array_equal(off.sr_mask(ch_e), qq_e))
    ok &= check('D_M sidebands apply to fake D from BB events',
                np.array_equal(cr.sr_mask(ch_e), (cat_e == 'bkg_fakeD') & ~qq_e))
    ok &= check('D_M sidebands and offres apply to disjoint signal-region events, '
                'together to all fake D and all qq events',
                not np.any(off.sr_mask(ch_e) & cr.sr_mask(ch_e))
                and np.array_equal(off.sr_mask(ch_e) | cr.sr_mask(ch_e), (cat_e == 'bkg_fakeD') | qq_e)
                and np.any((cat_e == 'bkg_fakeD') & qq_e))

    # excess variance: 3 % norm + 5 % shape recovered, independent of statistics
    rng = np.random.default_rng(5)
    shape = np.exp(-np.linspace(0, 3, 40))
    for ntot in (2e4, 2e6):
        mm = ntot * shape / shape.sum()
        res = [bco.excess_variance(rng.poisson(mm * 1.03 * (1 + rng.normal(0, 0.05, 40))), mm)
               for _ in range(200)]
        dlt, rr = np.mean([r['delta'] for r in res]), np.mean([r['R'] for r in res])
        ok &= check(f'excess variance N={ntot:.0e}: delta = {dlt:.4f}, R = {rr:.4f}',
                    abs(dlt - 0.05) < 0.006 and abs(rr - 1.03) < 0.005)
    lab = bco.mismodeling_label(res[0])
    ok &= check('mismodeling label has three lines', lab.count('\n') == 2)

    # a large qq mismodeling pushes the optimum to remove more qq background
    qq = [bco.ControlRegion.fixed('qq', 0.5, 0.0, 'B0_isContinuumEvent == 1')]
    grids = bco.make_grids(tgt, n=12, half='all')
    r_stat = bco.run_optimization(tgt, grids=grids, half='all', verbose=False)
    r_qq = bco.run_optimization(tgt, grids=grids, half='all', control_regions=qq, verbose=False)
    qq_frac = lambda th: (bco.evaluate_cut(tgt, th, half='all', control_regions=qq)['cr_qq_B']
                          / bco.evaluate_cut(tgt, th, half='all')['B'])
    f_stat, f_qq = qq_frac(r_stat['best_theta']), qq_frac(r_qq['best_theta'])
    ok &= check(f'qq systematics lower the qq share of the background ({f_stat:.3f} -> {f_qq:.3f})',
                f_qq <= f_stat)
    ok &= check('sigma_S/S at the qq optimum is not above the stat optimum (with qq systematics)',
                bco.evaluate_cut(tgt, r_qq['best_theta'], half='all', control_regions=qq)['rel_unc']
                <= bco.evaluate_cut(tgt, r_stat['best_theta'], half='all', control_regions=qq)['rel_unc']
                + 1e-12)
    rq = bco.evaluate_cut(tgt, r_qq['best_theta'], half='all', control_regions=qq)
    ok &= check('breakdown adds up in quadrature',
                np.isclose(rq['rel_unc'] ** 2, rq['rel_unc_stat'] ** 2 + rq['rel_unc_norm'] ** 2
                           + rq['rel_unc_shape'] ** 2))
    import matplotlib
    matplotlib.use('Agg')
    two = qq + [bco.ControlRegion.fixed('fake_D', 0.1, 0.02, "category == 'bkg_fakeD'")]
    sc = bco.n_minus_1_scans(tgt, r_qq['best_theta'], bco.make_grids(tgt, n=5, half='all'),
                             half='all', control_regions=two)
    f1 = bco.plot_uncertainty_breakdown(sc)
    f2 = bco.plot_uncertainty_breakdown(sc, per_cr=True)
    n_lines = lambda f: len(f.axes[0].get_lines())
    ok &= check('breakdown plot: title lists the control regions, per_cr adds one line each',
                'qq, fake_D' in f1._suptitle.get_text() and n_lines(f2) == n_lines(f1) + 2)
    z0, sz, _ = bco.bootstrap_z(tgt, r_qq['best_theta'], n_boot=20, half='all', control_regions=qq)
    ok &= check(f'bootstrap with control regions runs (Z = {z0:.3f} +- {sz:.3f})', sz > 0)

    # ---------------- weighted MC (PID corrections) ----------------
    # a constant weight w scales every yield by w: Z^2 -> w Z^2 for both estimators
    tgt_w = bco.FOMInput({c: df.assign(pidw=0.5) for c, df in dfs.items()}, bins=coarse,
                         weight_col='pidw')
    for est in ('unbiased', 'plain'):
        z1 = bco.evaluate_cut(tgt, util.lgb_loose, half='all', estimator=est)['Z']
        zw = bco.evaluate_cut(tgt_w, util.lgb_loose, half='all', estimator=est)['Z']
        ok &= check(f'constant weight 0.5, {est}: Z^2 ratio {zw ** 2 / z1 ** 2:.6f}',
                    np.isclose(zw ** 2, 0.5 * z1 ** 2))
    ruw = bco.relative_uncertainty(tgt_w, util.lgb_loose, [], half='all')
    ok &= check('weighted: sigma_S/S without CRs = 1/Z',
                np.isclose(ruw['rel_unc'], 1 / bco.evaluate_cut(tgt_w, util.lgb_loose, half='all')['Z']))
    # luminosity scale: all yields at data luminosity
    tgt_l = bco.FOMInput(dfs, bins=coarse, lumi_scale=0.25)
    for est in ('unbiased', 'plain'):
        z1 = bco.evaluate_cut(tgt, util.lgb_loose, half='all', estimator=est)['Z']
        zl = bco.evaluate_cut(tgt_l, util.lgb_loose, half='all', estimator=est)['Z']
        ok &= check(f'lumi_scale 0.25, {est}: Z halves ({zl / z1:.6f})', np.isclose(zl, 0.5 * z1))
    rl = bco.evaluate_cut(tgt_l, util.lgb_loose, half='all')
    e1 = bco.efficiency_table(tgt, {'c': util.lgb_loose}, level='candidate')
    e4 = bco.efficiency_table(tgt_l, {'c': util.lgb_loose}, level='candidate')
    ok &= check('efficiencies and their errors do not depend on lumi_scale',
                np.allclose(e1['eff'], e4['eff'], equal_nan=True)
                and np.allclose(e1['eff_err'], e4['eff_err'], equal_nan=True)
                and np.allclose(e4['n_before'], 0.25 * e1['n_before'])
                and np.allclose(e4['n_mc_before'], e1['n_mc_before']))
    r1 = bco.evaluate_cut(tgt, util.lgb_loose, half='all')
    ok &= check(f'excess_frac does not depend on lumi_scale ({r1["excess_frac"]:.4f} vs {rl["excess_frac"]:.4f})',
                np.isclose(r1['excess_frac'], rl['excess_frac']))
    ok &= check('sigma_S/S always reported (= 1/Z without CRs)', np.isclose(rl['rel_unc'], 1 / rl['Z']))
    ok &= check('optimum without CRs does not depend on lumi_scale',
                bco.run_optimization(tgt_l, grids=bco.make_grids(tgt, n=8, half='all'), half='all',
                                     verbose=False)['best_theta']
                == bco.run_optimization(tgt, grids=bco.make_grids(tgt, n=8, half='all'), half='all',
                                        verbose=False)['best_theta'])
    qq_fix = [bco.ControlRegion.fixed('qq', 0.2, 0.05, 'B0_isContinuumEvent == 1')]
    r_raw = bco.evaluate_cut(tgt, util.lgb_loose, half='all', control_regions=qq_fix)
    r_lum = bco.evaluate_cut(tgt_l, util.lgb_loose, half='all', control_regions=qq_fix)
    ok &= check(f'with CRs: statistical part doubles at 1/4 luminosity '
                f'({r_lum["rel_unc_stat"] / r_raw["rel_unc_stat"]:.4f}), total grows',
                np.isclose(r_lum['rel_unc_stat'], 2 * r_raw['rel_unc_stat'])
                and r_lum['rel_unc'] > r_raw['rel_unc'])
    # control region with an MC input already at data luminosity: no mc_scale needed
    as_data_l = bco.FOMInput({c: df.assign(category='data') for c, df in dfs.items()})
    mc_l = bco.FOMInput(dfs, lumi_scale=1.25)
    m_l = bco.d_mass_sideband_cr(as_data_l, mc_l).measure(util.lgb_loose)
    ok &= check(f'CR with lumi-scaled MC input: R = {m_l["R"]:.4f}', np.isclose(m_l['R'], 0.8))
    import matplotlib
    matplotlib.use('Agg')
    scn = bco.n_minus_1_scans(tgt_l, util.lgb_loose, bco.make_grids(tgt, n=5, half='all'), half='all')
    fz = bco.plot_n_minus_1(scn, sigma_z=0.5, z_ref=rl['Z'])
    fr = bco.plot_n_minus_1(scn, sigma_z=0.5, z_ref=rl['Z'], y='rel_unc')
    ok &= check('plot_n_minus_1 y=Z / y=rel_unc',
                'Z' in fz.axes[0].get_ylabel() and 'sigma_S' in fr.axes[0].get_ylabel())

    # per-region binning: 'auto' by default (quantile range + overflow folded into the edge bins)
    off_def = bco.offres_cr(as_data_l, mc_l)
    sb_def = bco.d_mass_sideband_cr(as_data_l, mc_l)
    ok &= check('control regions default to auto binning with overflow folding',
                off_def.fold_overflow and sb_def.fold_overflow
                and not np.array_equal(off_def.bins[0], bco.CR_BINS[0]))
    # after BCS, the D_M signal window holds n_region data events; all must be in the histogram
    n_region = 0
    for c, ch in as_data_l.channels.items():
        sel = bco.select(as_data_l, c, bco.OPEN_CUT, 'all')
        n_region += int(((ch.dm[sel] > 1.855) & (ch.dm[sel] < 1.885)).sum())
    h_d = off_def.histograms(bco.OPEN_CUT)[0]
    ok &= check(f'folding: every event of the region is counted ({h_d.sum():.0f} of {n_region})',
                np.isclose(h_d.sum(), n_region))
    fixed_nofold = bco.ControlRegion('nofold', as_data_l, mc_l, 'B0_isContinuumEvent == 1',
                                     bins=bco.CR_BINS, fold_overflow=False)
    ok &= check('fixed narrow binning without folding loses events',
                fixed_nofold.histograms(bco.OPEN_CUT)[0].sum() < n_region)
    # one extreme outlier must neither stretch the range nor be lost
    out_df = {c: df.copy() for c, df in dfs.items()}
    k = out_df['e'].index[(out_df['e'].D_M > 1.86) & (out_df['e'].D_M < 1.88)][0]
    out_df['e'].loc[k, 'B0_recMissM2'] = 500.0
    data_o = bco.FOMInput({c: df.assign(category='data') for c, df in out_df.items()})
    cr_o = bco.offres_cr(data_o, bco.FOMInput(out_df, lumi_scale=1.25))
    ok &= check(f'outlier at M2_miss = 500 does not stretch the auto range (upper edge {cr_o.bins[0][-1]})',
                cr_o.bins[0][-1] < 20)
    ch_o = data_o.channels['e']
    flat = cr_o._cr_bins(ch_o)[ch_o.x == 500.0]
    ny = len(cr_o.bins[1]) - 1
    ok &= check('outlier event is counted in the last M2_miss bin',
                len(flat) == 1 and flat[0] // ny == len(cr_o.bins[0]) - 2)

    # 'auto' sets the range of both fit variables from the region's own events
    win = lambda ch: (ch.dm > 1.855) & (ch.dm < 1.885)
    for k_ax, (name_ax, arr) in enumerate((('M2_miss', 'x'), ('p_D + p_l', 'y'))):
        vals = np.concatenate([getattr(ch, arr)[win(ch)] for ch in mc_l.channels.values()])
        e = off_def.bins[k_ax]
        inside = np.mean((vals >= e[0]) & (vals <= e[-1]))
        ok &= check(f"'auto' {name_ax} range [{e[0]:.1f}, {e[-1]:.1f}] covers {100 * inside:.2f}% of the region",
                    inside > 0.995 and len(e) == 21 and (e[-1] - e[0]) < 1.2 * (vals.max() - vals.min()))

    # default reference for the auto range: the BCS candidate per event without a BDT cut
    ok &= check('auto range reference defaults to the open cut',
                bco.CR_AUTO_REF_CUTS == (bco.OPEN_CUT,)
                and all(np.array_equal(a, b) for a, b in zip(
                    off_def.bins, bco.auto_cr_bins([as_data_l, mc_l], bco.D_MASS_SR_WINDOW,
                                                   ref_cuts=bco.OPEN_CUT))))

    # every region has its own binning and uses it for all its numbers: shift M2_miss
    # and p_D + p_l of the D_M sideband candidates, so the two regions' ranges differ
    sh_df = {}
    for c, df in dfs.items():
        df = df.copy()
        sb = (df.D_M < 1.85) | (df.D_M > 1.9)
        df.loc[sb, 'B0_recMissM2'] += 6.0
        df.loc[sb, 'p_D_l'] += 1.5
        sh_df[c] = df
    d_sh = bco.FOMInput({c: df.assign(category='data') for c, df in sh_df.items()})
    m_sh = bco.FOMInput(sh_df, lumi_scale=1.25)
    sb_sh = bco.d_mass_sideband_cr(d_sh, m_sh)      # same inputs, built one after the other
    off_sh = bco.offres_cr(d_sh, m_sh)
    ok &= check(f'regions get their own binning (M2_miss upper edge: sidebands '
                f'{sb_sh.bins[0][-1]:.1f}, offres {off_sh.bins[0][-1]:.1f})',
                sb_sh.bins[0][-1] - off_sh.bins[0][-1] > 4.0
                and sb_sh.bins[1][-1] - off_sh.bins[1][-1] > 1.0)

    def manual_hist(cr, inp, window):
        """histogram of the region's BCS candidates at the open cut in cr.bins
        (values beyond the range clipped into the edge bins)"""
        bx, by = cr.bins
        h = np.zeros((len(bx) - 1, len(by) - 1))
        for c, ch in inp.channels.items():
            sel = bco.select(inp, c, bco.OPEN_CUT, 'all')
            inw = np.zeros(len(sel), dtype=bool)
            for lo, hi in window:
                inw |= (ch.dm[sel] > lo) & (ch.dm[sel] < hi)
            x = np.clip(ch.x[sel][inw], bx[0], bx[-1])
            y = np.clip(ch.y[sel][inw], by[0], by[-1])
            h += np.histogram2d(x, y, bins=(bx, by), weights=ch.w[sel][inw])[0]
        return h

    for cr_k, win_k in ((sb_sh, bco.D_MASS_SB_WINDOW), (off_sh, bco.D_MASS_SR_WINDOW)):
        d_k, m_k, _ = cr_k.histograms(bco.OPEN_CUT)
        meas_k = cr_k.measure(bco.OPEN_CUT)
        ref_k = bco.excess_variance(d_k, m_k, model_err=cr_k.histograms(bco.OPEN_CUT)[2],
                                    normalize=True, min_count=cr_k.min_count)
        of_k = cr_k.overflow_fraction(bco.OPEN_CUT)['data']
        ok &= check(f'{cr_k.name}: histograms, R, delta, chi2 and overflow use its own binning',
                    d_k.shape == (len(cr_k.bins[0]) - 1, len(cr_k.bins[1]) - 1)
                    and np.allclose(d_k, manual_hist(cr_k, d_sh, win_k))
                    and np.allclose(m_k, manual_hist(cr_k, m_sh, win_k))
                    and np.isclose(meas_k['n_data'], d_k.sum())
                    and np.isclose(meas_k['R'], ref_k['R']) and np.isclose(meas_k['chi2'], ref_k['chi2'])
                    and max(of_k.values()) < 0.01)
    off_alone = bco.offres_cr(d_sh, m_sh)
    ok &= check('a region built next to another on the same inputs gives the same result as alone',
                all(np.array_equal(a, b) for a, b in zip(off_alone.bins, off_sh.bins))
                and np.allclose(off_alone.histograms(util.lgb_loose)[0], off_sh.histograms(util.lgb_loose)[0]))

    # halves: signal-region yields scaled to the full luminosity, control regions split too
    ra, rall = (bco.evaluate_cut(tgt_l, util.lgb_loose, half=h) for h in ('A', 'all'))
    ok &= check(f"half A yields scaled to the full luminosity (S_A = {ra['S']:.1f}, S_all = {rall['S']:.1f})",
                abs(ra['S'] / rall['S'] - 1) < 0.1 and bco.HALF_LUMI_FACTOR == 2.0)
    m_all, m_a, m_b = (sb_sh.measure(util.lgb_loose, half=h) for h in ('all', 'A', 'B'))
    d_a, _, _ = sb_sh.histograms(util.lgb_loose, half='A')
    d_b, _, _ = sb_sh.histograms(util.lgb_loose, half='B')
    ok &= check(f"CR on half A / B uses that half of the data ({m_a['n_data']:.0f} + {m_b['n_data']:.0f} "
                f"= {m_all['n_data']:.0f})",
                np.isclose(m_a['n_data'] + m_b['n_data'], m_all['n_data'])
                and np.isclose(m_a['n_data'], d_a.sum()) and m_a['half'] == 'A'
                and abs(m_a['n_data'] / m_all['n_data'] - 0.5) < 0.1)
    ok &= check(f"CR R_err on a half scaled to the full data set ({m_a['R_err_half']:.4f} -> {m_a['R_err']:.4f}, "
                f"all: {m_all['R_err']:.4f})",
                np.isclose(m_a['R_err'], m_a['R_err_half'] / np.sqrt(2))
                and abs(m_a['R_err'] / m_all['R_err'] - 1) < 0.15)
    ru_a = bco.relative_uncertainty(tgt_l, util.lgb_loose, [sb_sh], half='A')
    ok &= check('relative_uncertainty passes the half to the control regions',
                np.isclose(ru_a[f'cr_{sb_sh.name}_R'], m_a['R'])
                and np.isclose(ru_a[f'cr_{sb_sh.name}_n'], m_a['n']))
    # changing only half-B data leaves the half-A measurement untouched
    sh_b = {c: df.assign(category='data') for c, df in sh_df.items()}
    for c, df in sh_b.items():
        odd = df['__event__'] % 2 == 1
        df.loc[odd, 'B0_recMissM2'] += 2.0
    sb_mod = bco.d_mass_sideband_cr(bco.FOMInput(sh_b), m_sh, bins=sb_sh.bins)
    ok &= check('CR measurement on half A is independent of the half-B data',
                np.isclose(sb_mod.measure(util.lgb_loose, half='A')['delta'], m_a['delta'])
                and np.isclose(sb_mod.measure(util.lgb_loose, half='A')['R'], m_a['R'])
                and not np.isclose(sb_mod.measure(util.lgb_loose, half='B')['chi2'], m_b['chi2']))
    sb_nos = bco.d_mass_sideband_cr(d_sh, m_sh, split_halves=False)
    ok &= check('split_halves=False: halves use all control-region data',
                np.isclose(sb_nos.measure(util.lgb_loose, half='A')['n_data'], m_all['n_data']))

    # data/MC per slice of one BDT output: slices add up to the region at the loosest cut
    rs = bco.ratio_slices(off_sh, util.lgb_loose, 'continuum_prob')
    th_open = dict(bco.as_theta(util.lgb_loose), continuum_prob=1.0)
    m_open = off_sh.measure(th_open)
    ok &= check(f"ratio_slices add up to the region ({rs['n_data'].sum():.0f} vs {m_open['n_data']:.0f}), "
                f"R_total = R ({rs.attrs['R_total']:.4f})",
                np.isclose(rs['n_data'].sum(), m_open['n_data']) and np.isclose(rs['n_mc'].sum(), m_open['n_mc'])
                and np.isclose(rs.attrs['R_total'], m_open['R']))
    # a mismodeled output: data events with high continuum_prob get duplicated
    dup = {c: pd.concat([df.assign(category='data'),
                         df[df['continuum_prob'] > 0.3].assign(category='data', __event__=df['__event__'] + 10 ** 7)])
           for c, df in sh_df.items()}
    rs_dup = bco.ratio_slices(bco.offres_cr(bco.FOMInput(dup), m_sh, bins=off_sh.bins),
                              util.lgb_loose, 'continuum_prob')
    ok &= check(f"ratio_slices see an output-dependent data/MC ratio "
                f"(chi2 {rs_dup.attrs['chi2_shape']:.0f}/{rs_dup.attrs['ndf_shape']} vs "
                f"{rs.attrs['chi2_shape']:.0f}/{rs.attrs['ndf_shape']})",
                rs_dup.attrs['chi2_shape'] > 5 * max(rs_dup.attrs['ndf_shape'], 1) > 0
                and rs.attrs['chi2_shape'] < 3 * max(rs.attrs['ndf_shape'], 1))
    fs = bco.plot_ratio_slices({'offres': rs, 'offres (dup)': rs_dup})
    ok &= check('plot_ratio_slices', len(fs.axes) == 4)

    # the auto range follows the BCS candidates, not all candidates: in each event the
    # best-ranked candidate has a hard p_D + p_l in 5 % of the events, the many other
    # candidates are soft (0.05 % of all candidates are hard, so a candidate-level
    # 99.9 % quantile would miss them; this is the off-resonance situation)
    rng_h = np.random.default_rng(21)
    n_ev, n_c = 1000, 100
    ev_h = np.repeat(np.arange(n_ev), n_c)
    rank_h = rng_h.uniform(1, 10, n_ev * n_c)
    best_h = np.zeros(n_ev * n_c, dtype=bool)
    best_h[np.arange(n_ev) * n_c] = True
    rank_h[best_h] = 0.5
    hard_h = best_h & np.repeat(rng_h.random(n_ev) < 0.05, n_c)
    df_h = pd.DataFrame({'sig_prob': 0.5, 'fakeD_prob': 0.1, 'continuum_prob': 0.2,
                         'combinatorial_prob': 0.2,
                         'B0_recMissM2': rng_h.normal(0, 2, n_ev * n_c),
                         'p_D_l': np.where(hard_h, rng_h.uniform(4.5, 6.5, n_ev * n_c),
                                           rng_h.uniform(0.3, 3.0, n_ev * n_c)),
                         'D_M': 1.87, 'B_D_ReChi2': rank_h,
                         '__experiment__': 26, '__run__': 1, '__event__': ev_h, '__production__': 1,
                         'category': 'data', 'B0_isContinuumEvent': 1})
    in_h = bco.FOMInput({'e': df_h})
    cr_h = bco.offres_cr(in_h, in_h)
    hi_cand = np.quantile(df_h['p_D_l'], 0.999)
    ok &= check(f'auto p_D + p_l range follows the selected events (upper edge {cr_h.bins[1][-1]:.1f}, '
                f'candidate-level 99.9 % quantile {hi_cand:.2f})',
                cr_h.bins[1][-1] >= 6.0 and hi_cand < 3.1)
    of_h = cr_h.overflow_fraction(bco.OPEN_CUT)
    ok &= check(f"overflow fraction small with the auto range ({of_h['data']['p_D_l high']:.4f})",
                of_h['data']['p_D_l high'] < 0.01)
    cr_n = bco.offres_cr(in_h, in_h, bins=[cr_h.bins[0], np.linspace(0.3, 3.0, 21)])
    of_n = cr_n.overflow_fraction(bco.OPEN_CUT)
    ok &= check(f"overflow fraction reports a too narrow range ({of_n['data']['p_D_l high']:.3f})",
                abs(of_n['data']['p_D_l high'] - 0.05) < 0.02
                and np.isclose(cr_n.histograms(bco.OPEN_CUT)[0].sum(), n_ev))
    cr_f = bco.offres_cr(in_h, in_h, auto_kw=dict(ranges=(None, (0.0, 6.5))))
    ok &= check('auto_kw ranges fixes the p_D + p_l range',
                cr_f.bins[1][0] == 0.0 and cr_f.bins[1][-1] == 6.5
                and np.array_equal(cr_f.bins[0], cr_h.bins[0]))

    # sparse regions: shape from the M2_miss projection, or not at all
    cr_s = bco.offres_cr(as_data_l, mc_l)
    rng_s = np.random.default_rng(3)
    mx = np.outer(np.exp(-np.linspace(0, 3, 20)), np.ones(20))          # 20 x 20 expectation
    # util.excess_variance merges adjacent low-count bins (low_count='merge', its default) into
    # groups of >= 10 expected events. The number of usable groups is then about total / 10 in 2D,
    # and never more in the M2_miss projection: 400 events still give a 2D measurement, and when
    # 2D has too few groups the projection has too few as well. Merging is the chosen rule
    # (docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md A.5); with low_count='drop' 400 events would fall
    # back to 1D.
    for total, expect in ((2e6, '2d'), (400, '2d'), (40, 'none')):
        m_h = total * mx / mx.sum()
        d_h = rng_s.poisson(m_h * (1 + rng_s.normal(0, 0.05, 20))[:, None])  # 5 % distortion in M2_miss
        r_s = cr_s._shape_measurement(d_h.astype(float), m_h, np.zeros_like(m_h))
        ok &= check(f'shape measurement with {total:.0e} events: {r_s["shape_dim"]} '
                    f'(delta = {r_s["delta"] if np.isfinite(r_s["delta"]) else float("nan"):.3f})',
                    r_s['shape_dim'] == expect)
    ok &= check('the region reports which shape measurement it used',
                cr_s.measure(util.lgb_loose)['shape_dim'] in ('2d', '1d', 'none'))

    # an empty selection gives Z = 0 and sigma_S/S = inf, not a crash
    nothing = {'sig_prob': 0.999, 'fakeD_prob': 1e-4, 'continuum_prob': 1e-4, 'combinatorial_prob': 1e-4}
    re_ = bco.evaluate_cut(tgt, nothing, half='all')
    ok &= check('empty selection: Z = 0, sigma_S/S = inf', re_['Z'] == 0 and re_['rel_unc'] == np.inf)

    # ---------------- Optuna ----------------
    try:
        import optuna  # noqa: F401
        have_optuna = True
    except ImportError:
        have_optuna = False
        print('SKIP  optuna not installed')
    if have_optuna:
        g8 = bco.make_grids(tgt_l, n=10, half='A')
        cd = bco.run_optimization(tgt_l, grids=g8, half='A', verbose=False)
        op = bco.run_optuna(tgt_l, n_trials=120, grids=g8, half='A', n_boot=10, verbose=False)
        on_grid = all(op['best_theta'][v] in set(op['grids'][v]) for v in bco.MVA_OUTPUTS)
        ok &= check('optuna: chosen working point is on the grid', on_grid)
        ok &= check('optuna: chosen by the validation half',
                    op['summary']['Z_val'].iloc[0] == op['summary']['Z_val'].max())
        z_cd = bco.evaluate_cut(tgt_l, cd['best_theta'], half='A')['Z']
        ok &= check(f'optuna best trial within 5% of coordinate descent '
                    f'({op["summary"]["Z_opt"].max():.3f} vs {z_cd:.3f})',
                    op['summary']['Z_opt'].max() >= 0.95 * z_cd)
        op2 = bco.run_optuna(tgt_l, n_trials=120, grids=g8, half='A', n_boot=10, verbose=False)
        ok &= check('optuna: same seed, same result', op2['best_theta'] == op['best_theta'])
        opc = bco.run_optuna(tgt_l, n_trials=60, grids=g8, half='A', n_boot=5, sampler='cmaes',
                             verbose=False)
        ok &= check('optuna: CMA-ES sampler runs', len(opc['summary']) > 0)
        crs_t = [bco.ControlRegion.fixed('qq', 0.15, 0.05, 'B0_isContinuumEvent == 1')]
        par = bco.run_optuna_pareto(tgt_l, crs_t, n_trials=150, grids=g8, half='A', verbose=False)
        fr = par['front']
        dominated = any(((fr.rel_unc_stat <= r.rel_unc_stat) & (fr.rel_unc_syst <= r.rel_unc_syst)
                         & ((fr.rel_unc_stat < r.rel_unc_stat) | (fr.rel_unc_syst < r.rel_unc_syst))).any()
                        for r in fr.itertuples())
        ok &= check(f'pareto: {len(fr)} front points, none dominated by another', len(fr) > 0 and not dominated)
        chk = bco.evaluate_cut(tgt_l, par['best_theta'], half='A', control_regions=crs_t)
        ok &= check('pareto: total = stat (+) syst, consistent with evaluate_cut',
                    np.allclose(fr.rel_unc ** 2, fr.rel_unc_stat ** 2 + fr.rel_unc_syst ** 2)
                    and np.isclose(chk['rel_unc'], fr.rel_unc.min()))
        try:
            bco.run_optuna_pareto(tgt_l, [], n_trials=5)
            ok &= check('pareto without control regions rejected', False)
        except ValueError:
            ok &= check('pareto without control regions rejected', True)
        fp = bco.plot_pareto(par, data=tgt_l, control_regions=crs_t, half='A',
                             marks={'lgb_tight': util.lgb_tight, 'nothing passes': nothing})
        ok &= check('plot_pareto runs, empty-selection mark labeled',
                    any('no bin passes' in tx.get_text() for tx in fp.axes[0].get_legend().get_texts()))

    # MC corrections: default none (weight 1), registered ones multiply
    frame = dfs['e'].head(50)
    w0 = bco.apply_corrections(frame, 'e', 'run2')
    ok &= check('no corrections: mc_weight = 1 and no other columns',
                list(w0.columns) == [bco.MC_WEIGHT] and (w0[bco.MC_WEIGHT] == 1).all())
    bco.register_correction('_test_a', lambda df, mode, run, k=2.0: np.full(len(df), k))
    bco.register_correction('_test_b', lambda df, mode, run: np.where(df['sig_prob'] > 0.5, 0.5, 1.0))
    w2 = bco.apply_corrections(frame, 'e', 'run2', ['_test_a', '_test_b'], {'_test_a': {'k': 3.0}})
    ok &= check('two corrections: per-correction columns and their product',
                np.allclose(w2[bco.MC_WEIGHT], w2['w__test_a'] * w2['w__test_b'])
                and np.allclose(w2['w__test_a'], 3.0))
    for bad, kw in ((['_nope'], dict(run='run2')), (['_test_a'], dict(run=None))):
        try:
            bco.apply_corrections(frame, 'e', corrections=bad, **kw)
            ok &= check(f'bad correction request {bad} {kw} rejected', False)
        except ValueError:
            ok &= check(f'bad correction request {bad} {kw} rejected', True)
    for name in ('_test_a', '_test_b'):
        bco.CORRECTIONS.pop(name)
    ok &= check("'pid' is registered", 'pid' in bco.CORRECTIONS)

    # weighted excess-free estimator: no excess for weights scattered around 1
    rng = np.random.default_rng(11)
    for (s_exp, b_exp, spread) in ((5, 5, 0.3), (2, 8, 0.1)):
        plain, unb, target = [], [], []
        for _ in range(20000):
            ks, kb = rng.poisson(s_exp), rng.poisson(b_exp)
            if ks + kb < 10:
                plain.append(0.0); unb.append(0.0); target.append(0.0)
                continue
            ws = 0.95 + spread * rng.standard_normal(ks)
            wb = 0.95 + spread * rng.standard_normal(kb)
            S_, B_ = np.array([ws.sum()]), np.array([wb.sum()])
            num, den = bco._unbiased_terms(S_, B_, np.array([(ws ** 2).sum()]),
                                           np.array([(ws ** 2).sum() + (wb ** 2).sum()]), 0.0)
            plain.append((S_ ** 2 / (S_ + B_))[0])
            unb.append((num / den)[0])
            target.append(0.95 * (ks + kb) * (s_exp / (s_exp + b_exp)) ** 2)
        tg = np.mean(target)
        ok &= check(f'weighted estimator s={s_exp} b={b_exp} spread={spread}: target {tg:.4f}, '
                    f'unbiased {np.mean(unb):.4f}, plain {np.mean(plain):.4f}',
                    abs(np.mean(unb) / tg - 1) < 0.02 and np.mean(plain) / tg - 1 > 0.04)

    r = bco.evaluate_cut(data, bco.OPEN_CUT, half='all')
    ok &= check(f'excess_frac reported ({r["excess_frac"]:.3f})', 0 <= r['excess_frac'] < 1)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ok &= test_beam_energy_correction(tmp)

    print('\nALL PASS' if ok else '\nSOME TESTS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())