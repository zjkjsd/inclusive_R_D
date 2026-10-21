#!/usr/bin/env python3
"""Check bdt_cut_optimization against a direct pandas implementation.

The module replaces "query(cut) -> groupby().idxmin() -> D_M window ->
histogram2d" by pre-sorted numpy arrays and bincount, for speed. These tests
build synthetic multi-candidate events (no ntuples needed) and require that
both routes give the same selected candidates and the same Z, for both BCS
orders, both halves and with NaN BCS ranks present. They then run the full
optimisation on a toy where the right answer is known in direction.

    python3 scripts/tests/test_bdt_cut_optimization.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
        'experiment': 26, 'run': ev // 1000, 'event': ev, 'production': 1,
        'category': cat,
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

    d = df if half == 'all' else df[df['event'] % 2 == {'A': 0, 'B': 1}[half]]
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
                        z_ref = np.sqrt(sum(reference(dfs[c], bco.as_theta(th), half, order, base,
                                                      min_count, count_on, bins, est)[0] ** 2
                                            for c in dfs))
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
                sorted(np.float32(best.sort_values('event')['sig_prob']))
                == sorted(p for _, p in fast_keys))

    # the full optimisation runs and does not end below its starting points
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

    r = bco.evaluate_cut(data, bco.OPEN_CUT, half='all')
    ok &= check(f'excess_frac reported ({r["excess_frac"]:.3f})', 0 <= r['excess_frac'] < 1)

    print('\nALL PASS' if ok else '\nSOME TESTS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
