#!/usr/bin/env python3
"""Analysis of the B-decay generator truth (piece 3, point 3.2): classification, PYTHIA hadron
content, inclusive charm rates of the MC, and the priors for N2 and the D_s-class scale r_s.

Input: Samples/mc_truth/truth_mixed.root, truth_charged.root (b_decay_truth_basf2.py).
Needs uproot, numpy, pandas and scripts/tools (decay_table, pdg_match).

Run:  python3 scripts/mc_truth/truth_analysis.py [--toys 20000]
Writes Notebooks/Plotting/csv_output/truth_b_decays_summary.csv and prints the tables.

Definitions
-----------
* Exclusive: the daughter set (PHOTOS photons removed, by their generator flag) is a final state
  listed for the B's own flavor in the decay file. The Belle II EvtGen patch
  `evtgen-02-00-01-avoid-double-counting` makes PYTHIA redecay a B whose first-generation daughters
  equal a listed set, so a listed set cannot come from PYTHIA. Everything else is PYTHIA.
* Upper / lower vertex: for a B with a b-bar quark (B0, B+; PDG > 0) the upper-vertex charm is a
  c quark (D0, D+, D_s+, Lambda_c+, Xi_c, Omega_c), the lower-vertex one a c-bar; reversed for B0bar, B-.
  Charmonium states are counted separately (every generated state, so a J/psi from psi(2S) or
  chi_c is counted again); N_up / N_lo count open-charm hadrons only.
* Rates are multiplicities per B decay (as the measurements), in %.
* Classes of piece 3 (docs/B0_PIECE3_REVIEW.md, 3.1): 'meas' = exclusive line of class measured;
  every exclusive decay has the weight c of the piece-2 table (vetoes 0, half_limit lines < 1); 'unm2c' = exclusive line of class none/limit with >= 2 open-charm hadrons
  among the B daughters; 'py2c' = PYTHIA decay with >= 2 open-charm hadrons among the B daughters;
  'other' = the rest (single charm, charmonium, charmless, semileptonic, PYTHIA 1 charm).
  D_s class: 'unm2c' or 'py2c' with a D_s-family meson among the B daughters.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'tools'))
import decay_table as dt          # noqa: E402
import pdg_match as pm            # noqa: E402

N_DAU = 12
EVT = ['__experiment__', '__run__', '__event__']
DS_FAMILY = {431, 433, 10431, 20433, 10433, 435}
# charm species of the chain trees: name -> (abs PDG codes, kind)
SPECIES = {'D0': {421}, 'Dp': {411}, 'Ds': {431}, 'Lc': {4122}, 'Xic': {4132, 4232, 4332}}
CHARMONIUM = {'Jpsi': 443, 'psi2S': 100443, 'chic0': 10441, 'chic1': 20443, 'chic2': 445, 'etac': 441}
CHARM_TREES = ['charm_D0', 'charm_Dplus', 'charm_D_splus', 'charm_Lambda_cplus', 'charm_Xi_c0',
               'charm_Xi_cplus', 'charm_Omega_c0', 'charm_Jpsi', 'charm_psi2S', 'charm_chi_c0',
               'charm_chi_c1', 'charm_chi_c2', 'charm_eta_c']


def to_df(tree):
    """Numeric branches only (variablesToNtuple also stores the event type as a string)."""
    keys = [k for k in tree.keys() if 'string' not in tree[k].typename]
    return tree.arrays(keys, library='pd')


def load(path, btree):
    f = uproot.open(path)
    b = to_df(f[btree])
    b = b.rename(columns={'PDG': 'b_pdg'})
    b['b_pdg'] = b['b_pdg'].astype(int)
    b['bid'] = b['genParticleID'].astype(int)
    c = pd.concat([to_df(f[t]).assign(tree=t) for t in CHARM_TREES], ignore_index=True)
    c['pdg'] = c['PDG'].astype(int)
    c['bid'] = c['ancestorBIndex'].astype(int)
    return b, c


def classify(b, mt, tab_lookup, tab):
    """Exclusive/PYTHIA class, decay-file line, piece-2 class and c, piece-3 class of every B."""
    pdl = mt.pdl
    D = b[[f'd{i}_PDG' for i in range(N_DAU)]].to_numpy().astype(int)
    P = b[[f'd{i}_photos' for i in range(N_DAU)]].to_numpy().astype(int)
    kept = np.where(P == 1, dt.EMPTY, D)
    out = dict(gen=[], line=[], order_ok=[], pclass=[], c=[], key=[])
    for row, m in zip(kept, b['b_pdg'].to_numpy()):
        d = [int(x) for x in row if x != dt.EMPTY]
        key = dt.mode_key(d)
        j = mt.lookup.get((int(m), key))
        out['key'].append(key)
        if j is None:
            out['gen'].append('pythia'); out['line'].append(-1); out['order_ok'].append(True)
            out['pclass'].append(''); out['c'].append(1.0)
            continue
        mode = mt.modes.iloc[j]
        out['gen'].append('exclusive'); out['line'].append(int(mode['lines'][0]))
        out['order_ok'].append(tuple(d) in {tuple(o) for o in mode['orders']})
        k = tab_lookup.get((int(m), key))
        if k is None:
            out['pclass'].append('?'); out['c'].append(1.0)
        else:
            out['pclass'].append(tab['pdg_class'].iat[k]); out['c'].append(float(tab['c'].iat[k]))
    for k, v in out.items():
        b[k] = v
    props = dt.decay_properties(np.where(kept == dt.EMPTY, dt.EMPTY, kept), pdl)
    b['n_charm'] = props['n_charm']
    b['is_sl'] = props['is_sl']
    b['has_ds'] = np.isin(np.abs(kept), list(DS_FAMILY)).any(axis=1)
    two_c = b['n_charm'] >= 2
    exc = b['gen'] == 'exclusive'
    b['cls'] = np.select([exc & (b['pclass'] == 'measured'),
                          exc & b['pclass'].isin(['none', 'limit']) & two_c & ~b['is_sl'],
                          ~exc & two_c],
                         ['meas', 'unm2c', 'py2c'], 'other')
    # every exclusive decay keeps the c of the piece-2 table (vetoes c = 0, half_limit, ...);
    # PYTHIA decays have weight 1 (fixed at 2026-11-20 after the independent check)
    b['ds_class'] = b['cls'].isin(['unm2c', 'py2c']) & b['has_ds']
    return b


def chain_counts(b, c):
    """Per B: upper/lower-vertex counts of each charm species, charmonium counts, n_c."""
    c = c[c['bid'] >= 0]
    bsign = b.set_index(EVT + ['bid'])['b_pdg'].map(np.sign)
    c = c.join(bsign.rename('bsign'), on=EVT + ['bid'])
    up = np.sign(c['pdg']) == c['bsign']           # b-bar B (sign +) -> upper vertex = charm (sign +)
    cols = {}
    for name, codes in SPECIES.items():
        isx = c['pdg'].abs().isin(codes)
        cols[f'{name}_up'] = isx & up
        cols[f'{name}_lo'] = isx & ~up
    for name, code in CHARMONIUM.items():
        cols[name] = c['pdg'] == code
    flags = pd.DataFrame(cols).astype(int)
    flags[EVT + ['bid']] = c[EVT + ['bid']]
    cnt = flags.groupby(EVT + ['bid']).sum()
    b = b.join(cnt, on=EVT + ['bid']).fillna({k: 0 for k in cnt.columns})
    open_up = sum(b[f'{n}_up'] for n in SPECIES)
    open_lo = sum(b[f'{n}_lo'] for n in SPECIES)
    ccbar = sum(b[n] for n in ('Jpsi', 'psi2S', 'chic0', 'chic1', 'chic2', 'etac'))
    b['N_up'] = open_up
    b['N_lo'] = open_lo
    b['n_ccbar'] = ccbar
    return b


def rates(b, w=None, by='flavor'):
    """Multiplicity rates (%) per B flavor group for the given weights."""
    w = np.ones(len(b)) if w is None else np.asarray(w)
    cols = [f'{n}_up' for n in SPECIES] + [f'{n}_lo' for n in SPECIES] + list(CHARMONIUM) + ['N_up', 'N_lo']
    out = {}
    for grp, idx in b.groupby(b['b_pdg'].abs()).groups.items():
        ww = w[b.index.get_indexer(idx)]
        sub = b.loc[idx]
        # per B decay (the number of B decays is fixed): no renormalization of the weights
        out[{511: 'B0', 521: 'B+'}[grp]] = {k: 100 * np.sum(ww * sub[k]) / len(sub) for k in cols}
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--toys', type=int, default=20000)
    a = ap.parse_args()
    mt = dt.build_mode_table()
    tab = pm.read_table(os.path.join(ROOT, 'Notebooks/Plotting/csv_output/b_mode_pdg_table.csv'))
    tab_lookup = {(int(m), tuple(k)): i for i, (m, k) in enumerate(zip(tab['mother_pdg'], tab['key']))}
    parts = []
    for fn, tree in (('truth_mixed.root', 'B0'), ('truth_charged.root', 'Bplus')):
        b, c = load(os.path.join(ROOT, 'Samples/mc_truth', fn), tree)
        b = classify(b, mt, tab_lookup, tab)
        b = chain_counts(b, c)
        parts.append(b)
    b = pd.concat(parts, ignore_index=True)

    # ---- 1. classification against the decay file
    print('\n=== 1. Classification (generator truth)')
    for grp, sub in b.groupby(b['b_pdg'].abs()):
        n = len(sub)
        fr = sub['gen'].value_counts(normalize=True)
        exp_excl = mt.exclusive_bf[grp] if grp in mt.exclusive_bf else np.nan
        print(f'  |B| = {grp}: {n} decays; exclusive {100 * fr.get("exclusive", 0):.2f} % '
              f'(decay file {100 * exp_excl:.2f} %), PYTHIA {100 * fr.get("pythia", 0):.2f} % '
              f'(decay file {100 * mt.pythia_bf[grp]:.2f} %); exclusive with an order not in the file: '
              f'{(~sub.loc[sub.gen == "exclusive", "order_ok"]).sum()}')
    # per-mode check: observed fraction vs BF for the 25 largest exclusive modes per flavor
    rows = []
    for m in (511, -511, 521, -521):
        sub = b[b['b_pdg'] == m]
        n = len(sub)
        obs = sub[sub.gen == 'exclusive'].groupby('key').size()
        modes = mt.modes[mt.modes['mother_pdg'] == m].set_index('key')
        for key, bf in modes['bf_mc'].items():
            if bf <= 0 and key not in obs.index:
                continue
            o = int(obs.get(key, 0))
            e = n * bf
            rows.append(dict(mother=m, key=key, line=modes.loc[[key], 'lines'].iloc[0][0], bf=bf, obs=o, exp=e,
                             pull=(o - e) / np.sqrt(e) if e > 0 else np.nan))
    chk = pd.DataFrame(rows)
    big = chk[chk['exp'] > 25]
    chi2 = np.nansum(big['pull'] ** 2)
    print(f'  modes with > 25 expected decays: {len(big)}; chi2 = {chi2:.0f} for {len(big)} modes; '
          f'|pull| > 4: {(big.pull.abs() > 4).sum()}')
    print(big.reindex(big.pull.abs().sort_values(ascending=False).index).head(10)
          [['mother', 'line', 'bf', 'obs', 'exp', 'pull']].to_string(index=False))
    print(f'  decays on zero-BF lines: {chk[(chk.bf == 0) & (chk.obs > 0)].obs.sum()}')

    # ---- 2. class shares and PYTHIA content
    print('\n=== 2. Classes (% of B decays) and the upper-vertex charm of all PYTHIA decays (fractions per upper-vertex c quark)')
    for grp, sub in b.groupby(b['b_pdg'].abs()):
        cls = 100 * sub['cls'].value_counts(normalize=True)
        print(f'  |B| = {grp}: ' + ', '.join(f'{k} {v:.2f}' for k, v in cls.items()) +
              f'; D_s class {100 * sub.ds_class.mean():.2f} (unm2c {100 * (sub.ds_class & (sub.cls == "unm2c")).mean():.2f},'
              f' py2c {100 * (sub.ds_class & (sub.cls == "py2c")).mean():.2f})')
        py = sub[sub['gen'] == 'pythia']
        up_c = py[['Ds_up', 'D0_up', 'Dp_up', 'Lc_up', 'Xic_up']].sum(axis=1)
        with_up = (up_c > 0) | (py['n_ccbar'] > 0)
        tot = up_c.sum() + py.loc[up_c == 0, 'n_ccbar'].clip(upper=1).sum()
        f = {'D_s': py['Ds_up'].sum() / tot, 'D (D0+D+)': (py['D0_up'] + py['Dp_up']).sum() / tot,
             'Lambda_c': py['Lc_up'].sum() / tot, 'Xi_c, Omega_c': py['Xic_up'].sum() / tot,
             'charmonium (no other upper c)': py.loc[up_c == 0, 'n_ccbar'].clip(upper=1).sum() / tot}
        print(f'    PYTHIA decays with an upper-vertex c quark: {100 * with_up.mean() * len(py) / len(sub):.2f} % of B decays; '
              'fractions: ' + ', '.join(f'{k} {v:.3f}' for k, v in f.items()))

    # ---- 3. inclusive rates: raw MC and with the piece-2 c
    print('\n=== 3. Inclusive rates of the MC (multiplicity per B, %)')
    r_raw = rates(b)
    r_c = rates(b, b['c'].to_numpy())
    show = pd.concat({'raw': r_raw, 'piece-2 c': r_c}, axis=1)
    print(show.round(3).to_string())
    show.to_csv(os.path.join(ROOT, 'Notebooks/Plotting/csv_output/truth_b_decays_summary.csv'))

    # ---- 4. priors for N2 and r_s from the upper-vertex D_s and D rates (BaBar 2007)
    print('\n=== 4. Priors: N2 and r_s from D_s+ X (upper) and D0 + D+ X (upper), BaBar 2007')
    meas = {511: dict(Ds=(10.3, 2.1, 1.8), D=(10.4, 1.9, 1.9)), 521: dict(Ds=(7.9, 1.4, 1.3), D=(11.1, 0.9, 0.9))}
    rng = np.random.default_rng(7)
    for grp, sub in b.groupby(b['b_pdg'].abs()):
        n = len(sub)
        w = sub['c'].to_numpy()
        fixed = ~sub['cls'].isin(['unm2c', 'py2c'])
        ds_up = sub['Ds_up'].to_numpy(); d_up = (sub['D0_up'] + sub['Dp_up']).to_numpy()
        A_s = 100 * np.sum(w[fixed] * ds_up[fixed]) / n          # fixed part of D_s+ X
        A_d = 100 * np.sum(w[fixed] * d_up[fixed]) / n
        dsc = sub['ds_class'].to_numpy(); fl = (~fixed.to_numpy())
        S_s = 100 * np.sum(ds_up[fl & dsc]) / n                    # D_s class, scaled by N2 r_s
        S_s0 = 100 * np.sum(ds_up[fl & ~dsc]) / n                  # rest of the class, scaled by N2
        S_d = 100 * np.sum(d_up[fl & dsc]) / n
        S_d0 = 100 * np.sum(d_up[fl & ~dsc]) / n
        print(f'  |B| = {grp}: D_s+ X = {A_s:.2f} + N2*({S_s0:.2f} + r_s*{S_s:.2f});  '
              f'D X(upper) = {A_d:.2f} + N2*({S_d0:.2f} + r_s*{S_d:.2f})')
        def draw(v, k):
            z = rng.standard_normal(k); return v[0] + np.where(z > 0, v[1], v[2]) * z
        Ds = draw(meas[grp]['Ds'], a.toys); Dd = draw(meas[grp]['D'], a.toys)
        # solve the 2x2 system for x = N2 and y = N2 r_s
        det = S_s0 * S_d - S_s * S_d0
        x = ((Ds - A_s) * S_d - S_s * (Dd - A_d)) / det
        y = (S_s0 * (Dd - A_d) - S_d0 * (Ds - A_s)) / det
        rs = y / x
        q = lambda v: np.percentile(v, [16, 50, 84])
        print(f'    N2 = {q(x)[1]:.2f} [{q(x)[0]:.2f}, {q(x)[2]:.2f}]; r_s = {q(rs)[1]:.2f} [{q(rs)[0]:.2f}, {q(rs)[2]:.2f}]; '
              f'raw MC: D_s+ X {A_s + S_s0 + S_s:.2f}, D X(upper) {A_d + S_d0 + S_d:.2f}')

    print('\n=== 5. Combined fit of N2 and r_s (common to B0 and B+)')
    combined_fit(b, max(a.toys // 10, 500), rng)


def combined_fit(b, toys, rng):
    """Common N2 and r_s for B0 and B+ from five measurements: BaBar 2007 upper-vertex D_s+ X and
    D0 + D+ X for B0 and B+ (hep-ex/0606026), and Belle 2023 B -> D_s X (both charges, average of
    B0 and B+, arXiv:2305.10098). Each observable is O = A + N2*S0 + N2*r_s*S1 (from the MC, piece-2 c
    on the measured lines); least squares with the measurement errors (asymmetric errors: the side
    towards the prediction), repeated on toys of the measurements for the 68 % intervals."""
    obs = {}
    for grp, sub in b.groupby(b['b_pdg'].abs()):
        n = len(sub)
        w = sub['c'].to_numpy()
        fl = sub['cls'].isin(['unm2c', 'py2c']).to_numpy()
        dsc = sub['ds_class'].to_numpy()
        for name, v in (('Ds_up', sub['Ds_up'].to_numpy()), ('D_up', (sub['D0_up'] + sub['Dp_up']).to_numpy()),
                        ('Ds_all', (sub['Ds_up'] + sub['Ds_lo']).to_numpy())):
            obs[(grp, name)] = (100 * np.sum(w[~fl] * v[~fl]) / n, 100 * np.sum(v[fl & ~dsc]) / n,
                                100 * np.sum(v[fl & dsc]) / n)
    rows = [((511, 'Ds_up'), (10.3, 2.1, 1.8)), ((521, 'Ds_up'), (7.9, 1.4, 1.3)),
            ((511, 'D_up'), (10.4, 1.9, 1.9)), ((521, 'D_up'), (11.1, 0.9, 0.9)),
            (('avg', 'Ds_all'), (11.28, 0.43, 0.43))]
    def coeff(k):
        if k[0] == 'avg':
            return tuple(0.5 * (obs[(511, k[1])][i] + obs[(521, k[1])][i]) for i in range(3))
        return obs[k]
    C = np.array([coeff(k) for k, _ in rows])            # columns A, S0, S1
    M = np.array([v for _, v in rows])
    print('  observable                 MC (N2 = r_s = 1)   measured')
    for (k, v), cc in zip(rows, C):
        print(f'    {str(k):24s} {cc.sum():8.2f}          {v[0]:.2f} +{v[1]:.2f} -{v[2]:.2f}')

    def fit(y):
        from scipy.optimize import least_squares
        def res(p):
            pred = C[:, 0] + p[0] * C[:, 1] + p[0] * p[1] * C[:, 2]
            sig = np.where(pred > y, M[:, 1], M[:, 2])
            return (y - pred) / sig
        r = least_squares(res, [1.0, 1.0], bounds=([0, 0], [5, 5]))
        return r.x, np.sum(r.fun ** 2)
    best, chi2 = fit(M[:, 0])
    print(f'  best fit: N2 = {best[0]:.2f}, r_s = {best[1]:.2f}, chi2 = {chi2:.1f} for 3 degrees of freedom')
    xs = []
    for _ in range(toys):
        z = rng.standard_normal(len(M))
        y = M[:, 0] + np.where(z > 0, M[:, 1], M[:, 2]) * z
        xs.append(fit(y)[0])
    xs = np.array(xs)
    q = lambda v: np.percentile(v, [16, 50, 84])
    a, c = q(xs[:, 0]), q(xs[:, 1])
    rho = np.corrcoef(xs[:, 0], xs[:, 1])[0, 1]
    print(f'  toys: N2 = {a[1]:.2f} [{a[0]:.2f}, {a[2]:.2f}], r_s = {c[1]:.2f} [{c[0]:.2f}, {c[2]:.2f}], '
          f'correlation {rho:.2f}')
    # leave-one-out
    for i, (k, _) in enumerate(rows):
        keep = [j for j in range(len(rows)) if j != i]
        Cs, Ms = C[keep], M[keep]
        from scipy.optimize import least_squares
        def res(p):
            pred = Cs[:, 0] + p[0] * Cs[:, 1] + p[0] * p[1] * Cs[:, 2]
            return (Ms[:, 0] - pred) / np.where(pred > Ms[:, 0], Ms[:, 1], Ms[:, 2])
        r = least_squares(res, [1.0, 1.0], bounds=([0, 0], [5, 5]))
        print(f'    without {str(k):22s}: N2 = {r.x[0]:.2f}, r_s = {r.x[1]:.2f}, chi2 = {np.sum(r.fun**2):.1f}')


if __name__ == '__main__':
    main()
