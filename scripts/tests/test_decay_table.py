"""Tests of scripts/tools/decay_table.py (decay-file parser and generator classification).

Run from the repository root:  python3 scripts/tests/test_decay_table.py

The tests need only the files in the repository: the decay file, evt.pdl and the six decay-mode
CSVs in Notebooks/Plotting/csv_output (run1 e, MC_e_comb).
"""
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'tools'))
import decay_table as dt  # noqa: E402

CSV_DIR = os.path.join(ROOT, 'Notebooks', 'Plotting', 'csv_output')
CSVS = {'comb_charged_Dside': 521, 'comb_charged_ellside': 521, 'comb_neutral_Dside': 511,
        'comb_neutral_ellside': 511, 'hadB_charged': 521, 'hadB_neutral': 511}
RAW = [f'raw_{i}' for i in range(7)]

PDL = dt.load_pdl()
DEC = dt.parse_dec(pdl=PDL)
TABLE = dt.build_mode_table(DEC, PDL)


def pad(codes):
    return list(codes) + [-1] * (7 - len(codes))


def classify(lists, b_abs):
    return dt.classify_decays(np.array([pad(c) for c in lists]), b_abs, TABLE)


# ------------------------------------------------------------------------------------------
def test_pdl():
    n = PDL.name_to_pdg
    # names the `particle` package does not know or gets wrong
    assert n["D'_1+"] == 20413 and n['Xu0'] == 43 and n['Xu+'] == 44 and n['Xsd'] == 30343
    assert n["f'_0"] == 30221 and n['D(2S)+'] == 100411
    assert PDL.charge3[44] == 3 and PDL.charge3[-44] == -3 and PDL.charge3[43] == 0
    assert PDL.charge3[20413] == 3 and PDL.charge3[311] == 0
    assert PDL.conjugate(411) == -411 and PDL.conjugate(111) == 111 and PDL.conjugate(310) == 310
    assert {43, 44, -44, 30343, -30343} <= PDL.inclusive


def test_parse_blocks():
    counts = DEC.groupby('mother', sort=False).size().to_dict()
    assert counts == {'B0': 1381, 'anti-B0': 1381, 'B+': 1181, 'B-': 1181}, counts
    for m, s in DEC.attrs['bf_sum'].items():
        assert abs(s - 1) < 1e-7, (m, s)
    py = DEC[DEC['is_pythia']]
    assert py.groupby('mother', sort=False).size().to_dict() == {'B0': 20, 'anti-B0': 20, 'B+': 16, 'B-': 16}
    assert (py['model'] == 'PYTHIA').all() and (DEC.loc[~DEC['is_pythia'], 'model'] != 'PYTHIA').all()
    assert abs(TABLE.pythia_bf[511] - 0.48318311) < 1e-8 and abs(TABLE.pythia_bf[521] - 0.43336866) < 1e-8
    assert abs(TABLE.pythia_bf[511] + TABLE.exclusive_bf[511] - 1) < 1e-7
    zero = DEC[(DEC['bf'] == 0)].groupby('mother', sort=False).size().to_dict()
    assert zero == {'B0': 62, 'anti-B0': 62, 'B+': 34, 'B-': 34}, zero
    assert TABLE.max_daughters == 6
    assert not DEC['photos'].any() and not DEC['from_cdecay'].any()


def test_parse_lines():
    by_line = DEC.set_index('line')
    r = by_line.loc[366]                        # 0.0511  D*-  e+  nu_e  BGL ...
    assert r['mother'] == 'B0' and r['bf'] == 0.0511 and r['names'] == ('D*-', 'e+', 'nu_e')
    assert r['daughters'] == (-413, -11, 12) and r['model'] == 'BGL' and r['key'] == (-11, 12, -413)
    assert 'arxiv' in r['comment']
    # line with a D'_1 daughter: the name must not be taken for the model
    r = by_line.loc[370]
    assert r['names'][0] == "D'_1-" and r['daughters'][0] == -20413 and r['model'] == 'LLSW'
    # a ';' inside the comment, and a double ';;'
    assert by_line.loc[5544, 'model'] == 'LLSW' and by_line.loc[5544, 'n_daughters'] == 3
    assert by_line.loc[5544, 'model_args'] == '0.68 -0.2 0.3' and by_line.loc[5544, 'comment'] == ''
    # '#$' directive on the line above
    assert by_line.loc[428, 'directive'] == 'forbid_pythia()' and by_line.loc[428, 'bf'] == 0
    assert by_line.loc[429, 'directive'] == ''
    # PYTHIA line
    r = by_line.loc[2141]
    assert r['is_pythia'] and r['model_args'] == '23' and r['daughters'] == (2, -1, -4, 1)


def test_aliases_and_block_asymmetry():
    # J/psi K*0: a zero-BF veto line plus three alias lines (K*S, K*L, K*0T) form one mode
    m = TABLE.get(511, dt.mode_key([443, 313]))
    assert m['lines'] == (433, 434, 435, 436) and abs(m['bf_mc'] - 1.270001e-3) < 1e-12
    assert not m['zero_bf'] and m['n_charmonium'] == 1 and m['n_charm'] == 0
    # the anti-B0 block keeps K*S and K*L as K*0 (+313): the blocks are not mirrors
    assert abs(TABLE.get(-511, dt.mode_key([443, 313]))['bf_mc'] - 4.23334e-4) < 1e-12
    assert abs(TABLE.get(-511, dt.mode_key([443, -313]))['bf_mc'] - 8.46667e-4) < 1e-12
    assert TABLE.get(511, dt.mode_key([443, -313])) is None
    # J/psi K0 exists only as a zero-BF veto; the physical modes are K_S0 and K_L0
    assert TABLE.get(511, dt.mode_key([443, 311]))['zero_bf']
    assert TABLE.get(511, dt.mode_key([443, 310]))['bf_mc'] > 0
    assert int(TABLE.modes['zero_bf'].sum()) == 178


def test_other_files():
    full = dt.parse_dec(mothers=None, pdl=PDL)
    assert full['mother'].nunique() == 363 and full['from_cdecay'].sum() > 0
    # explicit block: D+ -> K- pi+ pi+
    dp = full[full['mother'] == 'D+']
    assert abs(dp[dp['names'] == ('K-', 'pi+', 'pi+')].iloc[0]['bf'] - 0.0938) < 1e-9
    # CDecay: the D*- block is the conjugate of the D*+ block (lines 8113-8117)
    sp = full[full['mother'] == 'D*+']
    sm = full[full['mother'] == 'D*-']
    assert not sp['from_cdecay'].any() and sm['from_cdecay'].all() and len(sp) == len(sm) == 3
    assert list(sp['names']) == [('D0', 'pi+'), ('D+', 'pi0'), ('D+', 'gamma')]
    assert list(sm['names']) == [('anti-D0', 'pi-'), ('D-', 'pi0'), ('D-', 'gamma')]
    assert list(sm['bf']) == list(sp['bf']) == [0.677, 0.307, 0.016]
    assert sm['mother_pdg'].eq(-413).all() and list(sm['line']) == list(sp['line'])
    # signal decay files: aliases come from the master file; PHOTOS keyword before the model
    dec_dir = os.path.join(ROOT, 'Samples', 'dec')
    n_files = 0
    for f in sorted(os.listdir(dec_dir)):
        if f.endswith('.dec'):
            d = dt.parse_dec(os.path.join(dec_dir, f), mothers=None, pdl=PDL, alias_from=dt.DEFAULT_DEC)
            assert len(d) > 0
            n_files += 1
    assert n_files == 10
    d = dt.parse_dec(os.path.join(dec_dir, '1293710005_mod.dec'), mothers=['B+sig'], pdl=PDL,
                     alias_from=dt.DEFAULT_DEC)
    assert d['photos'].any() and d['mother_pdg'].eq(521).all() and 'LLSW' in set(d['model'])


def test_unknown_tokens_stop_the_parser():
    def parse(text, **kw):
        with tempfile.NamedTemporaryFile('w', suffix='.dec', delete=False) as f:
            f.write(text)
        try:
            return dt.parse_dec(f.name, mothers=['B0'], pdl=PDL, **kw)
        finally:
            os.unlink(f.name)
    ok = parse('Decay B0\n0.6 D- pi+ PHSP;\n0.4 D*- pi+ SVS; # c\nEnddecay\nEnd\n')
    assert list(ok['bf']) == [0.6, 0.4]
    for bad in ('Decay B0\n1.0 D- Xnew+ PHSP;\nEnddecay\n',        # unknown particle
                'Decay B0\n1.0 D- pi+ NEWMODEL;\nEnddecay\n'):      # unknown model
        try:
            parse(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('unknown token accepted')
    assert len(parse('Decay B0\n1.0 D- pi+ NEWMODEL 1 2;\nEnddecay\n', extra_models=['NEWMODEL'])) == 1


def test_quark_content():
    n = PDL.name_to_pdg
    incl = PDL.inclusive
    charm = ['D+', 'anti-D0', 'D*-', 'D_s+', 'D_s0*+', 'D_1+', "D'_1+", 'D_2*0', 'D(2S)+',
             'Lambda_c+', 'anti-Sigma_c0', 'Xi_c0', 'Omega_c0', 'B_c+']
    onium = ['J/psi', 'psi(2S)', 'eta_c', 'chi_c1', 'h_c', 'psi(3770)', 'X_1(3872)', 'Z(4430)+']
    neither = ['pi+', 'K0', 'K_S0', 'K_L0', 'K*0', 'rho0', 'eta', 'p+', 'Delta-', 'B0', 'Upsilon',
               'e-', 'nu_e', 'gamma', 'Xu0', 'Xu+', 'Xsd', 'anti-Xsu', 'cd_0', 'anti-cs_1', 'c', 'u']
    for name in charm:
        assert dt.is_open_charm(np.array([n[name]]), incl)[0], name
        assert not dt.is_charmonium(np.array([n[name]]), incl)[0], name
    for name in onium:
        assert dt.is_charmonium(np.array([n[name]]), incl)[0], name
        assert not dt.is_open_charm(np.array([n[name]]), incl)[0], name
    for name in neither:
        assert not dt.is_open_charm(np.array([n[name]]), incl)[0], name
        assert not dt.is_charmonium(np.array([n[name]]), incl)[0], name
    for name in ['p+', 'anti-Delta-', 'Lambda0', 'Lambda_c+', 'Xi_c0']:
        assert dt.is_baryon(np.array([n[name]]), incl)[0], name
    for name in ['pi+', 'D+', 'K_S0', 'cd_0', 'Xsd', 'e-', 'gamma']:
        assert not dt.is_baryon(np.array([n[name]]), incl)[0], name
    # every particle of the list: the digit rule agrees with the name
    for name, code in n.items():
        if abs(code) <= 100 or code in incl or -code in incl or abs(code) >= 1_000_000_000:
            continue
        base = name.replace('anti-', '')
        if (abs(code) // 10) % 10 == 0:          # diquarks
            continue
        by_name = (base.startswith('D') and not base.startswith('Delta')) or '_c' in base and base[0] == 'B' \
            or any(x in base for x in ('Lambda_c', 'Sigma_c', 'Xi_c', "Xi'_c", 'Omega_c', 'Xi_cc', 'Omega_cc'))
        assert bool(dt.is_open_charm(np.array([code]), incl)[0]) == by_name, (name, code)


def test_canonical_order_matches_csv():
    for name in CSVS:
        c = pd.read_csv(os.path.join(CSV_DIR, name + '.csv'))
        canon = dt.canonical_order(c[RAW].to_numpy())
        assert (canon == c[[f'daughter_{i}' for i in range(7)]].to_numpy()).all(), name
    assert dt.mode_key([413, -1, 22, -413, 311, -311]) == (22, 311, -311, 413, -413)


def test_classify_synthetic():
    # exclusive, in the order of the line; flavor from the only block that lists it
    r = classify([[-413, -11, 12], [413, 11, -12]], 511)
    assert (r['gen_class'] == dt.GEN_EXCLUSIVE).all() and list(r['b_flavor']) == [511, -511]
    assert r['is_sl'].all() and (r['sl_lepton'] == 11).all() and (r['n_charm'] == 1).all()
    assert (r['dec_line'] == [366, 2176]).all() and abs(r['bf_mc'][0] - 0.0511) < 1e-12
    assert (r['n_body'] == 3).all() and (r['n_fsr'] == 0).all() and (r['list_state'] == dt.LIST_COMPLETE).all()

    # photons from PHOTOS are stripped, any number of them
    r = classify([[-413, -11, 12, 22], [-413, -11, 12, 22, 22, 22]], 511)
    assert (r['gen_class'] == dt.GEN_EXCLUSIVE).all() and list(r['n_fsr']) == [1, 3] and (r['n_body'] == 3).all()

    # exclusive decay + photons filling the 7 slots: still exclusive and complete
    r = classify([[-413, -11, 12, 22, 22, 22, 22]], 511)
    assert r['gen_class'][0] == dt.GEN_EXCLUSIVE and r['list_state'][0] == dt.LIST_COMPLETE and r['n_fsr'][0] == 4

    # same set, other order: not EvtGen
    r = classify([[-11, -413, 12]], 511)
    assert r['gen_class'][0] == dt.GEN_PYTHIA and r['set_match'][0] and not r['order_match'][0]

    # a genuine photon is kept (B0 -> K*0 gamma, line 871); extra photons are FSR
    r = classify([[313, 22], [313, 22, 22]], 511)
    assert (r['gen_class'] == dt.GEN_EXCLUSIVE).all() and list(r['n_fsr']) == [0, 1] and (r['n_body'] == 2).all()
    assert (r['dec_line'] == 871).all()
    # eta K0 gamma (line 996) must not be read as eta K0 (zero BF) + FSR
    r = classify([[221, 311, 22]], 511)
    assert r['gen_class'][0] == dt.GEN_EXCLUSIVE and r['dec_line'][0] == 996 and not r['zero_bf'][0]

    # not listed: PYTHIA. Charm counts from the daughters.
    r = classify([[-411, 213, 223], [-411, 411, 311, 111], [-411, 2212, -2112], [443, 321, -211, 111]], 511)
    assert (r['gen_class'] == dt.GEN_PYTHIA).all() and r['bf_mc'].isna().all() and (r['dec_line'] == -1).all()
    assert list(r['n_charm']) == [1, 2, 1, 0] and list(r['n_charmonium']) == [0, 0, 0, 1]
    assert list(r['n_baryon']) == [0, 0, 2, 0] and not r['is_sl'].any()

    # final state listed for both flavors in the same order (lines 797 and 2607): flavor ambiguous
    r = classify([[-413, 311, 211]], 511)
    assert r['gen_class'][0] == dt.GEN_EXCLUSIVE and r['flavor_ambiguous'][0] and r['b_flavor'][0] == 0
    # listed for both flavors in different orders (D*+ D*-): the order gives the flavor
    r = classify([[413, -413], [-413, 413]], 511)
    assert (r['gen_class'] == dt.GEN_EXCLUSIVE).all() and not r['flavor_ambiguous'].any()
    assert sorted(r['b_flavor']) == [-511, 511]

    # charged B: flavor from the charge
    r = classify([[-423, -11, 12], [423, 11, -12]], 521)
    assert (r['gen_class'] == dt.GEN_EXCLUSIVE).all() and list(r['b_flavor']) == [521, -521]
    # the B- line is not accepted for a B+ (charge +1) and vice versa
    assert TABLE.get(521, dt.mode_key([423, 11, -12])) is None

    # inclusive pseudo-particles: exclusive lines, flagged, not counted as charm
    r = classify([[-44, -11, 12]], 511)
    assert r['gen_class'][0] == dt.GEN_EXCLUSIVE and r['is_inclusive'][0] and r['n_charm'][0] == 0 and r['is_sl'][0]
    r = classify([[-30343, 22]], 511)
    assert r['gen_class'][0] == dt.GEN_EXCLUSIVE and r['n_fsr'][0] == 0 and r['n_charm'][0] == 0

    # 7 non-photon daughters: PYTHIA; charge consistent -> full, inconsistent -> truncated
    r = classify([[-411, 211, -211, 211, -211, 211, 111], [-411, 211, -211, 211, 211, 211, 111]], 511)
    assert (r['gen_class'] == dt.GEN_PYTHIA).all()
    assert list(r['list_state']) == [dt.LIST_FULL, dt.LIST_TRUNCATED] and (r['n_body'] == 7).all()

    # input checks
    for bad in ([[-1, 211, -1, -1, -1, -1, -1]], [[np.nan] * 7]):
        try:
            dt.classify_decays(np.array(bad), 511, TABLE)
        except ValueError:
            pass
        else:
            raise AssertionError('bad input accepted')
    assert len(dt.classify_decays(np.empty((0, 7)), 511, TABLE)) == 0


def test_classify_csv_tables():
    """Reference numbers of the run1 e comb CR tables (scouting, B.12 of the working document)."""
    ref_excl = {'comb_charged_Dside': 17842, 'comb_charged_ellside': 24149, 'comb_neutral_Dside': 56756,
                'comb_neutral_ellside': 81145, 'hadB_charged': 6748, 'hadB_neutral': 19272}
    ref_wrong_order = {'comb_neutral_ellside': 8}
    for name, b in CSVS.items():
        c = pd.read_csv(os.path.join(CSV_DIR, name + '.csv'))
        w = c['n_candidates']
        r = dt.classify_decays(c[RAW].to_numpy(), b, TABLE)
        excl = r['gen_class'] == dt.GEN_EXCLUSIVE
        assert int(w[excl].sum()) == ref_excl[name], (name, int(w[excl].sum()))
        assert int(w[r['set_match'] & ~r['order_match']].sum()) == ref_wrong_order.get(name, 0), name
        assert not (excl & r['zero_bf']).any(), name                    # veto lines are never populated
        assert not (~excl & r['is_sl']).any(), name                     # PYTHIA makes no semileptonic decay
        assert (r.loc[~excl, 'n_body'] >= 3).all(), name                # nor 2-body decays
        assert (r.loc[excl, 'n_body'] <= 6).all(), name
        assert r.loc[~excl, 'n_fsr'].ge(0).all() and (r['n_body'] + r['n_fsr'] == c['n_daughters']).all()
        # charged B: flavor known unless the list is truncated
        if b == 521:
            assert ((r['b_flavor'] == 0) == (r['list_state'] == dt.LIST_TRUNCATED)).all(), name
        # 7 non-photon daughters can only be PYTHIA
        assert (r.loc[r['n_body'] == 7, 'gen_class'] == dt.GEN_PYTHIA).all(), name
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            rep = dt.validation_report(r, w, name, verbose=False)
        assert rep['ok'] and abs(rep['exclusive'] + rep['pythia'] - 1) < 1e-12


def test_classify_b_decays_ntuple():
    """Ntuple-level wrapper: the CSV rows expanded into '<side>_<b>_<i>_daughterPDG' columns."""
    cn = pd.read_csv(os.path.join(CSV_DIR, 'hadB_neutral.csv')).head(300)
    cc = pd.read_csv(os.path.join(CSV_DIR, 'hadB_charged.csv')).head(200)
    empty = np.full((1, 7), -1)
    d511 = np.vstack([cn[RAW].to_numpy(), np.full((len(cc), 7), -1), empty])
    d521 = np.vstack([np.full((len(cn), 7), -1), cc[RAW].to_numpy(), empty])
    df = pd.DataFrame(np.hstack([d511, d521]).astype('float32'),
                      columns=[f'D_{b}_{i}_daughterPDG' for b in (511, 521) for i in range(7)],
                      index=np.arange(len(d511)) + 1000)
    out = dt.classify_b_decays(df, 'D', TABLE)
    assert out.index.equals(df.index) and len(out) == len(df)
    assert (out['D_B_abs'].to_numpy() == [511] * len(cn) + [521] * len(cc) + [0]).all()
    rn = dt.classify_decays(cn[RAW].to_numpy(), 511, TABLE)
    rc = dt.classify_decays(cc[RAW].to_numpy(), 521, TABLE)
    for col in rn.columns:
        got = out['D_B_' + col].to_numpy()
        exp = np.concatenate([rn[col].to_numpy(), rc[col].to_numpy()])
        assert np.array_equal(got[:-1].astype(float), exp.astype(float), equal_nan=True), col
    assert out['D_B_gen_class'].iloc[-1] == 0 and out['D_B_dec_line'].iloc[-1] == -1


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')


if __name__ == '__main__':
    main()
