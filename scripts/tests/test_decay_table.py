"""Tests of scripts/tools/decay_table.py (decay-file parser and generator classification).

Run from the repository root:  python3 scripts/tests/test_decay_table.py

The tests need only the files in the repository: the decay file, evt.pdl and the decay-mode
tables in Notebooks/Plotting/csv_output. The reference sample is bbbar_modes_run1_e_combCR.csv
(run1 e, MC_e_comb); the other bbbar_modes_*.csv files are checked with validation_report.
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
REFERENCE = os.path.join(CSV_DIR, 'bbbar_modes_run1_e_combCR.csv')
# table name -> (component, side, B type) in the per-sample files
TABLE_KEYS = {'comb_charged_Dside': ('comb', 'D', 521), 'comb_charged_ellside': ('comb', 'ell', 521),
              'comb_neutral_Dside': ('comb', 'D', 511), 'comb_neutral_ellside': ('comb', 'ell', 511),
              'hadB_charged': ('hadB', 'D', 521), 'hadB_neutral': ('hadB', 'D', 511)}
CSVS = {'comb_charged_Dside': 521, 'comb_charged_ellside': 521, 'comb_neutral_Dside': 511,
        'comb_neutral_ellside': 511, 'hadB_charged': 521, 'hadB_neutral': 511}
RAW = [f'raw_{i}' for i in range(7)]

PDL = dt.load_pdl()
DEC = dt.parse_dec(pdl=PDL)
TABLE = dt.build_mode_table(DEC, PDL)
_REF_TABLES = dt.load_mode_tables(REFERENCE)


def ref_table(name):
    """One of the six tables of the reference sample, by its old file name."""
    return _REF_TABLES[TABLE_KEYS[name]]


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
    # the PHOTOS keyword before the model (EvtGen syntax; not used in the B blocks of the master)
    with tempfile.NamedTemporaryFile('w', suffix='.dec', delete=False) as f:
        f.write('Decay B+\n1.0 anti-D0 e+ nu_e PHOTOS ISGW2;\nEnddecay\nEnd\n')
    try:
        d = dt.parse_dec(f.name, mothers=['B+'], pdl=PDL)
    finally:
        os.unlink(f.name)
    assert d['photos'][0] and d['model'][0] == 'ISGW2' and d['daughters'][0] == (-421, -11, 12)


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
    # generator-specific and exotic codes are not baryons
    for code in (9052513, 9152513, 88511, 99513, 99523):
        assert code in PDL.pdg_to_name and not dt.is_baryon(np.array([code]), incl)[0], code
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
        c = ref_table(name)
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
    assert not r['set_match_all'][0] and np.isnan(r['bf_mc'][0]) and r['dec_line'][0] == -1
    # a photon that is not last is not stripped
    r = classify([[22, 313], [-413, 22, -11, 12]], 511)
    assert (r['gen_class'] == dt.GEN_PYTHIA).all() and list(r['n_fsr']) == [0, 0]
    # leptons without a neutrino, or a neutrino pair, are not semileptonic; tau modes are
    r = classify([[321, -11, 11], [311, 12, -12], [-413, -15, 16], [-15, 16]], 511)
    assert list(r['is_sl']) == [False, False, True, True] and list(r['sl_lepton']) == [0, 0, 15, 15]

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
    assert abs(r['bf_mc'][0] - 3.0e-4) < 1e-12 and r['dec_line'][0] == 797 and not r['bf_blocks_differ'][0]
    # J/psi K*0 with a K*0: both blocks, different BFs -> the mean, flagged
    r = classify([[443, 313]], 511)
    assert r['flavor_ambiguous'][0] and r['bf_blocks_differ'][0] and not r['zero_bf'][0]
    assert abs(r['bf_mc'][0] - 0.5 * (1.270001e-3 + 4.23334e-4)) < 1e-12
    # with an anti-K*0 it is listed for the anti-B0 only
    r = classify([[443, -313]], 511)
    assert r['b_flavor'][0] == -511 and abs(r['bf_mc'][0] - 8.46667e-4) < 1e-12
    # listed for both flavors in different orders (D*+ D*-): the order gives the flavor
    r = classify([[413, -413], [-413, 413]], 511)
    assert (r['gen_class'] == dt.GEN_EXCLUSIVE).all() and not r['flavor_ambiguous'].any()
    assert sorted(r['b_flavor']) == [-511, 511]

    # charged B: flavor from the charge
    r = classify([[-423, -11, 12], [423, 11, -12]], 521)
    assert (r['gen_class'] == dt.GEN_EXCLUSIVE).all() and list(r['b_flavor']) == [521, -521]
    # the B- line is not accepted for a B+ (charge +1) and vice versa
    assert TABLE.get(521, dt.mode_key([423, 11, -12])) is None
    r = classify([[423, 11, -12, 22, 22, 22, 22]], 521)     # exclusive + photons: charge known
    assert r['b_flavor'][0] == -521 and r['list_state'][0] == dt.LIST_COMPLETE
    # 7 daughters with the wrong total charge: flavor unknown, list truncated
    r = classify([[-421, 211, -211, 211, 211, 211, 111]], 521)
    assert r['b_flavor'][0] == 0 and r['list_state'][0] == dt.LIST_TRUNCATED

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
        c = ref_table(name)
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


def test_all_samples():
    """Every per-sample file: the mechanism holds (no warning), and weights are finite."""
    import glob
    files = sorted(glob.glob(os.path.join(CSV_DIR, 'bbbar_modes_*.csv')))
    assert REFERENCE in files
    for f in files:
        tables = dt.load_mode_tables(f)
        assert set(tables) <= set(TABLE_KEYS.values()) and len(tables) in (4, 6), f
        for key, t in tables.items():
            assert np.isfinite(t['sum_w']).all() and (t['n_candidates'] > 0).all(), (f, key)
            r = dt.classify_decays(t[RAW].to_numpy(), key[2], TABLE)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                assert dt.validation_report(r, t['sum_w'], f'{f} {key}', verbose=False)['ok']


def test_validation_report_flags():
    def report(lists, b_abs=511, **kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rep = dt.validation_report(classify(lists, b_abs), verbose=False, **kw)
        return rep, len(caught)
    good = [[-413, -11, 12]] * 50 + [[-411, 213, 223]] * 50
    rep, n = report(good)
    assert rep['ok'] and n == 0 and rep['exclusive'] == 0.5
    rep, n = report(good + [[443, 311]])                  # a decay on a zero-BF mode
    assert not rep['ok'] and n == 1 and rep['zero_bf'] > 0
    rep, n = report(good + [[-411, -11, 12, 111, 111]])   # an unlisted semileptonic decay
    assert not rep['ok'] and rep['pythia_sl'] > 0
    rep, n = report(good + [[-411, 9000111]])             # an unlisted 2-body decay
    assert not rep['ok'] and rep['pythia_few_body'] > 0
    rep, n = report(good + [[-11, -413, 12]])             # listed for one flavor, other order: 1 %
    assert not rep['ok'] and rep['wrong_order'] > 3e-4 and rep['wrong_order_all'] == 0
    rep, n = report(good + [[311, -413, 211]])            # listed for both flavors, other order
    assert not rep['ok'] and rep['wrong_order_all'] > 0
    rep, n = report([[-423, -11, 12]] * 10 + [[12, -11, -423]], 521)   # charged B, other order
    assert not rep['ok'] and rep['wrong_order_all'] > 0
    w = np.r_[np.ones(100), 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert dt.validation_report(classify(good + [[443, 311]], 511), w, verbose=False)['ok']


def test_parser_follows_evtgen_rules():
    def parse(text, mothers, **kw):
        with tempfile.NamedTemporaryFile('w', suffix='.dec', delete=False) as f:
            f.write(text)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                return dt.parse_dec(f.name, mothers=mothers, pdl=PDL, **kw), len(caught)
        finally:
            os.unlink(f.name)
    # a second Decay block replaces the first
    d, n = parse('Decay B0\n1.0 D- pi+ PHSP;\nEnddecay\nDecay B0\n1.0 D*- pi+ SVS;\nEnddecay\n', ['B0'])
    assert n == 1 and list(d['names']) == [('D*-', 'pi+')]
    # CDecay copies the conjugate block as it stands; a later Decay replaces the copy
    d, n = parse('Decay B-\n1.0 D0 pi- PHSP;\nEnddecay\nCDecay B+\nDecay B-\n1.0 D*0 pi- SVS;\nEnddecay\n',
                 ['B+', 'B-'])
    assert list(d['names']) == [('anti-D0', 'pi+'), ('D*0', 'pi-')] and list(d['from_cdecay']) == [True, False]
    d, n = parse('Decay B-\n1.0 D0 pi- PHSP;\nEnddecay\nCDecay B+\nDecay B+\n1.0 anti-D*0 pi+ SVS;\nEnddecay\n',
                 ['B+'])
    assert list(d['names']) == [('anti-D*0', 'pi+')]
    # a second statement on the line is not silently taken as a comment
    d, n = parse('Decay B0\n0.5 D- pi+ PHSP; 0.5 D*- pi+ SVS;\nEnddecay\n', ['B0'])
    assert n == 1 and len(d) == 1
    d, n = parse('Decay B0\n1.0 D- pi+ PHSP;; # c ; d\nEnddecay\n', ['B0'])
    assert n == 0 and d['comment'][0] == '# c ; d'
    for bad in ('Define x 0.3\nDecay B0\nx D- pi+ PHSP;\nEnddecay\n',            # BF is not a number
                'Decay B0\n1.0 PHSP;\nEnddecay\n',                                # no daughters
                'Decay Foo\n1.0 D- pi+ PHSP;\nEnddecay\n',                        # unknown mother
                'CDecay B+\n',                                                     # nothing to conjugate
                'Alias D0x D0\nDecay B-\n1.0 D0x pi- PHSP;\nEnddecay\nCDecay B+\n'):  # alias w/o ChargeConj
        try:
            parse(bad, None)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted: ' + bad)
    # alias mothers sharing a PDG code can't go into one table
    d, n = parse('Alias B0sig B0\nDecay B0\n1.0 D- pi+ PHSP;\nEnddecay\nDecay B0sig\n1.0 D- pi+ PHSP;\nEnddecay\n', None)
    try:
        dt.ModeTable(d, PDL)
    except ValueError:
        pass
    else:
        raise AssertionError('alias mothers merged')
    # is_pythia: quark-level lines only. The tau uses the PYTHIA model with hadron daughters.
    full = dt.parse_dec(mothers=None, pdl=PDL)
    tau = full[full['mother'] == 'tau-']
    assert (tau['model'] == 'PYTHIA').any() and not tau['is_pythia'].any()


def _utilities_functions():
    """decay_mode_table and export_bbbar_decay_mode_tables, taken from the source of utilities.py so
    that the test needs numpy and pandas only (importing utilities needs uproot, lightgbm, ...)."""
    import re
    src = open(os.path.join(ROOT, 'scripts', 'tools', 'utilities.py')).read()
    ns = {'np': np, 'pd': pd}
    for start in ('QUARK_LIKE_PDG = ', 'def decay_mode_table', 'def b_ancestry_check',
                  'def export_bbbar_decay_mode_tables'):
        a = src.index(start)
        exec(src[a:a + re.search(r'\n\n\n', src[a:]).start()], ns)
    return ns


def _fake_ntuple(names, b_abs_list, n_head=400, seed=0):
    """Candidates with '<side>_<b>_<i>_daughterPDG' columns built from rows of the CSVs."""
    cols = [f'{s}_{b}_{i}_daughterPDG' for s in ('D', 'ell') for b in (511, 521) for i in range(7)]
    parts = []
    for (name_d, name_l), b in zip(names, b_abs_list):
        d = ref_table(name_d).head(n_head)
        d = np.repeat(d[RAW].to_numpy(), np.minimum(d['n_candidates'], 5), axis=0)
        ell = d if name_l is None else ref_table(name_l)[RAW].to_numpy()[:len(d)]
        df = pd.DataFrame(-1.0, index=range(len(d)), columns=cols, dtype='float32')
        df[[f'D_{b}_{i}_daughterPDG' for i in range(7)]] = d
        df[[f'ell_{b}_{i}_daughterPDG' for i in range(7)]] = ell
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def test_export_and_long_format():
    import contextlib
    import io
    ns = _utilities_functions()
    export = ns['export_bbbar_decay_mode_tables']
    comb = _fake_ntuple([('comb_charged_Dside', 'comb_charged_ellside'),
                         ('comb_neutral_Dside', 'comb_neutral_ellside')], [521, 511])
    hadB = _fake_ntuple([('hadB_charged', None), ('hadB_neutral', None)], [521, 511])
    comb['w'], hadB['w'] = 0.5, 2.0
    samples = {'bkg_combinatorial': comb, 'bkg_hadronicB_secondaryL': hadB}
    quiet = contextlib.redirect_stdout(io.StringIO())
    with tempfile.TemporaryDirectory() as out_dir:
        with quiet:
            tabs = export(samples, 'unit_test', out_dir=out_dir, selection='cut a,\n "b"', weight_col='w')
        path = os.path.join(out_dir, 'bbbar_modes_unit_test.csv')
        assert sorted(tabs) == sorted(CSVS)
        long = dt.load_mode_tables(path)
        assert sorted(long) == [('comb', 'D', 511), ('comb', 'D', 521), ('comb', 'ell', 511),
                                ('comb', 'ell', 521), ('hadB', 'D', 511), ('hadB', 'D', 521)]
        t = long[('comb', 'D', 511)]
        assert (t['tag'] == 'unit_test').all() and (t['selection'] == 'cut a, "b"').all()
        assert t['n_candidates'].sum() == (comb['D_511_0_daughterPDG'] != -1).sum()
        assert np.allclose(t['sum_w'], 0.5 * t['n_candidates'])
        assert np.allclose(long[('hadB', 'D', 521)]['sum_w'], 2.0 * long[('hadB', 'D', 521)]['n_candidates'])
        # counts against a direct count
        direct = comb.loc[comb['D_511_0_daughterPDG'] != -1, [f'D_511_{i}_daughterPDG' for i in range(7)]]
        direct = direct.astype('int64').value_counts()
        assert len(direct) == len(t) and (direct.reindex(pd.MultiIndex.from_frame(t[RAW])).to_numpy()
                                          == t['n_candidates'].to_numpy()).all()
        # each mode is one block of rows; mode_total and n_orders are consistent
        modes = t[[f'daughter_{i}' for i in range(7)]].astype(str).agg(' '.join, axis=1)
        assert (modes != modes.shift()).sum() == modes.nunique()
        assert (t.groupby(modes)['n_candidates'].transform('sum') == t['mode_total']).all()
        assert t['mode_total'].is_monotonic_decreasing
        # the table does not depend on the order of the candidates
        with quiet:
            tabs2 = export({'bkg_combinatorial': comb.iloc[::-1], 'bkg_hadronicB_secondaryL': hadB.iloc[::-1]},
                           'unit_test2', out_dir=out_dir, weight_col='w')
        for k in tabs:
            assert tabs[k].equals(tabs2[k]), k
        # the long file can be classified directly
        with quiet:
            cls = dt.classify_mode_tables(long, TABLE)
        assert set(cls[('comb', 'ell', 521)]['gen_class']) <= {dt.GEN_EXCLUSIVE, dt.GEN_PYTHIA}
        # no silent overwrite; no hadB is fine; bad inputs are refused
        for kw in (dict(tag='unit_test'), dict(tag=None), dict(tag='a/b'), dict(tag='.x'),
                   dict(tag='ok1', selection=['a'])):
            try:
                with quiet:
                    export(samples, out_dir=out_dir, **kw)
            except (FileExistsError, ValueError):
                pass
            else:
                raise AssertionError(f'accepted {kw}')
        with quiet:
            export(samples, 'unit_test', out_dir=out_dir, overwrite=True)
            only_comb = export({'bkg_combinatorial': comb}, 'wc', out_dir=out_dir)
        assert len(only_comb) == 4 and len(dt.load_mode_tables(os.path.join(out_dir, 'bbbar_modes_wc.csv'))) == 4
        for bad_col, bad_val in (('w', np.nan), ('D_511_2_daughterPDG', np.nan)):
            broken = comb.copy()
            broken.loc[broken.index[3], bad_col] = bad_val
            try:
                with quiet:
                    export({'bkg_combinatorial': broken}, 'broken', out_dir=out_dir, weight_col='w')
            except ValueError:
                pass
            else:
                raise AssertionError(f'NaN in {bad_col} accepted')
        try:
            with quiet:
                export({'bkg_combinatorial': comb.iloc[:0]}, 'empty', out_dir=out_dir)
        except ValueError:
            pass
        else:
            raise AssertionError('empty sample accepted')


def test_classify_b_decays_ntuple():
    """Ntuple-level wrapper: the CSV rows expanded into '<side>_<b>_<i>_daughterPDG' columns."""
    cn = ref_table('hadB_neutral').head(300)
    cc = ref_table('hadB_charged').head(200)
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
    assert out['D_B_gen_class'].dtype.kind == 'i' and out['D_B_is_sl'].dtype == bool
    # a candidate with both ancestor lists filled is not classified
    both = df.copy()
    both.iloc[0, 7:] = both.iloc[len(cn), 7:].to_numpy()
    assert dt.classify_b_decays(both, 'D', TABLE)['D_B_abs'].iloc[0] == 0
    # the report works on the prefixed columns and leaves out the unclassified row
    rep = dt.validation_report(out, verbose=False, prefix='D_B_')
    assert rep['ok'] and abs(rep['unclassified'] - 1 / len(df)) < 1e-12
    assert abs(rep['exclusive'] + rep['pythia'] - 1) < 1e-12
    bad = df.copy()
    bad.iloc[0, 0] = np.nan
    try:
        dt.classify_b_decays(bad, 'D', TABLE)
    except ValueError:
        pass
    else:
        raise AssertionError('NaN accepted')


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')


if __name__ == '__main__':
    main()
