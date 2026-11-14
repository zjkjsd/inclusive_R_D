"""Tests of scripts/tools/pdg_match.py (comparison of the decay file with PDG).

Run from the repository root:  python3 scripts/tests/test_pdg_match.py

The tests need only files of the repository: the decay file, evt.pdl, the cached PDG table
(Samples/dec/pdg_bf_B_2026.json), the exceptions file and the decay-mode tables in
Notebooks/Plotting/csv_output. They do not import the `pdg` package and need no network.
"""
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'tools'))
import decay_table as dt  # noqa: E402
import pdg_match as pm  # noqa: E402

PDL = dt.load_pdl()
PDG = pm.load_pdg_table()
MODES = dt.build_mode_table(pdl=PDL)
DEC_ALL = dt.parse_dec(mothers=None, pdl=PDL)
TABLE, NODES = pm.build_mode_pdg_table(MODES, PDG, dec_all=DEC_ALL, pdl=PDL, return_nodes=True)
LOOKUP = pm.table_lookup(TABLE)


def code(names):
    return [PDL.name_to_pdg[n] for n in names.split()]


def row(mother, names, table=None, lookup=None):
    """Row of the mode table for a final state given by evt.pdl names."""
    r = pm.pdg_info(TABLE if table is None else table, mother, code(names),
                    LOOKUP if table is None else lookup)
    assert r is not None, f'{mother} -> {names} is not in the table'
    return r


def flags(r):
    return set(r['flags'].split(';')) - {''}


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------------------------------------------------------------------------------
# cached PDG table
# ------------------------------------------------------------------------------------------
def test_cache():
    assert 'pdg' not in sys.modules                       # analysis code does not import it
    meta = PDG['meta']
    assert meta['edition'] == '2026' and 'CC BY' in meta['license']
    assert 'Particle Data Group' in meta['citation']
    assert meta['data_release_timestamp'] and meta['package_version']
    nodes = {n['pdgid']: n for n in PDG['nodes']}
    assert len(nodes) == len(PDG['nodes']) == 1273
    assert sum(n['parent'] == 'B0' for n in PDG['nodes']) == 584
    # D- l nu is an indented node, hidden by the API unless the sub-decays are asked for
    n = nodes['S042.37']
    assert n['description'] == 'B0 --> D- lepton+ nu_lepton' and n['level'] == 1
    assert n['value'] == 0.0212 and n['error_positive'] == 0.0006 and not n['is_limit']
    assert max(x['level'] for x in PDG['nodes']) == 3
    assert sum(x['level'] > 0 for x in PDG['nodes']) > 300
    assert nodes['S042.418']['is_limit'] and nodes['S042.418']['confidence_level'] == 90.0
    for k in ('parent', 'description', 'value', 'error_positive', 'error_negative', 'is_limit',
              'confidence_level', 'scale_factor', 'value_type', 'data_type', 'level'):
        assert k in n
    # some nodes have no value; they are kept and never used
    assert nodes['S042.619']['value'] is None


def test_name_table():
    for pdg_name, evt in pm.PDG_NAMES.items():
        assert evt.startswith('@') or evt in PDL.name_to_pdg, (pdg_name, evt)
    n = pm.PDG_NAMES
    assert n['K(1270)0'] == n['K_1(1270)0'] == 'K_10'              # PDG spells K_1(1270) both ways
    assert n['D_s0(2317)+'] == n['D_s0^*(2317)+'] == 'D_s0*+'
    assert n['D_sJ(2457)+'] == 'D_s1+' and n['D_s1(2536)+'] == "D'_s1+"
    assert n['Dbar_0^*(2420)0'] == n['Dbar_0^*(2400)0'] == 'anti-D_0*0'
    assert n["eta^'"] == "eta'"
    for ambiguous in ('f_0()', 'rho()0', 'psi()', 'D_sJ(2700)+', 'D^*(2010)'):
        assert ambiguous not in n


# ------------------------------------------------------------------------------------------
# text parser
# ------------------------------------------------------------------------------------------
def parse(text):
    return pm.parse_description(text, PDL)


def test_parser_plain():
    p = parse('B0 --> D- lepton+ nu_lepton')
    assert p['kind'] == 'plain' and p['lepton'] and p['parent'] == 'B0'
    v = pm.node_variants(p)
    assert [x[0] for x in v] == ['e', 'mu']
    assert v[0][1] == [('D-', 'e+', 'nu_e')] and v[1][1] == [('D-', 'mu+', 'nu_mu')]
    p = parse('B0 --> J/psi(1S) K(1270)0')
    assert p['kind'] == 'plain' and p['main'] == [('J/psi', 'K_10')] and not p['lepton']
    assert parse("B0 --> eta^' K0")['main'] == [("eta'", 'K0')]
    assert parse('B+ --> Dbar0 K+ Kbar0')['main'] == [('anti-D0', 'K+', 'anti-K0')]
    assert parse('B0 --> D- 3pi+ 2pi-')['main'] == [('D-', 'pi+', 'pi+', 'pi+', 'pi-', 'pi-')]
    assert parse('B0 --> K0S K0S K0S')['n_ks'] == 3 and parse('B0 --> J/psi(1S) K0')['n_ks'] == 0
    assert parse('B+ --> K+ lepton+ lepton-')['lepton']


def test_parser_product():
    a = parse('B0 --> D_s0^*(2317)+ D- , D_s0^*()+ --> D_s()+ pi0')
    b = parse('B0 --> D_s0(2317)+ D^*(2010)- , D_s0()+ --> D_s()+ pi0')
    assert a['kind'] == b['kind'] == 'product'
    assert a['main'] == [('D_s0*+', 'D-')] and a['subs'] == [('D_s0*+', [('D_s+', 'pi0')])]
    assert b['main'] == [('D_s0*+', 'D*-')] and b['subs'] == a['subs']
    p = parse('B0 --> D_sJ(2457)+ D()- , D_sJ()+ --> D_s()+ gamma')
    assert p['subs'] == [('D_s1+', [('D_s+', 'gamma')])]
    # second syntax, with and without blanks
    p = parse('B+ --> D_s0(2317)+ Dbar^*(2007)0 x B(D_s0(2317)+ --> D_s()+ pi0)')
    assert p['kind'] == 'product' and p['main'] == [('D_s0*+', 'anti-D*0')]
    p = parse('B+ --> Dbar_2^*(2460)0 pi+xB(Dbar_2^*()0 --> Dbar^*()0 pi+ pi-)')
    assert p['subs'] == [('anti-D_2*0', [('anti-D*0', 'pi+', 'pi-')])]
    # lepton nodes, "Dbar_0^*(2420)0" is the D_0^*(2300)
    p = parse('B+ --> Dbar_0^*(2420)0 lepton+ nu_lepton , Dbar_0^*()0 --> D- pi+')
    assert p['kind'] == 'product' and p['lepton'] and p['subs'] == [('anti-D_0*0', [('D-', 'pi+')])]
    assert pm.node_variants(p)[1][1] == [('anti-D_0*0', 'mu+', 'nu_mu')]
    # the mother of the sub-decay is found from the first part, whatever its spelling
    p = parse('B0 --> J/psi rho(1700)0 , rho()0 --> pi+ pi-')
    assert p['subs'] == [('rho(3S)0', [('pi+', 'pi-')])]
    p = parse('B0 --> psi(3770) K0 , psi --> Dbar0 D0')
    assert p['subs'] == [('psi(3770)', [('anti-D0', 'D0')])]
    p = parse('B0 --> Dbar_2^*(2460)0 omega , Dbar_2()0 --> D^*()- pi+')
    assert p['subs'] == [('anti-D_2*0', [('D*-', 'pi+')])]
    # a sum inside the sub-decay, and two sub-decays
    p = parse('B0 --> D- D_s1(2536)+ , D_s1()+ --> D^*()0 K+ + D^*()+ K0')
    assert p['subs'] == [("D'_s1+", [('D*0', 'K+'), ('D*+', 'K0')])]
    p = parse('B0 --> a_1(1260)+ a_1(1260)- , a_1()+ --> 2pi+ pi- , a_1()- --> 2pi- pi+')
    assert [m for m, _ in p['subs']] == ['a_1+', 'a_1-']
    p = parse('B0 --> eta_c(2S) K0S , eta_c --> p pbar pi+ pi-')
    assert p['kind'] == 'product' and p['n_ks'] == 1


def test_parser_other_kinds():
    p = parse('B0 --> D^*(2010)- D+ K0 + D- D^*(2010)+ K0')
    assert p['kind'] == 'sum' and p['main'] == [('D*-', 'D+', 'K0'), ('D-', 'D*+', 'K0')]
    for text in ('B0 --> (D- pi+ pi+ pi-) nonresonant', 'B0 --> eta_c(1S) K+ pi- (NR)',
                 'B+ --> NR(1--) K+ , NR(1--) --> D^*(2010)-+ D+-',
                 'B+ --> Dbar_1(2420)0 pi+ x B(Dbar_1()0 --> Dbar0 pi+ pi- (nonresonant))'):
        assert parse(text)['kind'] == 'nonres', text
    unkeyed = ['B0 --> D0 X',                                    # inclusive
               'B0 --> lepton+ nu_lepton X_u',
               'B0 --> D+- D^*()-+ (CP-averaged)',                # charge sum
               'B0 --> J/psi(1S) K0 K- pi+ + c.c.',
               'B0 --> ( Dbar + Dbar^* )( D + D^* ) K',           # generic
               'B+ --> [ K- pi+ ](D) K+',                         # D decay
               'B0 --> D- D_sJ(2700)+ , D_sJ()+ --> D0 K+',       # explicit mass, not in the table
               'B0 --> D_sJ(2457)+ D^*(2010) , D_sJ()+ --> D_s()+ gamma',   # no charge on the D*
               'B0 --> J/psi(1S) f_0(500) , f_0() --> pi pi',     # "pi pi"
               'B0 --> chi_c0(3915) , chi_c0() --> J/psi omega',  # one daughter
               'B+ --> D^*()- p pbar pi+ pi-',                    # charge not conserved
               'B0 --> K0 nu nubar', 'B0 --> Lambda psi_DS']
    for text in unkeyed:
        p = parse(text)
        assert p['kind'] == 'unkeyed' and p['reason'], text
    assert 'charge' in parse('B+ --> D^*()- p pbar pi+ pi-')['reason']
    assert 'D_sJ(2700)+' in parse('B0 --> D- D_sJ(2700)+ , D_sJ()+ --> D0 K+')['reason']


def test_parser_on_the_cache():
    nodes = pm.make_nodes(PDG, PDL)
    kinds = {}
    for n in nodes:
        kinds.setdefault(n['kind'], set()).add(n['pdgid'])
    n_kind = {k: len(v) for k, v in kinds.items()}
    assert sum(n_kind.values()) == 1273
    assert n_kind['plain'] > 700 and n_kind['product'] > 150 and n_kind['nonres'] == 37
    assert n_kind['unkeyed'] < 330
    by_uid = {n['uid']: n for n in nodes}
    # lepton nodes exist once per lepton flavor
    assert by_uid['S042.37[e]']['targets'] == [pm.collapse_key(code('D- e+ nu_e'))]
    assert by_uid['S042.37[mu]']['targets'] == [pm.collapse_key(code('D- mu+ nu_mu'))]
    assert 'S042.37' not in by_uid
    # a single K_S0 doubles the value; several give no c
    ks = [n for n in nodes if n['description'] == 'B0 --> K0S K0S K0S']
    assert len(ks) == 1 and ks[0]['multi_ks'] and not ks[0]['usable']
    raw = {n['pdgid']: n for n in PDG['nodes']}
    one = [n for n in nodes if n['kind'] == 'plain' and ' K0S' in n['description']
           and not n['multi_ks'] and n['usable']]
    assert one and all(close(n['value'], 2 * raw[n['pdgid']]['value']) for n in one)
    # asymmetric errors: the mean
    n = by_uid['S042.2']
    assert close(n['err'], 0.5 * (raw['S042.2']['error_positive'] + raw['S042.2']['error_negative']))
    assert raw['S042.2']['error_positive'] != raw['S042.2']['error_negative']


def test_keys():
    assert pm.collapse_key([443, 310]) == pm.collapse_key([443, 130]) == pm.collapse_key([-311, 443])
    assert pm.collapse_key([443, 313]) != pm.collapse_key([443, -313])
    assert pm._loose_key([443, 313]) == pm._loose_key([443, -313])
    assert pm.conjugate_key((-11, 12, -411), PDL) == dt.mode_key([11, -12, 411])
    assert pm.conjugate_key((111, 310, 443), PDL) == (111, 310, 443)


# ------------------------------------------------------------------------------------------
# toy decay file with known answers
# ------------------------------------------------------------------------------------------
TOY_DEC = """
Alias K*0T K*0
Alias anti-K*0T anti-K*0
ChargeConj K*0T anti-K*0T
Decay B0
0.0200 D- e+ nu_e PHSP;
0.0210 D- mu+ nu_mu PHSP;
0.0010 D- tau+ nu_tau PHSP;
0.0006 J/psi K*0 PHSP;
0.0004 J/psi K*0T PHSP;
0.0004 J/psi K+ pi- PHSP;
0.0005 J/psi K_2*0 PHSP;
0.0100 D*- a_1+ PHSP;
0.0020 D*- rho0 pi+ PHSP;
0.0010 D*- pi+ pi+ pi- PHSP;
0.0080 D- rho+ PHSP;
0.0007 D- pi+ pi0 PHSP;
0.0100 D_s0*+ D- PHSP;
0.0020 J/psi K_S0 PHSP;
0.0020 J/psi K_L0 PHSP;
0.0001 K_S0 K_S0 pi0 PHSP;
0.0003 pi+ pi- PHSP;
0.0003 K+ K- PHSP;
0.0002 Xsd gamma PHSP;
0.0000 D- pi+ PHSP;
0.0009 D_1- pi+ PHSP;
0.4000 anti-d u d anti-d PYTHIA 48;
Enddecay
Decay anti-B0
0.0200 D+ e- anti-nu_e PHSP;
0.0006 J/psi anti-K*0 PHSP;
0.0004 J/psi K*0 PHSP;
0.0020 J/psi K_S0 PHSP;
0.0003 K0 pi0 PHSP;
Enddecay
Decay B+
0.0090 anti-D0 pi+ pi0 PHSP;
0.0010 anti-D0 rho+ PHSP;
Enddecay
CDecay B-
Decay K*0
0.6667 K+ pi- PHSP;
0.3333 K0 pi0 PHSP;
Enddecay
CDecay anti-K*0
Decay K*0T
1.0 K+ pi- PHSP;
Enddecay
CDecay anti-K*0T
Decay K_2*0
0.5 K+ pi- PHSP;
0.5 K*0 pi0 PHSP;
Enddecay
Decay a_1+
0.6 rho0 pi+ PHSP;
0.4 pi+ pi+ pi- PHSP;
Enddecay
Decay rho0
2.0 pi+ pi- PHSP;
Enddecay
Decay rho+
1.0 pi+ pi0 PHSP;
Enddecay
Decay D_s0*+
0.8 D_s+ pi0 PHSP;
0.2 D_s*+ gamma PHSP;
Enddecay
Decay D_1-
0.5 D- pi+ pi- PHSP;
0.5 anti-D*0 pi- PHSP;
Enddecay
"""

# (pdgid, text, value, error, is_limit)
TOY_PDG = [
    ('T.1', 'B0 --> D- lepton+ nu_lepton', 0.021, 0.001, False),
    ('T.2', 'B0 --> D- tau+ nu_tau', 0.0011, 0.0002, False),
    ('T.3', 'B0 --> J/psi(1S) K^*(892)0', 0.0012, 0.0001, False),
    ('T.4', 'B0 --> J/psi(1S) K+ pi-', 0.0015, 0.0002, False),
    ('T.5', 'B0 --> D^*(2010)- a_1(1260)+', 0.012, 0.0005, False),
    ('T.6', 'B0 --> D^*(2010)- pi+ rho0', 0.009, 0.0002, False),
    ('T.7', 'B0 --> D^*(2010)- pi+ pi+ pi-', 0.016, 0.0002, False),
    ('T.8', 'B0 --> D- rho+', 0.0076, 0.0005, False),
    ('T.9', 'B0 --> D- pi+ pi0', 0.007, 0.001, False),
    ('T.10', 'B0 --> D_s0^*(2317)+ D- , D_s0^*()+ --> D_s()+ pi0', 0.001, 0.0002, False),
    ('T.11', 'B0 --> D_s0(2317)+ D()- , D_s0()+ --> D_s^*()+ gamma', 0.0004, 0.0001, False),
    ('T.12', 'B0 --> J/psi(1S) K0S', 0.0022, 0.0001, False),
    ('T.13', 'B0 --> K0S K0S pi0', 0.00005, 0.00001, False),
    ('T.14', 'B0 --> pi+ pi-', 0.0005, 0.0, True),
    ('T.15', 'B0 --> K+ K-', 0.0001, 0.0, True),
    ('T.16', 'B0 --> D- pi+', 0.0025, 0.0001, False),
    ('T.17', 'B0 --> D_1(2420)- pi+ , D_1()- --> D- pi+ pi-', 0.00036, 0.00009, False),
    ('T.18', 'B0 --> D- pi+ pi+ pi-', 0.0006, 0.0001, False),
    ('T.20', 'B+ --> Dbar0 rho+', 0.008, 0.0008, False),
    ('T.21', 'B+ --> Dbar0 pi+ pi0', 0.010, 0.001, False),
    ('T.22', 'B0 --> D0 X', 0.08, 0.01, False),
]


def toy(exceptions=(), pdg=TOY_PDG):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'toy.dec')
        with open(path, 'w') as f:
            f.write(TOY_DEC)
        dec_all = dt.parse_dec(path, mothers=None, pdl=PDL)
        modes = dt.build_mode_table(dt.parse_dec(path, pdl=PDL), PDL)
    table = dict(meta={}, nodes=[dict(pdgid=i, parent=t.split()[0], description=t, value=v,
                                      error_positive=e, error_negative=e, is_limit=lim, level=0)
                                 for i, t, v, e, lim in pdg])
    out, nodes = pm.build_mode_pdg_table(modes, table, exceptions=list(exceptions), dec_all=dec_all,
                                         pdl=PDL, return_nodes=True)
    return out, nodes, pm.table_lookup(out)


def test_toy_graph():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'toy.dec')
        with open(path, 'w') as f:
            f.write(TOY_DEC)
        g = pm.DecayGraph(dt.parse_dec(path, mothers=None, pdl=PDL), PDL)
    assert {'K*0', 'K*0T', 'anti-K*0', 'K_2*0', 'a_1+', 'rho0', 'rho+', 'D_1-'} <= g.broad
    assert 'D_s0*+' not in g.broad                         # narrow: never folded by default
    assert close(sum(p for p, _ in g.blocks['rho0']), 1.0)  # block sums are normalised
    fold = g.default_fold()
    d = g.expand(('J/psi', 'K*0'), fold)
    assert close(d[pm.collapse_key(code('J/psi K+ pi-'))], 0.6667)
    assert close(d[pm.collapse_key(code('J/psi K0 pi0'))], 0.3333)
    assert close(g.expand(('J/psi', 'K*0T'), fold)[pm.collapse_key(code('J/psi K+ pi-'))], 1.0)
    # nested: K_2*0 -> K*0 pi0 -> K+ pi- pi0
    d = g.expand(('J/psi', 'K_2*0'), fold)
    assert close(d[pm.collapse_key(code('J/psi K+ pi- pi0'))], 0.5 * 0.6667)
    # a particle named in the node is not folded
    d = g.expand(('D*-', 'a_1+'), g.default_fold(code('D*- pi+ rho0')))
    assert close(d[pm.collapse_key(code('D*- rho0 pi+'))], 0.6)
    assert close(d[pm.collapse_key(code('D*- pi+ pi+ pi-'))], 0.4)
    assert close(g.expand(('D*-', 'a_1+'), fold)[pm.collapse_key(code('D*- pi+ pi+ pi-'))], 1.0)
    assert g.expand(('D*-', 'a_1+'), g.default_fold(code('a_1+'))) == {pm.collapse_key(code('D*- a_1+')): 1.0}
    # the depth is bounded
    shallow = pm.DecayGraph.__new__(pm.DecayGraph)
    shallow.__dict__.update(g.__dict__)
    shallow.max_depth, shallow._memo = 1, {}
    assert pm.collapse_key(code('J/psi K*0 pi0')) in shallow.expand(('J/psi', 'K_2*0'), fold)


def test_toy_exact_and_lepton():
    t, nodes, lk = toy()
    e, mu = row(511, 'D- e+ nu_e', t, lk), row(511, 'D- mu+ nu_mu', t, lk)
    assert e['how'] == mu['how'] == 'exact' and e['pdgids'] == mu['pdgids'] == 'T.1'
    assert close(e['c'], 0.021 / 0.020) and close(mu['c'], 1.0)      # each flavor with the full value
    assert close(e['sigma_c'], 0.001 / 0.020) and close(e['pull'], 1.0)
    assert e['pdg_class'] == 'measured' and e['is_sl'] and not e['tau'] and flags(e) == set()
    tau = row(511, 'D- tau+ nu_tau', t, lk)
    assert tau['tau'] and 'tau' in flags(tau) and close(tau['c'], 1.1)
    # alias lines of one final state are summed before the comparison
    r = row(511, 'J/psi K*0', t, lk)
    assert close(r['bf_mc'], 0.0010) and r['how'] == 'exact' and close(r['c'], 1.2)
    assert close(r['sigma_c'], 0.1)
    # zero BF in the decay file, measured in PDG: no c
    r = row(511, 'D- pi+', t, lk)
    assert r['pdg_class'] == 'measured' and r['how'] == '' and r['c'] == 1.0 and 'zero_bf' in flags(r)
    assert math.isnan(r['pull'])
    # inclusive pseudo-particle, and a mode without any node
    r = row(511, 'Xsd gamma', t, lk)
    assert r['pdg_class'] == 'inclusive' and r['c'] == 1.0 and r['how'] == ''
    assert set(t['pdg_class']) <= {'measured', 'limit', 'none', 'inclusive'}
    assert len(t) == len(t.drop_duplicates(['mother_pdg', 'key'])) and 'T.22' in set(nodes['pdgid'])
    assert nodes.set_index('uid').loc['T.22', 'kind'] == 'unkeyed'


def test_toy_node_and_remainder():
    t, nodes, lk = toy()
    n = nodes.set_index('uid')
    # J/psi K+ pi-: K* (alias line folded with its own table) + direct + K_2*
    feed_kst = 0.0006 * 0.6667 + 0.0004 * 1.0
    bf_node = feed_kst + 0.0004 + 0.0005 * 0.5
    assert close(n.loc['T.4', 'bf_mc'], bf_node) and n.loc['T.4', 'n_feeders'] == 3
    assert close(n.loc['T.4', 'ratio'], 0.0015 / bf_node) and n.loc['T.4', 'role'] == 'node'
    c = (0.0015 - 1.2 * feed_kst) / (0.0004 + 0.00025)
    s = math.sqrt(0.0002 ** 2 + (0.1 * feed_kst) ** 2) / (0.0004 + 0.00025)
    for names, p in (('J/psi K+ pi-', 1.0), ('J/psi K_2*0', 0.5)):
        r = row(511, names, t, lk)
        assert r['how'] == 'node' and r['pdgids'] == 'T.4' and close(r['c'], c), names
        assert close(r['sigma_c'], s) and close(r['p_node'], p) and close(r['bf_pdg'], 0.0015)
        assert 'remainder_unstable' not in flags(r)
    assert close(row(511, 'J/psi K*0', t, lk)['c'], 1.2)            # the exact line is untouched
    # nested nodes, from the fewest bodies up
    assert close(row(511, 'D*- a_1+', t, lk)['c'], 1.2)
    r3 = row(511, 'D*- rho0 pi+', t, lk)
    c3 = (0.009 - 1.2 * 0.006) / 0.002
    s3 = math.sqrt(0.0002 ** 2 + (0.05 * 0.006) ** 2) / 0.002
    assert r3['how'] == 'node' and r3['pdgids'] == 'T.6' and close(r3['c'], c3) and close(r3['sigma_c'], s3)
    r4 = row(511, 'D*- pi+ pi+ pi-', t, lk)
    c4 = (0.016 - 1.2 * 0.010 - c3 * 0.002) / 0.001
    s4 = math.sqrt(0.0002 ** 2 + (0.05 * 0.010) ** 2 + (s3 * 0.002) ** 2) / 0.001
    assert r4['pdgids'] == 'T.7' and close(r4['c'], c4) and close(r4['sigma_c'], s4)
    assert close(r4['pull'], (c4 - 1) / s4)
    assert close(n.loc['T.7', 'bf_mc'], 0.013) and n.loc['T.5', 'role'] == 'exact'
    # resonant fractions of the 4-body node: the a_1 node is inside the rho0 node, counted once
    assert close(n.loc['T.7', 'f_res_mc'], 1 - 0.001 / 0.013)
    assert close(n.loc['T.7', 'f_res_pdg'], 0.009 / 0.016)
    assert 'resonance_mismatch' not in flags(r4)


def test_toy_unstable_remainder():
    t, nodes, lk = toy()
    r = row(511, 'D- pi+ pi0', t, lk)
    # remainder (0.007 - 0.95 * 0.008) < 0: inclusive ratio of the node instead
    assert r['how'] == 'node' and 'remainder_unstable' in flags(r)
    assert close(r['c'], 0.007 / 0.0087) and close(r['sigma_c'], r['c'] * 0.001 / 0.007)
    assert close(row(511, 'D- rho+', t, lk)['c'], 0.95)
    # a remainder more than 100 % uncertain is unstable too
    pdg = [x if x[0] != 'T.9' else ('T.9', x[1], 0.0078, 0.001, False) for x in TOY_PDG]
    t2, _, lk2 = toy(pdg=pdg)
    r = row(511, 'D- pi+ pi0', t2, lk2)
    assert 'remainder_unstable' in flags(r) and close(r['c'], 0.0078 / 0.0087)
    # and a well-measured one is not
    pdg = [x if x[0] != 'T.9' else ('T.9', x[1], 0.0083, 0.0001, False) for x in TOY_PDG]
    t3, _, lk3 = toy(pdg=pdg)
    r = row(511, 'D- pi+ pi0', t3, lk3)
    assert 'remainder_unstable' not in flags(r) and close(r['c'], (0.0083 - 0.0076) / 0.0007)


def test_toy_product():
    t, nodes, lk = toy()
    r = row(511, 'D_s0*+ D-', t, lk)
    # two product nodes: 0.001 / (0.01 x 0.8) and 0.0004 / (0.01 x 0.2)
    c1, s1, c2, s2 = 0.125, 0.025, 0.2, 0.05
    w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
    c = (w1 * c1 + w2 * c2) / (w1 + w2)
    assert r['how'] == 'product' and r['pdgids'] == 'T.10;T.11' and r['n_prod'] == 2
    assert close(r['c'], c) and close(r['c'], 0.14) and close(r['sigma_c'], 1 / math.sqrt(w1 + w2))
    assert close(r['chi2_prod'], w1 * (c1 - c) ** 2 + w2 * (c2 - c) ** 2) and close(r['chi2_prod'], 1.8)
    assert close(r['c_prod'], r['c']) and 'c_extreme' in flags(r)
    assert close(nodes.set_index('uid').loc['T.10', 'bf_mc'], 0.008)
    # a line with a product node that also feeds a final-state node: the product fixes it
    r = row(511, 'D_1- pi+', t, lk)
    assert r['how'] == 'product' and close(r['c'], 0.00036 / (0.0009 * 0.5))
    assert nodes.set_index('uid').loc['T.18', 'role'] == 'covered'
    old = pm.PRODUCT_BEFORE_NODE
    pm.PRODUCT_BEFORE_NODE = False
    try:
        t2, _, lk2 = toy()
    finally:
        pm.PRODUCT_BEFORE_NODE = old
    r = row(511, 'D_1- pi+', t2, lk2)
    assert r['how'] == 'node' and r['pdgids'] == 'T.18' and close(r['c'], 0.0006 / 0.00045)
    assert close(r['c_prod'], 0.8)                                   # still stored
    assert row(511, 'D_s0*+ D-', t2, lk2)['how'] == 'product'


def test_toy_kaons_and_limits():
    t, nodes, lk = toy()
    ks, kl = row(511, 'J/psi K_S0', t, lk), row(511, 'J/psi K_L0', t, lk)
    # PDG K_S0 node x 2, compared with the K_S0 + K_L0 lines
    for r in (ks, kl):
        assert r['how'] == 'exact' and close(r['bf_pdg'], 0.0044) and close(r['c'], 1.1)
        assert close(r['bf_mc'], 0.002) and close(r['bf_mc_sum'], 0.004)
    r = row(511, 'K_S0 K_S0 pi0', t, lk)
    assert 'multi_ks' in flags(r) and r['pdg_class'] == 'none' and r['c'] == 1.0 and r['how'] == ''
    assert nodes.set_index('uid').loc['T.13', 'role'] == 'multi_ks'
    r = row(511, 'pi+ pi-', t, lk)
    assert r['pdg_class'] == 'limit' and r['c'] == 1.0 and close(r['bf_pdg'], 0.0005)
    assert 'above_limit' not in flags(r) and math.isnan(r['pull']) and r['pdgids'] == 'T.14'
    r = row(511, 'K+ K-', t, lk)
    assert r['pdg_class'] == 'limit' and r['c'] == 1.0 and 'above_limit' in flags(r)


def test_toy_resonance_mismatch():
    t, nodes, lk = toy()
    direct, res = row(521, 'anti-D0 pi+ pi0', t, lk), row(521, 'anti-D0 rho+', t, lk)
    assert close(res['c'], 8.0) and res['how'] == 'exact'
    assert direct['how'] == 'node' and close(direct['c'], (0.010 - 8.0 * 0.001) / 0.009)
    for r in (direct, res):
        assert 'resonance_mismatch' in flags(r) and r['res_node'] == 'T.21'
        assert close(r['f_res_mc'], 0.1) and close(r['f_res_pdg'], 0.8)
    assert 'resonance_mismatch' not in flags(row(511, 'J/psi K+ pi-', t, lk))


def test_toy_conjugate_blocks():
    t, nodes, lk = toy()
    # B- is an exact mirror (CDecay)
    r = row(-521, 'D0 rho-', t, lk)
    assert close(r['c'], 8.0) and 'block_asymmetric' not in flags(r)
    r = row(-511, 'D+ e- anti-nu_e', t, lk)
    assert r['how'] == 'exact' and close(r['c'], 1.05) and flags(r) == set()
    assert 'block_asymmetric' not in flags(row(511, 'D- e+ nu_e', t, lk))
    # anti-B0 writes part of J/psi anti-K*0 with a K*0: both rows take the c of B0 -> J/psi K*0
    a, b = row(-511, 'J/psi anti-K*0', t, lk), row(-511, 'J/psi K*0', t, lk)
    assert close(a['c'], 1.2) and close(b['c'], 1.2) and close(a['bf_mc'], 0.0006)
    assert 'block_asymmetric' not in flags(a) and 'block_asymmetric' in flags(b)
    # self-conjugate final state
    assert close(row(-511, 'J/psi K_S0', t, lk)['c'], 1.1)
    # a final state of one block only
    r = row(-511, 'K0 pi0', t, lk)
    assert r['pdg_class'] == 'none' and r['c'] == 1.0 and 'block_asymmetric' in flags(r)
    assert 'block_asymmetric' in flags(row(511, 'J/psi K+ pi-', t, lk))
    # every final state of the four blocks has a row
    assert set(t['mother_pdg']) == {511, -511, 521, -521}


def test_toy_exceptions():
    base, _, lk0 = toy()
    exc = [dict(pdgid='T.4', action='exclude_node', reason='test')]
    t, nodes, lk = toy(exc)
    r = row(511, 'J/psi K+ pi-', t, lk)
    assert r['pdg_class'] == 'none' and r['c'] == 1.0 and 'exception_applied' in flags(r)
    assert nodes.set_index('uid').loc['T.4', 'role'] == 'excluded'
    assert close(row(511, 'J/psi K*0', t, lk)['c'], 1.2)
    # no_fold: the K_2* line no longer feeds the node
    exc = [dict(pdgid='T.4', action='no_fold', states=['K_2*0'], reason='test')]
    t, nodes, lk = toy(exc)
    assert row(511, 'J/psi K_2*0', t, lk)['pdg_class'] == 'none'
    r = row(511, 'J/psi K+ pi-', t, lk)
    assert close(r['c'], (0.0015 - 1.2 * (0.0006 * 0.6667 + 0.0004)) / 0.0004)
    assert 'exception_applied' in flags(r)
    # fold: a narrow state is folded for one node
    pdg = TOY_PDG + [('T.30', 'B0 --> D- D_s()+ pi0', 0.004, 0.001, False)]
    t, _, lk = toy(pdg=pdg)
    assert 'T.30' not in row(511, 'D_s0*+ D-', t, lk)['pdgids']
    exc = [dict(pdgid='T.10', action='exclude_node', reason='test'),
           dict(pdgid='T.11', action='exclude_node', reason='test'),
           dict(pdgid='T.30', action='fold', states=['D_s0*+'], reason='test')]
    t, _, lk = toy(exc, pdg)
    r = row(511, 'D_s0*+ D-', t, lk)
    assert r['how'] == 'node' and r['pdgids'] == 'T.30' and close(r['c'], 0.004 / 0.008)
    # force_class and rewrite
    exc = [dict(mother=511, key=code('D- rho+'), action='force_class', reason='test', **{'class': 'none'})]
    t, _, lk = toy(exc)
    r = row(511, 'D- rho+', t, lk)
    assert r['pdg_class'] == 'none' and r['c'] == 1.0 and r['how'] == '' and 'exception_applied' in flags(r)
    pdg = [x if x[0] != 'T.8' else ('T.8', 'B0 --> D- rho(770)', 0.0076, 0.0005, False) for x in TOY_PDG]
    t, nodes, lk = toy(pdg=pdg)
    assert row(511, 'D- rho+', t, lk)['how'] == 'node'              # unkeyed text: no exact node
    exc = [dict(pdgid='T.8', action='rewrite', description='B0 --> D- rho+', reason='test')]
    t, nodes, lk = toy(exc, pdg)
    r = row(511, 'D- rho+', t, lk)
    assert r['how'] == 'exact' and close(r['c'], 0.95) and 'exception_applied' in flags(r)
    # the file format is checked
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'exc.json')
        for bad in ('[{"pdgid": "T.1", "action": "exclude_node"}]',          # no reason
                    '[{"pdgid": "T.1", "action": "drop", "reason": "x"}]',   # unknown action
                    '[{"pdgid": "T.1", "action": "fold", "reason": "x"}]',   # no states
                    '{"a": 1}'):
            with open(path, 'w') as f:
                f.write(bad)
            try:
                pm.load_exceptions(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f'accepted: {bad}')
        assert pm.load_exceptions(os.path.join(tmp, 'missing.json')) == []
    for e in pm.load_exceptions():
        assert e['reason'] and e['action'] == 'rewrite'            # only misprints of the listing


# ------------------------------------------------------------------------------------------
# the real files
# ------------------------------------------------------------------------------------------
def test_real_coverage():
    modes = MODES.modes
    assert len(TABLE) == len(modes) == len(LOOKUP)
    assert set(zip(TABLE['mother_pdg'], TABLE['key'])) == set(zip(modes['mother_pdg'], modes['key']))
    assert np.allclose(TABLE['bf_mc'], modes['bf_mc'])
    assert set(TABLE['pdg_class']) == {'measured', 'limit', 'none', 'inclusive'}
    assert set(TABLE['how']) == {'exact', 'node', 'product', ''}
    meas = TABLE['how'] != ''
    assert (TABLE.loc[meas, 'pdg_class'] == 'measured').all() and (TABLE.loc[meas, 'pdgids'] != '').all()
    assert (TABLE.loc[~meas, 'c'] == 1.0).all() and TABLE['c'].notna().all() and (TABLE['c'] > 0).all()
    assert TABLE.loc[meas, 'sigma_c'].notna().all() and TABLE.loc[~meas, 'pull'].isna().all()
    incl = TABLE['pdg_class'] == 'inclusive'
    assert incl.sum() == modes['is_inclusive'].sum() and (TABLE.loc[incl, 'how'] == '').all()
    assert (pm.has_flag(TABLE, 'tau') == TABLE['tau']).all() and TABLE['tau'].sum() == 92
    assert (pm.has_flag(TABLE, 'zero_bf') == (TABLE['bf_mc'] == 0)).all()
    known = {'tau', 'resonance_mismatch', 'multi_ks', 'remainder_unstable', 'above_limit',
             'exception_applied', 'block_asymmetric', 'zero_bf', 'c_extreme', ''}
    assert set(f for fl in TABLE['flags'] for f in fl.split(';')) <= known
    # shares of all B decays (scouting, B.12: 34 / 38.5 % plain, 6-7 % product, ~10 % none)
    for m, lo, hi in ((511, 0.33, 0.37), (521, 0.37, 0.41)):
        sh = pm.class_shares(TABLE, m)
        assert lo < sh['measured_exact'] + sh['measured_node'] < hi, sh
        assert 0.05 < sh['measured_product'] < 0.08 and 0.09 < sh['none'] < 0.11 and sh['limit'] < 0.01
        assert close(sum(sh.values()), MODES.exclusive_bf[m], 1e-9)
    # the conjugate blocks give the same shares
    for m in (511, 521):
        a, b = pm.class_shares(TABLE, m), pm.class_shares(TABLE, -m)
        assert all(abs(a[k] - b[k]) < 2e-3 for k in a), (a, b)


def test_real_semileptonic():
    r = row(511, 'D- e+ nu_e')
    assert r['how'] == 'exact' and r['pdgids'] == 'S042.37' and close(r['c'], 0.0212 / 0.0214)
    assert 0.985 < r['c'] < 0.995 and close(r['sigma_c'], 0.0006 / 0.0214) and flags(r) == set()
    assert close(row(511, 'D- mu+ nu_mu')['c'], r['c'])
    assert close(row(-511, 'D+ e- anti-nu_e')['c'], r['c'])
    assert row(511, 'D*- e+ nu_e')['pdgids'] == 'S042.181'
    tau = row(511, 'D- tau+ nu_tau')
    assert tau['how'] == 'exact' and tau['tau'] and 'tau' in flags(tau)
    assert row(511, 'Xu- e+ nu_e')['pdg_class'] == 'inclusive'
    # D** l nu through the product nodes (D_1, D'_1, D_2*)
    for names, ids in (('D_1- e+ nu_e', 'S042.420;S042.623'), ("D'_1- e+ nu_e", 'S042.421'),
                       ('D_2*- mu+ nu_mu', 'S042.419;S042.422')):
        r = row(511, names)
        assert r['how'] == 'product' and r['prod_pdgids'] == ids and close(r['c'], r['c_prod']), names
        assert 0.8 < r['c'] < 1.0 and r['n_prod'] == len(ids.split(';'))
    r = row(511, 'D_1- e+ nu_e')
    c1, s1 = 0.002852150285313824 / (0.00704 * 0.3994), 0.0002532034593495046 / (0.00704 * 0.3994)
    c2, s2 = 0.00102 / (0.00704 * 0.1719), 0.000158113883008419 / (0.00704 * 0.1719)
    w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
    assert close(r['c'], (w1 * c1 + w2 * c2) / (w1 + w2), 1e-4)
    # the broad states in B+ are well above the PDG products
    assert row(521, 'anti-D_0*0 e+ nu_e')['c'] < 0.5 and row(521, "anti-D'_10 e+ nu_e")['c'] < 0.7
    # B0 -> D_0*- l nu has only a limit as a product: its c comes from the Dbar0 pi- l nu node
    r = row(511, 'D_0*- e+ nu_e')
    assert r['how'] == 'node' and r['pdgids'] == 'S042.311' and 'above_limit' in flags(r)
    # with the other priority the D** lines take the D(*) pi l nu nodes
    old = pm.PRODUCT_BEFORE_NODE
    pm.PRODUCT_BEFORE_NODE = False
    try:
        t2 = pm.build_mode_pdg_table(MODES, PDG, dec_all=DEC_ALL, pdl=PDL)
    finally:
        pm.PRODUCT_BEFORE_NODE = old
    r1, r2 = row(511, 'D_1- e+ nu_e'), pm.pdg_info(t2, 511, code('D_1- e+ nu_e'))
    assert r2['how'] == 'node' and r2['pdgids'] == 'S042.312' and close(r2['c_prod'], r1['c_prod'])
    assert r2['c'] < 0.8 < r1['c']
    assert pm.pdg_info(t2, 511, code('D_s0*+ D-'))['how'] == 'product'


def test_real_jpsi():
    n = NODES.set_index('uid')
    # J/psi K+ pi-: the K* lines are part of the PDG final state
    node = n.loc['S042.4']
    assert node['description'] == 'B0 --> J/psi(1S) K+ pi-' and node['role'] == 'node'
    direct = row(511, 'J/psi K+ pi-')
    assert 3.5 < node['value'] / direct['bf_mc'] < 4.1           # line by line: wrong
    assert 0.82 < node['ratio'] < 0.92 and node['n_feeders'] == 3  # after folding K*, K_2*
    kst = row(511, 'J/psi K*0')
    assert kst['how'] == 'exact' and kst['pdgids'] == 'S042.22' and 0.95 < kst['c'] < 1.05
    assert close(kst['bf_mc'], 0.0002117 * 2 + 0.000847, 1e-3)    # K*0, K*S, K*L, K*0T lines
    assert direct['how'] == 'node' and direct['pdgids'] == 'S042.4' and 0.4 < direct['c'] < 0.9
    # the anti-B0 block writes the K*S / K*L lines with a K*0
    a, b = row(-511, 'J/psi anti-K*0'), row(-511, 'J/psi K*0')
    assert close(a['c'], kst['c']) and close(b['c'], kst['c'])
    assert 'block_asymmetric' in flags(b) and 'block_asymmetric' not in flags(a)
    assert close(a['bf_mc'] + b['bf_mc'], kst['bf_mc'])
    # J/psi K_1(1270): PDG writes "K(1270)"
    for m, names, pid in ((511, 'J/psi K_10', 'S042.203'), (521, 'J/psi K_1+', 'S041.271')):
        r = row(m, names)
        assert r['how'] == 'exact' and r['pdgids'] == pid and close(r['c'], 1.0, 1e-3), names
    # J/psi K0: PDG K0 node against the K_S0 + K_L0 lines
    ks, kl = row(511, 'J/psi K_S0'), row(511, 'J/psi K_L0')
    assert ks['how'] == kl['how'] == 'exact' and close(ks['c'], kl['c'])
    assert close(ks['bf_mc_sum'], ks['bf_mc'] + kl['bf_mc']) and 0.9 < ks['c'] < 1.1
    assert ks['pdgids'] == 'S042.23'


def test_real_double_counting():
    n = NODES.set_index('uid')
    node = n.loc['S042.1']
    assert node['description'] == 'B0 --> Dbar0 pi+ pi-'
    # the decay file lists the full PDG value as a direct line and the resonances on top
    direct = row(511, 'anti-D0 pi- pi+')
    assert close(direct['bf_mc'], node['value'], 1e-3)
    assert 0.48 < node['ratio'] < 0.58 and node['n_feeders'] >= 4
    assert direct['how'] == 'node' and direct['pdgids'] == 'S042.1' and direct['c'] < 0.6
    assert direct['pull'] < -5
    # the D** pi lines are fixed by their product nodes (text corrected by the exceptions file)
    r = row(511, 'D_2*- pi+')
    assert r['how'] == 'product' and r['pdgids'] == 'S042.124' and 'exception_applied' in flags(r)
    assert close(r['c'], 0.000238 / (0.00043 * 0.3998), 2e-2)
    # same pattern in D- K0 pi+ (direct line = full PDG value, D- K*+ on top)
    r = row(511, 'D- K0 pi+')
    assert r['how'] == 'node' and r['c'] < 0.6 and row(511, 'D- K*+')['how'] == 'exact'


def test_real_ds0():
    # the D(*) D_s0*(2317) lines carry about ten times the PDG product
    for m, names, pid in ((511, 'D_s0*+ D-', 'S042.253'), (511, 'D*- D_s0*+', 'S042.97'),
                          (521, 'anti-D0 D_s0*+', 'S041.321'), (521, 'anti-D*0 D_s0*+', 'S041.26')):
        r = row(m, names)
        assert r['how'] == 'product' and r['pdgids'] == pid, names
        assert 0.04 < r['c'] < 0.16 and r['pull'] < -10 and 'c_extreme' in flags(r)
        assert close(r['c'], r['bf_pdg'] / r['bf_mc'])              # D_s0* -> D_s pi0 is 100 % in the file
        cc = row(-m, PDL.names(pm.conjugate_key(r['key'], PDL)))
        assert close(cc['c'], r['c']) and cc['how'] == 'product'
    # D_s1(2460): exact node; the product node is stored next to it
    r = row(511, 'D*- D_s1+')
    assert r['how'] == 'exact' and r['n_prod'] == 1 and r['prod_pdgids'] == 'S042.100'


def test_real_misc():
    # limits
    lim = TABLE[TABLE['pdg_class'] == 'limit']
    assert len(lim) > 50 and (lim['c'] == 1.0).all() and lim['bf_pdg'].notna().all()
    above = lim[pm.has_flag(lim, 'above_limit')]
    assert ((above['bf_mc'] >= above['bf_pdg']) | (above['n_prod'] == 0)).all()
    r = row(511, 'eta_c(2S) K*0')
    assert r['pdg_class'] == 'limit' and 'above_limit' in flags(r)
    # eta_c is not folded: p pbar K keeps its own node
    assert row(511, 'p+ anti-p- K0')['how'] == 'exact'
    # sum node
    a, b = row(511, 'D*- D+ K0'), row(511, 'D- D*+ K0')
    assert a['how'] == b['how'] == 'node' and a['pdgids'] == b['pdgids'] == 'S042.265'
    assert close(a['c'], b['c']) and close(a['c'], 0.00641 / (a['bf_mc'] + b['bf_mc']), 1e-2)
    # non-resonant PHSP line where PDG sees resonances
    r = row(511, 'D*- omega pi+')
    assert 'resonance_mismatch' in flags(r) and r['f_res_mc'] == 0.0 and r['f_res_pdg'] > 0.5
    # unkeyed nodes with a sizeable value are few and listed
    unk = NODES[(NODES['kind'] == 'unkeyed') & (NODES['value'] > 1e-4) & ~NODES['is_limit']]
    assert 40 < len(unk) < 70
    assert (NODES.loc[NODES['kind'] == 'unkeyed', 'reason'] != '').all()


def test_csv_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 't.csv')
        pm.write_table(TABLE, path)
        back = pm.read_table(path)
    assert list(back['key']) == list(TABLE['key']) and list(back['flags']) == list(TABLE['flags'])
    assert list(back['how']) == list(TABLE['how']) and list(back['pdgids']) == list(TABLE['pdgids'])
    assert np.allclose(back['c'], TABLE['c'], rtol=1e-5)
    assert close(pm.pdg_info(back, 511, code('D- e+ nu_e'))['c'], 0.0212 / 0.0214, 1e-5)
    assert pm.pdg_info(TABLE, 511, [211, 211]) is None
    # the committed table is the one the code gives
    if os.path.exists(pm.DEFAULT_TABLE_CSV):
        disk = pm.read_table(pm.DEFAULT_TABLE_CSV)
        assert list(disk['key']) == list(TABLE['key']) and list(disk['how']) == list(TABLE['how'])
        assert np.allclose(disk['c'], TABLE['c'], rtol=1e-5)


def test_review_report():
    review, summary = pm.review_report(TABLE, mode_table=MODES)
    files = [f for f in os.listdir(pm.CSV_DIR) if f.startswith('bbbar_modes_')]
    assert len(files) == 12
    # summary: percent of the yield per region, component and side
    assert set(summary['region']) == {'SR', 'combCR', 'WC'}
    assert np.allclose(summary[pm.SUMMARY_CLASSES].sum(axis=1), 100.0)
    s = summary.set_index(['region', 'component', 'side'])
    assert s.loc[('WC', 'comb', 'ell'), 'excl_sl'] > 70 and s.loc[('SR', 'hadB', 'D'), 'excl_sl'] < 2
    # the same split as the first look of B.13 (PYTHIA columns are independent of the PDG match)
    assert abs(s.loc[('SR', 'comb', 'D'), 'pythia_1c'] - 38.6) < 0.1
    assert abs(s.loc[('combCR', 'hadB', 'D'), 'pythia_2c'] - 24.8) < 0.1
    assert abs(s.loc[('SR', 'comb', 'D'), 'excl_sl'] - 13.4) < 0.1
    # review: B0 / B+ rows, sorted by the largest yield share
    assert (review['max_share'].diff().dropna() <= 1e-12).all()
    assert set(review['mother_pdg']) <= {511, 521, -511, -521}
    assert (review.loc[review['n_top95'] > 0, 'mother_pdg'] > 0).mean() > 0.95
    assert 100 < (review['n_top95'] > 0).sum() < 250
    why = set(w for x in review['why'] for w in x.split(';'))
    assert why == {'top95', 'pull', 'flag'}
    big = review[np.isfinite(review['pull']) & (review['pull'].abs() > pm.PULL_REVIEW)]
    assert big['why'].str.contains('pull').all() and len(big) > 20
    in_table = TABLE[np.isfinite(TABLE['pull']) & (TABLE['pull'].abs() > pm.PULL_REVIEW) & (TABLE['mother_pdg'] > 0)]
    assert set(zip(in_table['mother_pdg'], in_table['key'])) <= set(zip(review['mother_pdg'], review['key']))
    r = review[(review['mother_pdg'] == 511) & (review['names'] == 'D_s0*+ D-')].iloc[0]
    assert r['how'] == 'product' and r['max_share'] > 0.02 and r['n_top95'] > 10
    # one file is enough to run it
    rev1, sum1 = pm.review_report(TABLE, [os.path.join(pm.CSV_DIR, 'bbbar_modes_run1_e_combCR.csv')], MODES)
    assert set(sum1['region']) == {'combCR'} and len(sum1) == 3


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')


if __name__ == '__main__':
    main()
