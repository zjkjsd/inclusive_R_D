"""Tests of scripts/tools/pdg_match.py (comparison of the decay file with PDG).

Run from the repository root:  python3 scripts/tests/test_pdg_match.py

The tests need only files of the repository: the decay file, evt.pdl, the cached PDG table
(Samples/dec/pdg_bf_B_2026.json), the exceptions file and the decay-mode tables in
Notebooks/Plotting/csv_output. They do not import the `pdg` package and need no network.
"""
import json
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
    """Relative comparison (BF-sized numbers are compared at their own scale)."""
    return a == b or abs(a - b) <= tol * max(abs(a), abs(b))


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
    # asymmetric errors: the mean (S041.725: the two errors differ by a factor 2)
    ep, en = raw['S041.725']['error_positive'], raw['S041.725']['error_negative']
    assert abs(ep - en) > 0.5 * min(ep, en)
    assert close(by_uid['S041.725']['err'], 0.5 * (ep + en), 1e-9)
    # the K_S0 factor 2 is not applied to nodes with several K_S0
    assert close(ks[0]['value'], raw[ks[0]['pdgid']]['value'], 1e-12)


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


def toy(exceptions=(), pdg=TOY_PDG, dec=TOY_DEC):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'toy.dec')
        with open(path, 'w') as f:
            f.write(dec)
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
    assert close(n.loc['T.4', 'ratio'], 0.0015 / bf_node)
    # the K_2* line reaches the node through a resonance: it takes no c from it (unmeasured,
    # c = 1 +- 1). The direct line takes the remainder: total minus K* (exact c = 1.2) minus K_2*
    # (1 +- 1), with 20 % of the subtracted part added as interference allowance
    r = row(511, 'J/psi K_2*0', t, lk)
    assert r['how'] == '' and r['pdg_class'] == 'none' and r['c'] == 1.0 and close(r['sigma_c'], 1.0)
    assert 'unmeasured' in flags(r) and 'remainder_unstable' not in flags(r)
    sub = 1.2 * feed_kst + 0.00025
    c = (0.0015 - sub) / 0.0004
    s = math.hypot(math.sqrt(0.0002 ** 2 + (0.1 * feed_kst) ** 2 + 0.00025 ** 2), pm.F_INTERFERENCE * sub) / 0.0004
    r = row(511, 'J/psi K+ pi-', t, lk)
    assert r['how'] == 'node' and r['pdgids'] == 'T.4' and close(r['c'], c) and close(r['sigma_c'], s)
    assert close(r['p_node'], 1.0) and 'remainder_unstable' in flags(r) and 'unmeasured' not in flags(r)
    assert n.loc['T.4', 'role'] == 'node' and close(n.loc['T.4', 'closure'], 1.0)
    # with a precise total the remainder is well defined
    pdg = [x if x[0] != 'T.4' else ('T.4', x[1], 0.0020, 0.0001, False) for x in TOY_PDG]
    tp, np_, lkp = toy(pdg=pdg)
    r = row(511, 'J/psi K+ pi-', tp, lkp)
    c = (0.0020 - sub) / 0.0004
    s = math.hypot(math.sqrt(0.0001 ** 2 + (0.1 * feed_kst) ** 2 + 0.00025 ** 2), pm.F_INTERFERENCE * sub) / 0.0004
    assert r['how'] == 'node' and r['pdgids'] == 'T.4' and close(r['c'], c) and close(r['sigma_c'], s)
    assert close(r['p_node'], 1.0) and not flags(r) & {'unmeasured', 'remainder_unstable', 'node_not_closed'}
    assert np_.set_index('uid').loc['T.4', 'role'] == 'node'
    assert 'unmeasured' in flags(row(511, 'J/psi K_2*0', tp, lkp))
    assert close(row(511, 'J/psi K*0', t, lk)['c'], 1.2)            # the exact line is untouched
    # nested nodes, from the fewest bodies up
    assert close(row(511, 'D*- a_1+', t, lk)['c'], 1.2)
    r3 = row(511, 'D*- rho0 pi+', t, lk)
    c3 = (0.009 - 1.2 * 0.006) / 0.002
    s3 = math.hypot(math.sqrt(0.0002 ** 2 + (0.05 * 0.006) ** 2), pm.F_INTERFERENCE * 1.2 * 0.006) / 0.002
    assert r3['how'] == 'node' and r3['pdgids'] == 'T.6' and close(r3['c'], c3) and close(r3['sigma_c'], s3)
    r4 = row(511, 'D*- pi+ pi+ pi-', t, lk)
    sub4 = 1.2 * 0.010 + c3 * 0.002
    c4 = (0.016 - sub4) / 0.001
    s4 = math.hypot(math.sqrt(0.0002 ** 2 + (0.05 * 0.010) ** 2 + (s3 * 0.002) ** 2), pm.F_INTERFERENCE * sub4) / 0.001
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
    # remainder (0.007 - 0.95 * 0.008) < 0: the direct line is clamped at c = 0, with the
    # uncertainty of the remainder (and the interference allowance)
    sub = 0.95 * 0.008
    s = math.hypot(math.sqrt(0.001 ** 2 + (r_sigma(t, lk, 'D- rho+') * 0.008) ** 2), pm.F_INTERFERENCE * sub) / 0.0007
    assert r['how'] == 'node' and r['pdg_class'] == 'measured' and r['c'] == 0.0 and close(r['sigma_c'], s)
    assert {'remainder_clamped', 'remainder_unstable'} <= flags(r) and 'unmeasured' not in flags(r)
    assert close(row(511, 'D- rho+', t, lk)['c'], 0.95)
    # a remainder more than 100 % uncertain keeps its value and is flagged
    pdg = [x if x[0] != 'T.9' else ('T.9', x[1], 0.0078, 0.001, False) for x in TOY_PDG]
    t2, _, lk2 = toy(pdg=pdg)
    r = row(511, 'D- pi+ pi0', t2, lk2)
    assert close(r['c'], (0.0078 - sub) / 0.0007) and 'remainder_unstable' in flags(r)
    assert 'remainder_clamped' not in flags(r)
    # a small remainder of a large subtraction stays uncertain even with a precise total: the
    # interference allowance (20 % of 0.0076) dominates
    pdg = [x if x[0] != 'T.9' else ('T.9', x[1], 0.0083, 0.0001, False) for x in TOY_PDG]
    t3, _, lk3 = toy(pdg=pdg)
    r = row(511, 'D- pi+ pi0', t3, lk3)
    assert close(r['c'], (0.0083 - sub) / 0.0007) and 'remainder_clamped' not in flags(r)
    assert r['sigma_c'] > pm.F_INTERFERENCE * sub / 0.0007 and 'remainder_unstable' in flags(r)


def r_sigma(t, lk, names):
    return row(511, names, t, lk)['sigma_c']


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
    # above its own limit: c = limit/2 +- limit/2 since the N1 convention of point 3.3 (2026-11-20)
    assert r['pdg_class'] == 'limit' and r['how'] == 'half_limit' and 'limit_auto' in flags(r)
    assert r['c'] < 1.0 and close(r['c'], r['sigma_c']) and 'above_limit' in flags(r)


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
    # J/psi K0: the anti-B0 block has K_S0 only (BF 0.002 against 0.004 in B0): the corrected BF of
    # B0 is transferred (c x 0.004 / 0.002), flag cp_transfer
    r, src = row(-511, 'J/psi K_S0', t, lk), row(511, 'J/psi K_S0', t, lk)
    assert {'block_asymmetric', 'cp_transfer'} <= flags(r) and r['how'] == src['how'] and r['pdg_class'] == 'measured'
    assert close(r['bf_mc_sum'], 0.002) and close(r['c'], src['c'] * 0.004 / 0.002)
    assert close(r['sigma_c'], src['sigma_c'] * 0.004 / 0.002)
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
    r = row(511, 'D_s0*+ D-', t, lk)                           # feeds T.30 now, through a resonance
    assert r['how'] == '' and {'unmeasured', 'exception_applied'} <= flags(r) and close(r['sigma_c'], 1.0)
    # force_class and rewrite
    exc = [dict(mother=511, key=code('D- rho+'), action='force_class', reason='test', **{'class': 'none'})]
    t, _, lk = toy(exc)
    r = row(511, 'D- rho+', t, lk)
    assert r['pdg_class'] == 'none' and r['c'] == 1.0 and r['how'] == '' and 'exception_applied' in flags(r)
    pdg = [x if x[0] != 'T.8' else ('T.8', 'B0 --> D- rho(770)', 0.0076, 0.0005, False) for x in TOY_PDG]
    t, nodes, lk = toy(pdg=pdg)
    r = row(511, 'D- rho+', t, lk)                                  # unkeyed text: no exact node
    assert r['how'] == '' and 'unmeasured' in flags(r)
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
        assert e['reason'] and e['action']
        if 'pdgid' in e:
            assert 'PDG text' in e['reason']                       # justified from the listing


# a second toy for the checks added after the review (limits in remainders, repeated product
# nodes, P_MIN, closure, ambiguous nodes, twins, blocks with different BFs)
TOY2_DEC = """
Decay B0
0.0010 D- rho+ PHSP;
0.0010 D- pi+ pi0 PHSP;
0.0020 D_0*- e+ nu_e PHSP;
0.0010 anti-D0 pi- e+ nu_e PHSP;
0.0010 D_s0*+ D- PHSP;
0.0100 D- a_1+ PHSP;
0.0010 D- rho0 pi+ PHSP;
0.0003 K+ K- PHSP;
0.0005 D'_s1+ D- PHSP;
0.0010 J/psi K_2*0 PHSP;
0.0003 D*- K0 pi+ PHSP;
0.0002 D*+ K0 pi- PHSP;
Enddecay
CDecay anti-B0
Decay B+
0.0006 anti-D*0 K+ anti-K*0 PHSP;
0.0015 anti-D*0 K+ K*0 PHSP;
Enddecay
Decay B-
0.0006 D*0 K- anti-K*0 PHSP;
0.0015 D*0 K- K*0 PHSP;
Enddecay
Decay rho+
1.0 pi+ pi0 PHSP;
Enddecay
Decay rho0
0.5 pi+ pi- PHSP;
0.5 pi0 pi0 PHSP;
Enddecay
Decay D_0*-
0.5 anti-D0 pi- PHSP;
0.5 D- pi0 PHSP;
Enddecay
Decay D_s0*+
1.0 D_s+ pi0 PHSP;
Enddecay
Decay a_1+
0.05 pi+ pi+ pi- PHSP;
0.95 pi+ pi0 pi0 PHSP;
Enddecay
Decay D'_s1+
0.5 D*0 K+ PHSP;
0.5 D*+ K0 PHSP;
Enddecay
Decay K_2*0
0.5 K+ pi- PHSP;
0.5 K0 pi0 PHSP;
Enddecay
"""

TOY2_PDG = [
    ('U.1', 'B0 --> D- pi+ pi0', 0.010, 0.001, False),
    ('U.2', 'B0 --> D- rho+', 0.0015, 0.0, True),
    ('U.3', 'B0 --> Dbar0 pi- lepton+ nu_lepton', 0.003, 0.0002, False),
    ('U.4', 'B0 --> D_0^*(2300)- lepton+ nu_lepton , D_0^*()- --> Dbar0 pi-', 0.0002, 0.0, True),
    ('U.5', 'B0 --> D_s0^*(2317)+ D-', 0.002, 0.0002, False),
    ('U.6', 'B0 --> D_s0^*(2317)+ D- , D_s0^*()+ --> D_s()+ pi0', 0.0015, 0.0, True),
    ('U.7', 'B0 --> D- pi+ pi+ pi-', 0.002, 0.0002, False),
    ('U.8', 'B0 --> K+ K-', 0.0001, 0.0, True),
    ('U.9', 'B0 --> K+ K-', 0.0002, 0.0, True),
    ('U.10', 'B0 --> D- D_s1(2536)+ , D_s1()+ --> D^*()0 K+ + D^*()+ K0', 0.0006, 0.0001, False),
    ('U.11', 'B0 --> D- D_s1(2536)+ , D_s1()+ --> D^*()0 K+', 0.0003, 0.0001, False),
    ('U.12', 'B0 --> D- D_s1(2536)+ , D_s1()+ --> D^*()+ K0', 0.0004, 0.0002, False),
    ('U.13', 'B0 --> D- D_s1(2536)+ , D_s1()+ --> D^*()+ K0', 0.00035, 0.0001, False),
    ('U.14', 'B0 --> J/psi(1S) K+ pi-', 0.0006, 0.00003, False),
    ('U.15', 'B0 --> J/psi(1S) K0 pi0', 0.0002, 0.00002, False),
    ('U.16', 'B0 --> D^*(2010)- K0 pi+', 0.0003, 0.00003, False),
    ('U.20', 'B+ --> Dbar^*(2007)0 K+ Kbar^*(892)0', 0.0012, 0.0001, False),
]


def toy2():
    return toy(pdg=TOY2_PDG, dec=TOY2_DEC)


def test_toy_limit_caps():
    t, nodes, lk = toy2()
    n = nodes.set_index('uid')
    # D- pi+ pi0 = 0.010: D- rho+ (BF 1e-3, own limit 1.5e-3) takes no c from the node; it is
    # below its limit and is subtracted as it is, with 100 %
    r = row(511, 'D- rho+', t, lk)
    assert r['pdg_class'] == 'limit' and r['how'] == '' and r['c'] == 1.0 and r['pdgids'] == 'U.2'
    assert 'limit_capped' not in flags(r) and 'above_limit' not in flags(r)
    d = row(511, 'D- pi+ pi0', t, lk)
    assert d['how'] == 'node' and d['pdgids'] == 'U.1' and close(d['c'], (0.010 - 0.001) / 0.001)
    assert close(d['sigma_c'], math.hypot(math.sqrt(0.001 ** 2 + 0.001 ** 2), pm.F_INTERFERENCE * 0.001) / 0.001)
    # product limit: D_0*- l nu < 2e-4 (as D_0* -> Dbar0 pi-), below the decay file: limit/2 +- limit/2
    r = row(511, 'D_0*- e+ nu_e', t, lk)
    assert r['pdg_class'] == 'limit' and r['c'] == 1.0 and r['pdgids'] == 'U.4'
    assert {'limit_capped', 'above_limit'} <= flags(r)
    d = row(511, 'anti-D0 pi- e+ nu_e', t, lk)
    half = 0.5 * 0.0002
    assert d['how'] == 'node' and close(d['c'], (0.003 - half) / 0.001)
    assert close(d['sigma_c'], math.hypot(math.sqrt(0.0002 ** 2 + half ** 2), pm.F_INTERFERENCE * half) / 0.001)
    # half_limit exception: the line takes c = limit/2 +- limit/2 itself
    exc = [dict(mother=511, key=code('D_0*- e+ nu_e'), action='half_limit', reason='test')]
    th, _, lkh = toy(exc, TOY2_PDG, TOY2_DEC)
    r = row(511, 'D_0*- e+ nu_e', th, lkh)
    cap = 0.0002 / (r['bf_mc'] * 1.0)
    assert r['how'] == 'half_limit' and r['pdg_class'] == 'limit' and r['pdgids'] == 'U.4'
    assert close(r['c'], 0.5 * pm.pdg_info(th, 511, code('D_0*- e+ nu_e'))['bf_pdg'] / r['bf_mc'])
    assert close(r['sigma_c'], r['c']) and 'exception_applied' in flags(r)
    assert close(row(511, 'anti-D0 pi- e+ nu_e', th, lkh)['c'], d['c'])          # same remainder
    for bad in (dict(mother=511, key=code('D- rho+'), action='half_limit', reason='below its limit'),
                dict(mother=511, key=code('J/psi K*0'), action='half_limit', reason='no limit')):
        try:
            toy([bad], TOY2_PDG, TOY2_DEC)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted: {bad}')
    # the closure counts the capped line with the same limit/2 as the remainder: the node closes
    assert 'node_not_closed' not in flags(d) and d['unclosed_nodes'] == ''
    assert n.loc['U.3[e]', 'role'] == 'node' and abs(n.loc['U.3[e]', 'closure_pull']) < 1e-6
    # above_limit is tested on max(1, c) x BF: exact c = 2, product limit 1.5e-3 on 1e-3 x 1.0
    r = row(511, 'D_s0*+ D-', t, lk)
    assert r['how'] == 'exact' and close(r['c'], 2.0) and 'above_limit' in flags(r)
    # two limits on one final state: the tighter one is reported, both are listed
    r = row(511, 'K+ K-', t, lk)
    assert r['pdg_class'] == 'limit' and close(r['bf_pdg'], 0.0001) and r['pdgids'] == 'U.8;U.9'
    assert 'above_limit' in flags(r)


def test_toy_p_min_and_closure():
    t, nodes, lk = toy2()
    n = nodes.set_index('uid')
    # D- pi+ pi+ pi-: 5 % of D- a_1+ (below P_MIN) and 50 % of D- rho0 pi+ feed it
    a1, rho = row(511, 'D- a_1+', t, lk), row(511, 'D- rho0 pi+', t, lk)
    assert a1['how'] == '' and a1['c'] == 1.0 and a1['pdg_class'] == 'none'
    # the 50 % line reaches the node through a resonance: unmeasured, and the node keeps its MC sum
    assert rho['how'] == '' and rho['pdg_class'] == 'none' and 'unmeasured' in flags(rho)
    assert rho['c'] == 1.0 and close(rho['sigma_c'], 1.0)
    assert close(n.loc['U.7', 'bf_mc_weighted'], n.loc['U.7', 'bf_mc']) and n.loc['U.7', 'role'] == 'covered'
    # one resonance line in two 3-body nodes: no c from either; both closures are reported
    r = row(511, 'J/psi K_2*0', t, lk)
    assert r['how'] == '' and r['c'] == 1.0 and 'unmeasured' in flags(r) and 'node_ambiguous' not in flags(r)
    assert n.loc['U.14', 'role'] == n.loc['U.15', 'role'] == 'covered'
    assert close(n.loc['U.14', 'closure'], n.loc['U.14', 'bf_mc'] / 0.0006)
    assert 'U.15' in r['unclosed_nodes'].split(';') and n.loc['U.15', 'closure_pull'] > 3


def test_toy_independent_products():
    t, nodes, lk = toy2()
    n = nodes.set_index('uid')
    # sum node U.10 and its parts U.11, U.12 (U.13 repeats U.12): only the sum node is used
    r = row(511, "D'_s1+ D-", t, lk)
    assert r['how'] == 'product' and r['pdgids'] == 'U.10' and r['n_prod'] == 1
    assert close(r['c'], 1.2) and close(r['sigma_c'], 0.2) and close(r['chi2_prod'], 0.0)
    assert list(n.loc[['U.11', 'U.12', 'U.13'], 'role']) == ['duplicate'] * 3
    # without the sum node the two parts are used, the more precise of the repeated pair
    pdg = [x for x in TOY2_PDG if x[0] != 'U.10']
    t2, _, lk2 = toy(pdg=pdg, dec=TOY2_DEC)
    r = row(511, "D'_s1+ D-", t2, lk2)
    assert r['prod_pdgids'] == 'U.11;U.13' and r['n_prod'] == 2


def test_toy_twins_and_block_bf():
    t, nodes, lk = toy2()
    # D*+ K0 pi- has no node; its conjugate D*- K0 pi+ is measured
    r = row(511, 'D*+ K0 pi-', t, lk)
    assert r['pdg_class'] == 'none' and r['c'] == 1.0 and 'flavor_twin' in flags(r) and r['twin_pdgids'] == 'U.16'
    assert row(511, 'D*- K0 pi+', t, lk)['how'] == 'exact'
    # B+: K+ K*0 is the K+ anti-K*0 mode with the other K*0 flavor
    r = row(521, 'anti-D*0 K+ K*0', t, lk)
    assert r['pdg_class'] == 'none' and 'flavor_twin' in flags(r) and r['twin_pdgids'] == 'U.20'
    assert close(row(521, 'anti-D*0 K+ anti-K*0', t, lk)['c'], 2.0)
    # B- swaps the two BFs: the corrected BF of B+ (c = 2 on 6e-4) is transferred to the 1.5e-3 line
    r = row(-521, 'D*0 K- K*0', t, lk)
    assert {'block_asymmetric', 'cp_transfer'} <= flags(r) and r['pdgids'] == 'U.20'
    assert close(r['bf_mc_sum'], 0.0015) and close(r['c'], 2.0 * 0.0006 / 0.0015)
    r = row(-521, 'D*0 K- anti-K*0', t, lk)
    assert 'block_asymmetric' in flags(r) and r['pdg_class'] == 'none'
    # the mirrored anti-B0 block carries everything over
    r = row(-511, "D'_s1- D+", t, lk)
    assert close(r['c'], 1.2) and 'block_asymmetric' not in flags(r)


# ------------------------------------------------------------------------------------------
# the real files
# ------------------------------------------------------------------------------------------
def test_real_coverage():
    modes = MODES.modes
    assert len(TABLE) == len(modes) == len(LOOKUP)
    assert set(zip(TABLE['mother_pdg'], TABLE['key'])) == set(zip(modes['mother_pdg'], modes['key']))
    assert np.allclose(TABLE['bf_mc'], modes['bf_mc'])
    assert set(TABLE['pdg_class']) == {'measured', 'limit', 'none', 'inclusive'}
    assert set(TABLE['how']) == {'exact', 'node', 'product', 'isospin', 'half_limit', 'veto', 'nonres', 'external', ''}
    meas = TABLE['how'] != ''
    hl, vt = TABLE['how'] == 'half_limit', TABLE['how'] == 'veto'
    assert (TABLE.loc[meas & ~hl & ~vt, 'pdg_class'] == 'measured').all()
    assert (TABLE.loc[meas & ~vt, 'pdgids'] != '').all()
    assert (TABLE.loc[hl, 'pdg_class'] == 'limit').all() and (TABLE.loc[vt, 'pdg_class'] == 'none').all()
    assert (TABLE.loc[vt, 'c'] == 0).all() and vt.sum() == 6                  # B+ 4349, B0 806, 1002 (P7) and their conjugates
    assert (TABLE.loc[~meas, 'c'] == 1.0).all() and TABLE['c'].notna().all() and (TABLE['c'] >= 0).all()
    zero = (TABLE['c'] == 0) & (TABLE['how'] != 'veto')            # only clamped remainders
    assert zero.any() and (pm.has_flag(TABLE, 'remainder_clamped') == zero).all()
    assert TABLE.loc[meas, 'sigma_c'].notna().all() and TABLE.loc[~meas, 'pull'].isna().all()
    incl = TABLE['pdg_class'] == 'inclusive'
    assert incl.sum() == modes['is_inclusive'].sum() and (TABLE.loc[incl, 'how'] == '').all()
    assert (pm.has_flag(TABLE, 'tau') == TABLE['tau']).all() and TABLE['tau'].sum() == 92
    assert (pm.has_flag(TABLE, 'zero_bf') == (TABLE['bf_mc'] == 0)).all()
    known = {'tau', 'resonance_mismatch', 'multi_ks', 'remainder_unstable', 'above_limit', 'limit_auto',
             'exception_applied', 'block_asymmetric', 'zero_bf', 'c_extreme', 'limit_capped',
             'node_ambiguous', 'node_not_closed', 'flavor_twin', 'unmeasured', 'isospin',
             'remainder_clamped', 'measurement_conflict', 'cp_transfer', 'order_dependent',
             'nonres_source', 'external', ''}
    assert set(f for fl in TABLE['flags'] for f in fl.split(';')) <= known
    # shares of all B decays (scouting, B.12: 34 / 38.5 % plain, 6-7 % product, ~10 % none)
    for m, lo, hi in ((511, 0.33, 0.37), (521, 0.37, 0.41)):
        sh = pm.class_shares(TABLE, m)
        assert lo < sh['measured_exact'] + sh['measured_node'] < hi, sh
            # none: about 8 % since 3.4 (2026-11-20) moved 2.5 % to measured (LHCb external c)
        assert 0.05 < sh['measured_product'] < 0.08 and 0.07 < sh['none'] < 0.11 and sh['limit'] < 0.01
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
    # B0 -> D_0*- l nu has only a product limit (S042.418 < 4.4e-4): by exception it takes the c of
    # its isospin partner B+ -> anti-D_0*0 l nu (product S041.503); it stays above the limit
    src = row(521, 'anti-D_0*0 e+ nu_e')
    for names in ('D_0*- e+ nu_e', 'D_0*- mu+ nu_mu'):
        r = row(511, names)
        assert r['pdg_class'] == 'measured' and r['how'] == 'isospin' and r['pdgids'] == 'S041.503', names
        scale = 0.00389 / 0.00362 / pm.TAU_RATIO                 # BF(B+) = TAU_RATIO x BF(B0)
        assert close(r['c'], src['c'] * scale) and close(r['sigma_c'], src['sigma_c'] * scale) and 0.3 < r['c'] < 0.4
        assert {'isospin', 'exception_applied', 'above_limit'} <= flags(r)
        assert close(r['bf_pdg'], r['c'] * 0.00362)
    assert close(row(-511, 'D_0*+ e- anti-nu_e')['c'], row(511, 'D_0*- e+ nu_e')['c'])
    # the non-resonant D(*) pi l nu lines are not measured (exception, M2): c = 1 +- 1
    for m, names in ((511, 'anti-D0 pi- e+ nu_e'), (511, 'anti-D*0 pi- mu+ nu_mu'),
                     (521, 'D*- pi+ e+ nu_e'), (521, 'D- pi+ mu+ nu_mu')):
        d = row(m, names)
        assert d['pdg_class'] == 'none' and d['how'] == '' and d['c'] == 1.0 and close(d['sigma_c'], 1.0), names
        assert {'unmeasured', 'exception_applied'} <= flags(d) and 'remainder_unstable' not in flags(d)
    # the D** lines keep their product c whatever the D(*) pi l nu totals say
    assert row(511, 'D_1- e+ nu_e')['how'] == 'product' and row(511, 'D_s0*+ D-')['how'] == 'product'


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
    # J/psi K_2*0 has no entry of its own (c = 1 +- 1); the direct line takes the remainder,
    # computed here by hand from the node table, and the node closes
    k2 = row(511, 'J/psi K_2*0')
    assert k2['how'] == '' and 'unmeasured' in flags(k2)
    fk = {k: v for k, v, _ in node['feeders']}
    kk = lambda names: pm.collapse_key(code(names))
    sub = kst['c'] * fk[kk('J/psi K*0')] + fk[kk('J/psi K_2*0')]
    var = (kst['sigma_c'] * fk[kk('J/psi K*0')]) ** 2 + fk[kk('J/psi K_2*0')] ** 2 + node['err'] ** 2
    D = fk[kk('J/psi K+ pi-')]
    assert direct['how'] == 'node' and close(direct['c'], (node['value'] - sub) / D)
    assert close(direct['sigma_c'], math.hypot(math.sqrt(var), pm.F_INTERFERENCE * sub) / D)
    assert 'remainder_unstable' in flags(direct) and abs(node['closure_pull']) < 1e-6
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
    assert direct['how'] == 'node' and direct['pdgids'] == 'S042.1' and 0.25 < direct['c'] < 0.31
    assert 0.14 < direct['sigma_c'] < 0.18 and direct['pull'] < -3          # G4: 0.28 +- 0.16
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
    assert len(lim) > 50 and lim['bf_pdg'].notna().all()
    assert (lim['c'][lim['how'] == ''] == 1.0).all() and set(lim['how']) == {'', 'half_limit'}
    above = lim[pm.has_flag(lim, 'above_limit')]
    assert ((above['bf_mc'] >= above['bf_pdg']) | (above['n_prod'] == 0)).all()
    r = row(511, 'eta_c(2S) K*0')
    assert r['pdg_class'] == 'limit' and 'above_limit' in flags(r)
    # eta_c is not folded: p pbar K keeps its own node
    assert row(511, 'p+ anti-p- K0')['how'] == 'exact'
    # sum node
    a, b = row(511, 'D*- D+ K0'), row(511, 'D- D*+ K0')
    assert a['how'] == b['how'] == 'node' and a['pdgids'] == b['pdgids'] == 'S042.265'
    # P6: the D_s1(2536)+ line (D'_s1+ D-, half to D*+ K0) is folded into the total
    s1 = row(511, "D'_s1+ D-")['c'] * 0.00043 * 0.5
    assert close(a['c'], b['c']) and close(a['c'], (0.00641 - s1) / (a['bf_mc'] + b['bf_mc']), 1e-2)
    # non-resonant PHSP line where PDG sees resonances
    r = row(511, 'D*- omega pi+')
    assert 'resonance_mismatch' in flags(r) and r['f_res_mc'] == 0.0 and r['f_res_pdg'] > 0.5
    # unkeyed nodes with a sizeable value are few and listed
    unk = NODES[(NODES['kind'] == 'unkeyed') & (NODES['value'] > 1e-4) & ~NODES['is_limit']]
    assert 40 < len(unk) < 70
    assert (NODES.loc[NODES['kind'] == 'unkeyed', 'reason'] != '').all()


def a1_share():
    g = pm.DecayGraph(DEC_ALL, PDL)
    return g.decay_once('a_1+', g.default_fold()).get(pm.collapse_key(code('pi+ pi+ pi-')), 0.0)


def d1_share():
    g = pm.DecayGraph(DEC_ALL, PDL)
    return g.expand(('D_1-', 'pi+'), g.default_fold()).get(pm.collapse_key(code('D- pi+ pi+ pi-')), 0.0)


def test_real_review_fixes():
    n = NODES.set_index('uid')
    # D- a_1+ (D1): the line stands for the D- 3pi final state (direct_line exception); its exact
    # node S042.93 (54 % error) is excluded and c comes from the 10 % total S042.88, which closes
    r = row(511, 'a_1+ D-')
    assert r['how'] == 'node' and r['pdgids'] == 'S042.88' and 1.0 < r['c'] < 1.1 and 0.09 < r['sigma_c'] < 0.13
    assert 'exception_applied' in flags(r) and 'node_not_closed' not in flags(r)
    assert n.loc['S042.93', 'role'] == 'excluded' and abs(n.loc['S042.88', 'closure_pull']) < 1e-6
    d1 = row(511, 'D_1- pi+')
    assert close(r['c'], (n.loc['S042.88', 'value'] - d1['c'] * 0.00093 * d1_share()) / (0.01071429 * a1_share()), 1e-3)
    # D2: the D_s0* lines keep their product c; 20 % of c is added in quadrature (sub-decay BF)
    r = row(511, 'D_s0*+ D-')
    assert r['how'] == 'product' and close(r['sigma_c'], math.hypot(0.021656, 0.2 * r['c']), 1e-3)
    # psi(3770) K0: isospin image of B+ -> psi(3770) K+ (products, P+), all three neutral-kaon lines
    src = row(521, 'psi(3770) K+')
    for names in ('psi(3770) K_S0', 'psi(3770) K_L0'):
        r = row(511, names)
        assert r['how'] == 'isospin' and r['pdgids'] == 'S041.361;S041.362', names
        assert close(r['c'], src['c'] * src['bf_mc'] / 0.00048 / pm.TAU_RATIO)
    # P1 (P+): psi(3770) K+ takes its c from the per-channel products S041.361 / S041.362 (the total
    # S041.360 is excluded), inverse-variance mean computed here by hand, with 65 % added (LHCb)
    r = row(521, 'psi(3770) K+')
    c1, s1 = 1.5464e-4 / (4.8808e-4 * 0.52), 4.8937e-5 / (4.8808e-4 * 0.52)
    c2, s2 = 9.3926e-5 / (4.8808e-4 * 0.41), 3.4718e-5 / (4.8808e-4 * 0.41)
    w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
    c = (w1 * c1 + w2 * c2) / (w1 + w2)
    assert r['how'] == 'product' and r['pdgids'] == 'S041.361;S041.362' and close(r['c'], c, 2e-3)
    assert close(r['sigma_c'], math.hypot(1 / math.sqrt(w1 + w2), 0.65 * c), 2e-3) and 0.35 < r['sigma_c'] < 0.39
    assert {'measurement_conflict', 'exception_applied'} <= flags(r)
    assert NODES.set_index('uid').loc['S041.360', 'role'] == 'excluded'
    k0 = row(511, 'psi(3770) K_S0')
    assert 'measurement_conflict' in flags(k0) and 0.45 < k0['c'] < 0.55 and 0.3 < k0['sigma_c'] < 0.4
    d = row(521, 'D+ D- K+')
    assert d['how'] == 'node' and 0.38 < d['c'] < 0.48 and 0.4 < d['sigma_c'] < 0.55
    # own limits are not exceeded by a remainder
    for names, pid in (('anti-D_2*0 rho+', 'S041.213'), ('K*+ chi_c0', 'S041.435')):
        r = row(521, names)
        assert r['pdg_class'] == 'limit' and r['c'] == 1.0 and r['pdgids'] == pid, names
        assert ('limit_capped' in flags(r)) == ('above_limit' in flags(r)), names
    # B+ -> anti-D_10 rho+ (G3): 2.7 times its limit S041.215 (1.4e-3); half_limit exception
    r = row(521, 'anti-D_10 rho+')
    assert r['how'] == 'half_limit' and r['pdgids'] == 'S041.215' and close(r['c'], 0.5 * 0.0014 / 0.00372)
    assert close(r['sigma_c'], r['c']) and {'above_limit', 'exception_applied'} <= flags(r)   # the file is above
    # P4: S041.799 (LHCb 2023) excluded; the line has no other entry (was c = 6.8)
    r = row(521, 'chi_c0 K0 pi+')
    assert r['how'] == '' and r['pdg_class'] == 'none' and r['c'] == 1.0 and 'exception_applied' in flags(r)
    # exceptions: D*+- D-+ is a charge sum; psi(2S) K_1(1270) S041.858 is one partial wave
    for m, names in ((511, 'D*+ D-'), (511, 'D*- D+'), (-511, 'D*- D+'), (-511, 'D*+ D-')):
        r = row(m, names)
        assert r['how'] == 'node' and r['pdgids'] == 'S042.511' and 0.95 < r['c'] < 1.05, names
        assert 'exception_applied' in flags(r)
    # psi(2S) K_1+ reaches the psi(2S) K+ pi+ pi- total through the K_1: unmeasured; the direct
    # line takes the total minus the K_1 part (1.43e-4 +- 100 %)
    r = row(521, 'psi(2S) K_1+')
    assert r['how'] == '' and r['c'] == 1.0 and close(r['sigma_c'], 1.0)
    assert {'unmeasured', 'exception_applied'} <= flags(r)
    r = row(521, 'psi(2S) K+ pi+ pi-')
    assert r['how'] == 'node' and r['pdgids'] == 'S041.144' and 0.6 < r['c'] < 0.75 and 0.3 < r['sigma_c'] < 0.4
    assert abs(NODES.set_index('uid').loc['S041.144', 'closure_pull']) < 1e-6
    # repeated product measurements are used once
    r = row(511, "D'_s1+ D*-")
    assert r['pdgids'] == 'S042.490' and r['n_prod'] == 1 and 0.55 < r['c'] < 0.65
    assert set(n.loc[['S042.272', 'S042.392', 'S042.366'], 'role']) == {'duplicate'}
    assert row(511, "D'_s1+ D-")['pdgids'] == 'S042.489'
    assert row(521, "anti-D'_10 pi+")['n_prod'] == 1
    # flags for the analyst
    assert row(511, 'D*+ K0 pi-')['pdgids'] == 'S042.314'      # no longer a twin: part of the sum
    r = row(511, "D'_1- pi+")                                  # was c = 79 from a remainder
    assert r['how'] == '' and r['c'] == 1.0 and 'unmeasured' in flags(r) and 'node_ambiguous' not in flags(r)
    r = row(521, "anti-D'_10 rho+")                            # was c = 17
    assert r['how'] == '' and r['c'] == 1.0 and 'unmeasured' in flags(r)
    # G6: B+ -> anti-D*0 K+ K*0 (line 4349, Delta S = 2, Belle 2002's value with the K*0 not flipped)
    # is vetoed; its B- conjugate (5888) inherits c = 0. The allowed B- line (5892, 1.5e-3) takes the
    # corrected BF of the allowed B+ line (4345, 5.9e-4): c x 5.9e-4 / 1.5e-3
    v = row(521, 'anti-D*0 K+ K*0')
    assert v['how'] == 'veto' and v['c'] == 0.0 and v['sigma_c'] == 0.0 and 'c_extreme' not in flags(v)
    assert row(-521, 'D*0 K- anti-K*0')['c'] == 0.0
    a, b = row(521, 'anti-D*0 K+ anti-K*0'), row(-521, 'D*0 K- K*0')
    assert {'block_asymmetric', 'cp_transfer'} <= flags(b) and close(b['c'], a['c'] * 0.00059 / 0.0015)
    assert close(b['sigma_c'], a['sigma_c'] * 0.00059 / 0.0015) and 0.15 < b['c'] < 0.25
    # G6: S042.314 is the sum of both charge combinations (BaBar 2005, K_S): D*- K0 pi+ and D*+ K0 pi-
    # share the remainder after D*- K*+ (exact)
    n = NODES.set_index('uid').loc['S042.314']
    x, y, kst = row(511, 'D*- K0 pi+'), row(511, 'D*+ K0 pi-'), row(511, 'D*- K*+')
    fk = {k: v for k, v, _ in n['feeders']}
    sub = kst['c'] * fk[pm.collapse_key(code('D*- K*+'))]
    assert x['how'] == y['how'] == 'node' and close(x['c'], y['c'])
    assert close(x['c'], (n['value'] - sub) / (0.0003 + 0.0003))
    assert abs(n['closure_pull']) < 1e-6
    assert 'block_asymmetric' not in flags(row(-511, 'J/psi anti-K*0'))


def test_isospin():
    # u <-> d partners
    for a, b in (('D-', 'anti-D0'), ('D*-', 'anti-D*0'), ('D_0*-', 'anti-D_0*0'), ('D_2*+', 'D_2*0'),
                 ('K+', 'K0'), ('K-', 'anti-K0'), ('K*+', 'K*0'), ('K_1+', 'K_10'), ('pi+', 'pi-'),
                 ('rho+', 'rho-'), ('a_1+', 'a_1-')):
        ca, cb = PDL.name_to_pdg[a], PDL.name_to_pdg[b]
        assert pm.isospin_partner_code(ca) == cb and pm.isospin_partner_code(cb) == ca, (a, b)
    for a in ('pi0', 'eta', 'rho0', 'omega', 'phi', 'J/psi', 'psi(2S)', 'D_s+', 'D_s0*+', 'e+', 'nu_e', 'gamma'):
        assert pm.isospin_partner_code(PDL.name_to_pdg[a]) == PDL.name_to_pdg[a], a
    assert pm.isospin_partner_code(PDL.name_to_pdg['p+']) is None
    k = pm.collapse_key(code('D_0*- e+ nu_e'))
    assert pm.isospin_partner_keys(k, PDL, 3) == [pm.collapse_key(code('anti-D_0*0 e+ nu_e'))]
    # the collapsed K0 of a B0 line is a K+ in the B+ partner (charge decides)
    assert pm.isospin_partner_keys(pm.collapse_key(code('J/psi K_S0')), PDL, 3) == [pm.collapse_key(code('J/psi K+'))]
    assert pm.isospin_partner_keys(pm.collapse_key(code('J/psi K+ pi-')), PDL, 3) == \
        [pm.collapse_key(code('J/psi K0 pi+'))]
    # the report: pairs of the isospin-conserving transitions only
    iso = pm.isospin_report(TABLE, PDL)
    assert set(iso['kind']) == {'sl', 'ccs'} and set(iso['status']) == {'both', 'only_b0', 'only_bp', 'neither'}
    r = iso[iso['names_b0'] == 'D- e+ nu_e'].iloc[0]
    assert r['names_bp'] == 'anti-D0 e+ nu_e' and r['status'] == 'both' and abs(r['r_mc'] - 1) < 0.01
    assert abs(r['pull']) < 2 and 0.9 < r['r_pdg'] < 1.1
    assert not (iso['names_b0'] == 'D- pi+').any() and not (iso['names_b0'] == 'pi- e+ nu_e').any()
    both = iso[iso['status'] == 'both']
    assert len(both) > 40 and (both['pull'].abs() > 3).sum() == 0
    # the lines fixed by the isospin exception are not counted as measured twice
    r = iso[iso['names_b0'] == 'D_0*- e+ nu_e'].iloc[0]
    assert r['status'] == 'only_bp' and r['how_b0'] == 'isospin' and close(r['c_b0'], r['c_other'])
    # toy: direct_line makes a resonance line the direct line of a node; extra_uncertainty widens a c
    exc = [dict(pdgid='T.4', action='direct_line', mother=511, key=code('J/psi K_2*0'), reason='test')]
    pdg = [x if x[0] != 'T.4' else ('T.4', x[1], 0.0020, 0.0001, False) for x in TOY_PDG]
    td, _, lkd = toy(exc, pdg)
    c = (0.0020 - 1.2 * (0.0006 * 0.6667 + 0.0004)) / (0.0004 + 0.00025)   # shared by the two direct lines
    for names in ('J/psi K+ pi-', 'J/psi K_2*0'):
        r = row(511, names, td, lkd)
        assert r['how'] == 'node' and r['pdgids'] == 'T.4' and close(r['c'], c), names
    assert 'exception_applied' in flags(row(511, 'J/psi K_2*0', td, lkd))
    exc = [dict(mother=511, key=code('D_s0*+ D-'), action='extra_uncertainty', relative=0.5, reason='test')]
    te, _, lke = toy(exc)
    r0, r1 = row(511, 'D_s0*+ D-', *toy()[::2]), row(511, 'D_s0*+ D-', te, lke)
    assert close(r1['c'], r0['c']) and close(r1['sigma_c'], math.hypot(r0['sigma_c'], 0.5 * r0['c']))
    # toy: the B0 line D_1- pi+ takes the c of a B+ line; a source without an own c is refused
    t0, _, lk0 = toy()
    src = row(511, 'D_s0*+ D-', t0, lk0)
    assert src['how'] == 'product'
    for bad in (dict(mother=511, key=code('D- rho+'), action='isospin', from_mother=521,
                     from_key=code('anti-D0 rho+'), reason='test'),):
        try:
            toy([bad])
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError('accepted an isospin source without an own c')


def test_node_columns_and_report():
    # decay-file line numbers, fed totals and their summary text (point 2 of 2026-11-17)
    r = row(511, 'D_1- e+ nu_e')
    assert r['dec_lines'] == '368'
    r = row(511, 'K*0 psi(3770)')
    assert r['dec_lines'] == '597 598 599 600' or set(r['dec_lines'].split()) >= {'597'}
    assert 'S042.609' in r['fed_nodes'].split(';') and 'S042.609' in r['unclosed_nodes'].split(';')
    info = [x for x in r['node_info'].split(';') if x.startswith('S042.609')][0]
    assert 'PDG 0.000351' in info and 'sigma' in info
    fed = TABLE[[('S042.609' in str(f).split(';')) for f in TABLE['fed_nodes']]]
    assert len(fed) >= 3 and (fed['node_info'].str.contains('S042.609')).all()
    # every unclosed node is also a fed node
    for f, u in zip(TABLE['fed_nodes'], TABLE['unclosed_nodes']):
        assert set(filter(None, str(u).split(';'))) <= set(filter(None, str(f).split(';')))
    rep = pm.node_report(NODES, TABLE, PDL)
    x = rep.set_index('uid').loc['S042.609']
    assert close(x['value'], 3.50740e-4, 1e-4) and x['n_feeders'] == 3 and '597' in x['feeders']
    assert abs(x['closure_pull']) > 3 and set(rep['mother']) == {511, 521}
    assert rep['uid'].is_unique


def test_review_fixes_nov17():
    # node report: a K0 final state lists its K_S0 / K_L0 lines with BF > 0, not the zero-BF K0 line
    rep = pm.node_report(NODES, TABLE, PDL).set_index('uid')
    item = [x for x in rep.loc['S042.264', 'feeders'].split(' | ') if x.startswith('K0 psi(3770)')][0]
    assert '[593 594]' in item.replace(',', ' ') or ('593' in item and '594' in item and '592' not in item), item
    # the isospin target carries the extra relative uncertainty of its source
    assert close(row(511, 'psi(3770) K_S0')['extra_rel'], 0.65) and close(row(521, 'psi(3770) K+')['extra_rel'], 0.65)
    # a forced line keeps its class and enters the remainder with 1 +- 1, even with an own limit
    exc = [dict(mother=511, key=code('D_0*- e+ nu_e'), action='force_class', reason='test', **{'class': 'none'})]
    t, _, lk = toy(exc, TOY2_PDG, TOY2_DEC)
    r = row(511, 'D_0*- e+ nu_e', t, lk)
    assert r['pdg_class'] == 'none' and r['c'] == 1.0 and close(r['sigma_c'], 1.0) and 'limit_capped' not in flags(r)
    d = row(511, 'anti-D0 pi- e+ nu_e', t, lk)
    # in TOY2 the D_0*- line feeds anti-D0 pi- l nu with BF x P = 0.002 x 0.5 = 0.001 (c = 1)
    assert d["how"] == "node" and close(d["c"], (0.003 - 0.001) / 0.001)
    # an extra uncertainty on a line that takes its c from a node is applied before larger nodes
    # subtract it: the 4-body direct line of the toy sees the widened sigma
    base, nb, lkb = toy()
    exc = [dict(mother=511, key=code('D*- rho0 pi+'), action='extra_uncertainty', relative=1.0, reason='test')]
    te, ne, lke = toy(exc)
    r3b, r3 = row(511, 'D*- rho0 pi+', base, lkb), row(511, 'D*- rho0 pi+', te, lke)
    assert close(r3['sigma_c'], math.hypot(r3b['sigma_c'], 1.0 * r3b['c']))
    r4b, r4 = row(511, 'D*- pi+ pi+ pi-', base, lkb), row(511, 'D*- pi+ pi+ pi-', te, lke)
    sub4 = 1.2 * 0.010 + r3b['c'] * 0.002
    s4 = math.hypot(math.sqrt(0.0002 ** 2 + (0.05 * 0.010) ** 2 + (r3['sigma_c'] * 0.002) ** 2),
                    pm.F_INTERFERENCE * sub4) / 0.001
    assert close(r4['c'], r4b['c']) and close(r4['sigma_c'], s4) and r4['sigma_c'] > r4b['sigma_c']


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
    # the committed table is the one the code gives: a message, not a test (rerun main() to
    # refresh it); the behaviour is tested above
    if os.path.exists(pm.DEFAULT_TABLE_CSV):
        disk = pm.read_table(pm.DEFAULT_TABLE_CSV)
        same = (list(disk['key']) == list(TABLE['key']) and list(disk['how']) == list(TABLE['how'])
                and np.allclose(disk['c'], TABLE['c'], rtol=1e-5))
        if not same:
            print('    note: b_mode_pdg_table.csv is stale; run python3 scripts/tools/pdg_match.py')


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
    # the top-95 % lists count hadronic modes only
    top = review[review['n_top95'] > 0]
    sl = {(m, k) for m, k, s in zip(TABLE['mother_pdg'], TABLE['key'], TABLE['is_sl']) if s}
    assert not any((m, k) in sl for m, k in zip(top['mother_pdg'], top['key']))
    assert (review['max_share'] > 0.1).any()
    r = review[(review['mother_pdg'] == 511) & (review['names'] == 'D_s0*+ D-')].iloc[0]
    assert r['how'] == 'product' and r['max_share'] > 0.02 and r['n_top95'] > 10
    # one file is enough to run it
    rev1, sum1 = pm.review_report(TABLE, [os.path.join(pm.CSV_DIR, 'bbbar_modes_run1_e_combCR.csv')], MODES)
    assert set(sum1['region']) == {'combCR'} and len(sum1) == 3



def test_nonres_external_and_windows_nov18():
    """P2 (nonres_direct), P3 and P4 (exclusions), P5 (external); review of 2026-11-18."""
    # toy: a measured non-resonant entry sets the direct line; the total only closes
    pdg = TOY_PDG + [('T.40', 'B0 --> J/psi(1S) K+ pi- nonresonant', 0.0002, 0.00005, False)]
    base, _, lkb = toy(pdg=pdg)
    assert row(511, 'J/psi K+ pi-', base, lkb)['how'] == 'node'          # without the exception
    exc = [dict(pdgid='T.40', action='nonres_direct', mother=511, key=code('J/psi K+ pi-'), reason='test')]
    t, nodes, lk = toy(exc, pdg)
    r = row(511, 'J/psi K+ pi-', t, lk)
    assert r['how'] == 'nonres' and r['pdg_class'] == 'measured' and r['pdgids'] == 'T.40'
    assert close(r['c'], 0.5) and close(r['sigma_c'], 0.125) and {'nonres_source', 'exception_applied'} <= flags(r)
    roles = nodes.set_index('uid')['role']
    assert roles['T.40'] == 'nonres_direct' and roles['T.4'] == 'covered'
    tot = nodes.set_index('uid').loc['T.4']
    assert np.isfinite(tot['closure_pull'])                               # the total is still checked
    assert pm.nonres_final_state('B0 --> (K+ pi- pi0) nonresonant', PDL) == pm.collapse_key([321, -211, 111])
    assert pm.nonres_final_state('B0 --> Lambdabar_c()- p pi+ pi- (nonresonant)', PDL) == \
        pm.collapse_key([-4122, 2212, 211, -211])
    try:                                                                  # wrong final state
        toy([dict(pdgid='T.40', action='nonres_direct', mother=511, key=code('J/psi K_2*0'), reason='test')], pdg)
        raise AssertionError('nonres_direct on the wrong final state was accepted')
    except ValueError:
        pass
    # toy: external value for a line without an entry of its own
    exc = [dict(mother=511, key=code('J/psi K_2*0'), action='external', value=0.0004, error=0.0001,
                source='test note', reason='test')]
    t, _, lk = toy(exc)
    r = row(511, 'J/psi K_2*0', t, lk)
    assert r['how'] == 'external' and r['pdg_class'] == 'measured' and r['pdgids'] == 'external:test note'
    assert close(r['c'], 0.8) and close(r['sigma_c'], 0.2) and {'external', 'exception_applied'} <= flags(r)
    try:                                                                  # a PDG entry exists: refused
        toy([dict(mother=511, key=code('D- rho+'), action='external', value=0.008, error=0.001,
                  source='test', reason='test')])
        raise AssertionError('external on a final state with a PDG entry was accepted')
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:                            # loader checks
        for bad in (dict(mother=511, key=[1], action='external', value=1e-4, error=1e-5, reason='x'),
                    dict(pdgid='S1', action='nonres_direct', reason='x')):
            path = os.path.join(tmp, 'e.json')
            with open(path, 'w') as f:
                json.dump([bad], f)
            try:
                pm.load_exceptions(path)
                raise AssertionError(f'loader accepted {bad}')
            except ValueError:
                pass

    # real table, P2: the non-resonant entries that set their direct line
    for m, names, pid, c in ((511, 'anti-Lambda_c- p+ pi+ pi-', 'S042.360', 0.530),
                             (511, 'omega K+ pi-', 'S042.446', 1.0), (511, 'K+ K- K0', 'S042.506', 1.0),
                             (521, 'K+ K- K+', 'S041.231', 0.998), (521, 'K+ mu+ mu-', 'S041.704', 0.728)):
        r = row(m, names)
        assert r['how'] == 'nonres' and r['pdgids'] == pid and abs(r['c'] - c) < 0.005, names
        assert 'nonres_source' in flags(r), names
    r = row(511, 'anti-Lambda_c- p+ pi+ pi-')
    assert abs(r['sigma_c'] - 0.096) < 0.002
    # P2: the lines that stand for components without lines of their own keep their remainder
    for m, names in ((521, 'D+ D- K+'), (521, 'D*+ D- K+'), (521, 'D+ D*- K+'), (511, 'eta_c K+ pi-')):
        assert row(m, names)['how'] == 'node', names
    # P2: S041.579 is a product cross-check of B+ -> anti-D_10 pi+ only; S041.443 (same sub-decay,
    # includes D_1 -> D*- pi+ in its LHCb input) is not averaged with it
    r = row(521, 'anti-D_10 pi+')
    assert r['how'] == 'exact' and r['pdgids'] == 'S041.214' and 0.75 < r['c'] < 0.81
    assert 'S041.579' in r['prod_pdgids'].split(';') and 'S041.443' not in r['prod_pdgids'].split(';')
    roles = NODES.set_index('uid')['role']
    assert roles['S041.443'] == 'duplicate'
    # P3 and P4: window-restricted and LHCb 2023 entries excluded
    for pid in ('S042.461', 'S042.231', 'S041.307', 'S041.308', 'S041.474', 'S042.310', 'S042.376',
                'S041.356', 'S041.539', 'S041.744', 'S042.32', 'S042.571', 'S041.797', 'S041.798', 'S041.799'):
        assert roles[pid] == 'excluded', pid
    r = row(511, 'rho0 K+ pi-')
    assert r['how'] == '' and r['pdg_class'] == 'none' and r['c'] == 1.0
    r = row(521, 'K+ K- K0 pi+')
    assert r['how'] == '' and r['c'] == 1.0 and 'c_extreme' not in flags(r)
    assert 'node_not_closed' not in flags(row(511, 'K*0 gamma'))
    # P5: Belle + Belle II D(*) eta pi measurement, outside PDG; conjugates follow
    for m, names, c, s in ((511, 'D*- eta pi+', 2.78 / 2.56, 0.336 / 2.56), (-511, 'D*+ eta pi-', 2.78 / 2.56, 0.336 / 2.56),
                           (521, 'anti-D0 eta pi+', 5.70 / 4.00, 0.585 / 4.00), (-521, 'D0 eta pi-', 5.70 / 4.00, 0.585 / 4.00)):
        r = row(m, names)
        assert r['how'] == 'external' and abs(r['c'] - c) < 1e-3 and abs(r['sigma_c'] - s) < 1e-3, names
        assert r['pdgids'] == 'external:BELLE2-PUB-PH-2025-024 v12', names
    for m, names in ((511, 'D- eta pi+'), (521, 'anti-D*0 eta pi+')):
        r = row(m, names)
        assert r['how'] == '' and r['c'] == 1.0, names


def test_narrow_folds_p6():
    """P6: narrow resonances folded into the totals whose measurements include them (2026-11-18)."""
    # D_s1(2536)+ -> D* K is inside the BaBar 2011 D(*)D(*)K totals: the direct lines take the remainder
    # c = (total - c(D_s1 line) x BF x 0.5) / BF
    ds1 = {511: {"D'_s1+ D-": 0.00043, "D'_s1+ D*-": 0.00083}, 521: {"anti-D0 D'_s1+": 0.00045, "anti-D*0 D'_s1+": 0.00094}}
    for m, names, total, s1 in ((511, 'D*- D*+ K0', 8.10924e-3, "D'_s1+ D*-"), (511, 'D*- D*0 K+', 0.0106, "D'_s1+ D*-"),
                                (511, 'D- D*0 K+', 3.46e-3, "D'_s1+ D-"), (521, 'anti-D0 D*+ K0', 3.81e-3, "anti-D0 D'_s1+"),
                                (521, 'anti-D*0 D*+ K0', 9.17e-3, "anti-D*0 D'_s1+")):
        r, rs = row(m, names), row(m, s1)
        assert r['how'] == 'node', names
        expect = (total - rs['c'] * ds1[m][s1] * 0.5) / r['bf_mc']
        assert abs(r['c'] - expect) < 2e-3 and r['c'] < 0.98, (names, r['c'], expect)
    # the two direct lines of the sum S042.265 share one c
    assert close(row(511, 'D*- D+ K0')['c'], row(511, 'D- D*+ K0')['c']) and row(511, 'D- D*+ K0')['c'] < 0.97
    # Sigma_c(2455) inside Lambdabar_c p pi pi: the total (a closure check since P2) now closes
    rep = pm.node_report(NODES, TABLE, PDL).set_index('uid')
    assert abs(rep.loc['S042.161', 'closure_pull']) < 1.0
    # psi(2S) inside the CDF J/psi K*0 pi pi total
    assert abs(row(511, 'J/psi K*0 pi+ pi-')['c'] - 0.662) < 0.01
    # fold only acts on the named entries: the D* veto of D0bar pi pi stays (no D* folded)
    assert row(511, 'anti-D0 pi- pi+')['how'] == 'node' and abs(row(511, 'anti-D0 pi- pi+')['c'] - 0.279) < 0.01


def test_wrong_flavor_p7():
    """P7: B0 lines written with the wrong charm flavor are vetoed; the PDG texts get their bar (2026-11-19)."""
    for m, names in ((511, 'D*0 pi+ pi+ pi- pi-'), (-511, 'anti-D*0 pi- pi- pi+ pi+'),
                     (511, 'anti-Lambda0 Lambda0 D0'), (-511, 'Lambda0 anti-Lambda0 anti-D0')):
        r = row(m, names)
        assert r['how'] == 'veto' and r['c'] == 0.0 and r['pdg_class'] == 'none', names
    nodes = {n['pdgid']: n for n in pm.make_nodes(PDG, PDL, pm.load_exceptions()) if n['pdgid'] in ('S042.195', 'S042.457')}
    assert 'anti-D*0' in nodes['S042.195']['names'][0] and 'anti-D0' in nodes['S042.457']['names'][0]


def test_n1_convention_33():
    """Point 3.3 (N1, 2026-11-20): every hadronic line without a c gets c = 1 +- 1; a limit line above its
    own limit gets c = limit/2 +- limit/2 (how half_limit, flag limit_auto)."""
    x = TABLE[TABLE['mother_pdg'].isin([511, 521, -511, -521]) & (TABLE['bf_mc'] > 0)]
    noc = x[x['pdg_class'].isin(['none', 'limit']) & (x['how'] == '') & ~x['is_sl']]
    assert len(noc) > 1000 and (noc['c'] == 1.0).all() and (noc['sigma_c'] == 1.0).all()
    auto = x[x['flags'].str.contains('limit_auto')]
    assert len(auto) >= 8 and (auto['how'] == 'half_limit').all()
    assert np.allclose(auto['c'], auto['sigma_c']) and (auto['c'] < 0.5).all()
    r = row(511, 'eta_c(2S) K*0')
    assert r['how'] == 'half_limit' and 'limit_auto' in flags(r) and abs(r['c'] - 0.295) < 0.01
    # unchanged: vetoes, the half_limit by exception, measured lines
    assert row(521, 'anti-D*0 K+ K*0')['c'] == 0.0
    assert 'limit_auto' not in flags(row(521, 'anti-D_10 rho+'))


def test_lhcb_ds_lines_34():
    """Point 3.4 (2026-11-20): the D** D_s(*) and three-body D_s(*) D pi lines take LHCb-based external c.

    Three-body values corrected on 2026-11-25 (piece 4, 3.2 check 1: non-D2* part 0.778, not 0.72).
    """
    exp = {(521, 'anti-D_2*0 D_s+'): 0.148, (521, 'D_s+ D- pi+'): 0.241, (521, 'anti-D_10 D_s+'): 2.058,
           (511, 'D_2*- D_s+'): 0.136, (511, 'D_s*+ D- pi0'): 0.202, (-521, 'D_2*0 D_s-'): 0.148}
    for (m, names), c in exp.items():
        r = row(m, names)
        assert r['how'] == 'external' and r['pdg_class'] == 'measured' and abs(r['c'] - c) < 0.01, names
        assert r['pdgids'].startswith('external:LHCb'), names
    # the D pi pi lines stay unmeasured (N2 class)
    r = row(521, 'D_s+ D- pi+ pi0')
    assert r['pdg_class'] == 'none' and r['c'] == 1.0
    n = sum(1 for e in pm.load_exceptions() if e['action'] == 'external' and 'LHCb' in e.get('source', ''))
    assert n == 20


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')


if __name__ == '__main__':
    main()
