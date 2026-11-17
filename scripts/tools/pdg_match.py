"""Comparison of the exclusive B-decay lines of the decay file with PDG (step B0, piece 2).

What it does
------------
1. `build_pdg_table` dumps every branching-fraction node of B0 and B+ from the `pdg` package to one
   JSON file (run once per PDG edition). Everything else reads only that file (`load_pdg_table`).
2. `parse_description` turns the PDG text of a node into a node kind:
     plain   : a set of daughters, as signed PDG codes of evt.pdl;
     product : B -> X Y with one or more sub-decays X -> ... (a product of branching fractions);
     sum     : A + B;
     nonres  : explicitly non-resonant (stored, not used);
     unkeyed : generic, inclusive, charge-summed or unknown names. Nothing is guessed.
3. `DecayGraph` follows the decay file's own sub-decay tables: BF_MC(node) is the sum over the
   exclusive B lines of BF_line x P(line -> final state of the node).
4. `build_mode_pdg_table` gives every exclusive final state of the four B blocks a class, the PDG
   value, c = BF_PDG / BF_MC and its uncertainty.
5. `review_report` lists the modes to check by hand, with their yield shares in the samples.

Rules
-----
* Unit of comparison: the PDG node, not the decay-file line.
* Folding: a daughter of a line is replaced by its decays if it is BROAD (evt.pdl width >= 10 MeV
  and lifetime 0) and is not named in the node itself. Narrow states (D*, D_s*, D_s0*, J/psi, eta,
  omega, phi, K_S0, ...) are never folded by default, nor are eta_c, eta_c(2S) and chi_c0
  (`NO_FOLD_DEFAULT`). At most `MAX_DEPTH` nested levels.
* Neutral kaons: K0, anti-K0, K_S0 and K_L0 are one symbol. A PDG node with one K_S0 (or K_L0) is
  multiplied by 2; nodes with two or more are flagged `multi_ks` and give no c.
* PDG "lepton" nodes are compared with the e and with the mu final state, each with the full value.
* Priority for the c of a final state:
    1. exact   : a measured plain node equal to the final state, and fed by no other line.
    2. product : c = PDG product / (BF_line x BF_sub of the decay file); several product nodes
                 are combined with inverse-variance weights (the chi2 is kept).
    3. node    : only for a DIRECT line of a measured multi-body node, i.e. a line whose final
                 state is the node's own (in the decay file the PHSP line meant as the non-resonant
                 part). The node is the inclusive rate of that final state; the non-resonant part is
                 never measured, only a remainder: c = (node - other lines) / BF of the direct
                 line, clamped at 0 (`remainder_clamped`). Lines with a c are subtracted with c and
                 sigma_c, lines without one with c = 1 +- 1, lines that the decay file puts above
                 their own PDG limit as limit/2 +- limit/2 (`limit_capped`). The total is not exactly
                 the sum of its parts (interference): F_INTERFERENCE (20 %) of the subtracted part,
                 divided by the BF of the direct line, is added in quadrature to sigma_c.
                 `remainder_unstable`: sigma_c > c. Nodes are taken from the fewest bodies up (then
                 the smaller relative PDG error).
    A line that reaches a measured node only through a resonance never takes a c from it: the
    node measures a sum, not this line (decision M1 of docs/B0_PIECE2_REVIEW.md). Without an
    exact or product entry of its own such a line is flagged `unmeasured`: class none, c = 1,
    sigma_c = `UNMEASURED_SIGMA` (100 %).
    The product result is always stored (`c_prod`), whatever was used for c.
    Product nodes of one line that repeat one measurement are dropped before the mean: of nodes
    with the same sub-decays the most precise is kept, and a node whose sub-decay alternatives are
    a part of another node's (the components of a PDG sum node) is dropped for the sum node.
* Checks: `above_limit` compares max(1, c) x BF_MC with every own limit of a line. After all c are
  set, every measured final-state node is checked for closure (sum of c x BF x P against
  BF_PDG); |pull| > 3 sets `node_not_closed` on its feeding lines (column `unclosed_nodes`).
  `node_ambiguous`: the line was free in another measured node with as many bodies as the node
  that gave its c (column `alt_pdgids`). `flavor_twin`: a 'none' line equals a measured or limit
  final state of its block once the K0 / K*0 flavor is ignored or the state is conjugated
  (column `twin_pdgids`); c is not changed.
* Classes: measured, limit (only an upper limit; c = 1), none (no PDG node; c = 1), inclusive
  (Xu, Xsd, Xsu lines; c = 1).
* The B0 and B+ blocks are matched; anti-B0 and B- final states take the result of their
  charge conjugate. Where the conjugate is not listed, the neutral-kaon and K*0 flavor is ignored
  and `block_asymmetric` is set. Where it is listed with another BF (the blocks are not mirrors),
  `block_asymmetric` is set too, and a c from an exact, plain-node or product match is not carried
  over (c = 1, how = '').

Exceptions file (`Samples/dec/pdg_exceptions.json`): a JSON list of entries
    {"pdgid": "S042.1", "action": "exclude_node", "reason": "..."}
    {"pdgid": "S042.1", "action": "fold", "states": ["D*-"], "reason": "..."}
    {"pdgid": "S042.1", "action": "no_fold", "states": ["D_2*-"], "reason": "..."}
    {"mother": 511, "key": [-411, 211], "action": "force_class", "class": "none", "reason": "..."}
    {"pdgid": "S042.124", "action": "rewrite", "description": "B0 --> ...", "reason": "..."}
    {"mother": 511, "key": [...], "action": "isospin", "from_mother": 521, "from_key": [...], "reason": "..."}
`states` are evt.pdl names. `rewrite` replaces the PDG text of a node before it is parsed (for
misprints in the listing); the value is not changed. `force_class` to anything but 'measured' also resets c to 1.
`isospin` gives a B0 (B+) line the corrected BF of its isospin partner in the other block, which
must have an exact or product c: BF(B+) = TAU_RATIO x BF(B0), so c = c_partner x BF_MC(partner) /
BF_MC(line) x (1 / TAU_RATIO for a B0 line, TAU_RATIO for a B+ line) (how = 'isospin', flag `isospin`).
`direct_line` declares a line as the direct line of a multi-body node although it reaches it through
a resonance (the decay file routes the whole final state through that resonance):
    {"pdgid": "S042.88", "action": "direct_line", "mother": 511, "key": [...], "reason": "..."}
`half_limit` gives a line that the decay file puts above its own PDG upper limit c = limit/2 with
sigma_c = limit/2 (how = 'half_limit', class limit):
    {"mother": 521, "key": [...], "action": "half_limit", "reason": "..."}
`force_class` is applied before the totals are split: a forced line takes no c from a total.
`extra_uncertainty` adds a relative uncertainty in quadrature to the sigma_c of a line with a c:
    {"mother": 511, "key": [...], "action": "extra_uncertainty", "relative": 0.2, "reason": "..."}
    (optional "flag": a flag added to the line, e.g. "measurement_conflict")

Analysis-time code needs numpy and pandas only. The `pdg` package is imported inside
`build_pdg_table` only.

Run  python3 scripts/tools/pdg_match.py  to write the mode table and the review table.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decay_table as dt  # noqa: E402

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
_DEC_DIR = os.path.join(_ROOT, 'Samples', 'dec')
DEFAULT_PDG_JSON = os.path.join(_DEC_DIR, 'pdg_bf_B_2026.json')
DEFAULT_EXCEPTIONS = os.path.join(_DEC_DIR, 'pdg_exceptions.json')
CSV_DIR = os.path.join(_ROOT, 'Notebooks', 'Plotting', 'csv_output')
DEFAULT_TABLE_CSV = os.path.join(CSV_DIR, 'b_mode_pdg_table.csv')
DEFAULT_REVIEW_CSV = os.path.join(CSV_DIR, 'b_mode_pdg_review.csv')
DEFAULT_ISOSPIN_CSV = os.path.join(CSV_DIR, 'b_mode_isospin_check.csv')
DEFAULT_NODES_CSV = os.path.join(CSV_DIR, 'b_mode_pdg_nodes.csv')

PDG_PARENTS = {'B0': ('S042', 511), 'B+': ('S041', 521)}

BROAD_WIDTH = 0.010        # GeV: resonances at least this wide are folded by default
MAX_DEPTH = 4              # nested folding levels
# Charmonium below the open-charm threshold is wider than BROAD_WIDTH but is not folded: the
# charmless PDG nodes exclude it, and its own decays are quoted as products.
NO_FOLD_DEFAULT = ('eta_c', 'eta_c(2S)', 'chi_c0')
P_MIN = 0.10               # a line takes its c from a node only if this share of it feeds the node
UNMEASURED_SIGMA = 1.0      # sigma_c of a line flagged `unmeasured` (c = 1)
F_INTERFERENCE = 0.2        # direct line: extra uncertainty, this fraction of what was subtracted
REL_TOL = 1e-6              # relative tolerance when a BF is compared with a limit
RES_MISMATCH = 0.30        # f_res_PDG - f_res_MC above this sets `resonance_mismatch`
C_EXTREME = 3.0            # c outside [1/3, 3] sets `c_extreme`
PULL_REVIEW = 3.0
YIELD_COVER = 0.95
TAU_RATIO = 1.076          # tau(B+) / tau(B0), PDG: 1.076 +- 0.004

KSYM = 311                 # symbol of the neutral kaon after collapsing K0, anti-K0, K_S0, K_L0
_K0_CODES = (311, -311, 310, 130)
_OPAQUE = 0                # daughters of a quark-level (PYTHIA) sub-decay: matches no node

# --------------------------------------------------------------------------------------------
# PDG names -> evt.pdl names
# --------------------------------------------------------------------------------------------
# Only names listed here are keyed. A name with an explicit mass that is missing stays unkeyed.
# Names with empty parentheses are the PDG spelling of a state whose name is unique without the
# mass (D_s()+, D^*()-, b_1()0); those that are ambiguous (f_0(), psi(), rho()0) are not listed and
# are resolved only as the mother of a sub-decay, from the first part of the description.
_L_PLUS, _L_MINUS, _NU_L = '@l+', '@l-', '@nu'

PDG_NAMES = {
    # leptons and photon
    'e+': 'e+', 'e-': 'e-', 'mu+': 'mu+', 'mu-': 'mu-', 'tau+': 'tau+', 'tau-': 'tau-',
    'nu_e': 'nu_e', 'nu_mu': 'nu_mu', 'nu_tau': 'nu_tau', 'gamma': 'gamma',
    'lepton+': _L_PLUS, 'lepton-': _L_MINUS, 'nu_lepton': _NU_L,
    # light unflavored mesons
    'pi+': 'pi+', 'pi-': 'pi-', 'pi0': 'pi0', 'eta': 'eta', "eta^'": "eta'", "eta^'(958)": "eta'",
    'rho0': 'rho0', 'rho+': 'rho+', 'rho-': 'rho-', 'rho(770)0': 'rho0', 'rho(770)+': 'rho+',
    'omega': 'omega', 'omega(782)': 'omega', 'phi': 'phi', 'phi(1020)': 'phi',
    'f_0(500)': 'sigma_0', 'f_0(980)': 'f_0', 'f_2(1270)': 'f_2', 'f_2(1270)0': 'f_2',
    'f_2': 'f_2',                                  # PDG "Dbar0 f_2": the f_2(1270)
    'f_0(1370)': "f'_0", 'f_0(1370)0': "f'_0", 'f_0(1500)': 'f_0(1500)', 'f_0(1710)': 'f(0)(1710)',
    "f_2^'(1525)": "f'_2", "f_2^'(1525)0": "f'_2", 'f_1(1285)': 'f_1', 'f_1(1420)': "f'_1",
    'f_2(2010)': 'f(2)(2010)',
    'a_1(1260)+': 'a_1+', 'a_1(1260)-': 'a_1-', 'a_1(1260)0': 'a_10',
    'a_2(1320)+': 'a_2+', 'a_2(1320)-': 'a_2-', 'a_2(1320)': 'a_20',
    'a_0(980)+': 'a_0+', 'a_0(980)-': 'a_0-', 'a_0(980)0': 'a_00', 'a_0(980)': 'a_00',
    'b_1(1235)+': 'b_1+', 'b_1()+': 'b_1+', 'b_1()-': 'b_1-', 'b_1()0': 'b_10',
    'h_1(1415)': "h'_1", 'eta(1405)': 'eta(1405)', 'eta(1475)': 'eta(1475)',
    'rho(1450)0': 'rho(2S)0', 'rho(1450)+': 'rho(2S)+', 'rho(1450)-': 'rho(2S)-',
    'rho(1700)0': 'rho(3S)0', 'rho(1700)-': 'rho(3S)-', 'phi(1680)': 'phi(1680)',
    'eta_2(1645)': 'eta(2)(1645)',
    # strange mesons. PDG writes K_1(1270) also as K(1270).
    'K+': 'K+', 'K-': 'K-', 'K0': 'K0', 'Kbar0': 'anti-K0',
    'K0S': 'K_S0', 'KS0': 'K_S0', 'K0L': 'K_L0',
    'K^*(892)0': 'K*0', 'Kbar^*(892)0': 'anti-K*0', 'K^*(892)+': 'K*+', 'K^*(892)-': 'K*-',
    'K^*()0': 'K*0', 'Kbar^*()0': 'anti-K*0', 'K^*()+': 'K*+',
    'K_1(1270)+': 'K_1+', 'K_1(1270)0': 'K_10', 'K(1270)+': 'K_1+', 'K(1270)0': 'K_10',
    'K_1(1400)+': "K'_1+", 'K_1(1400)0': "K'_10", 'K(1400)+': "K'_1+",
    'K^*(1410)+': "K'*+", 'K^*(1410)0': "K'*0", 'K^*(1680)+': "K''*+", 'K^*(1680)0': "K''*0",
    'K_0^*(1430)0': 'K_0*0', 'K_0^*(1430)+': 'K_0*+', 'Kbar_0^*(1430)0': 'anti-K_0*0',
    'K_2^*(1430)0': 'K_2*0', 'K_2^*(1430)+': 'K_2*+', 'K^*_2(1430)+': 'K_2*+',
    'K_3^*(1780)0': 'K_3*0', 'K_3^*(1780)+': 'K_3*+', 'K_4^*(2045)0': 'K_4*0',
    'K_4^*(2045)+': 'K_4*+',
    # charm mesons. PDG "Dbar" with a minus sign is the D^- state itself.
    'D+': 'D+', 'D-': 'D-', 'D0': 'D0', 'Dbar0': 'anti-D0',
    'D()-': 'D-', 'D()0': 'D0', 'Dbar()0': 'anti-D0',
    'D^*(2010)+': 'D*+', 'D^*(2010)-': 'D*-', 'D^*()+': 'D*+', 'D^*()-': 'D*-',
    'D^*(2007)0': 'D*0', 'Dbar^*(2007)0': 'anti-D*0', 'D^*()0': 'D*0', 'Dbar^*()0': 'anti-D*0',
    'D_0^*(2300)-': 'D_0*-', 'Dbar_0^*(2400)-': 'D_0*-', 'D_0^*(2300)0': 'D_0*0',
    'Dbar_0^*(2400)0': 'anti-D_0*0', 'Dbar_0^*(2420)0': 'anti-D_0*0',
    'D_1(2420)-': 'D_1-', 'Dbar_1(2420)0': 'anti-D_10', 'Dbar_1(2421)0': 'anti-D_10',
    'Dbar_1^*(2420)0': 'anti-D_10',                # PDG spelling of the D_1(2420) in B+ -> D_1 pi+
    "D_1^'(2430)-": "D'_1-", 'Dbar_1(2430)0': "anti-D'_10", "Dbar_1^'(2430)0": "anti-D'_10",
    "Dbar_1^'(2427)0": "anti-D'_10",
    'D_2^*(2460)-': 'D_2*-', 'Dbar_2^*(2460)-': 'D_2*-', 'D^*_2(2460)-': 'D_2*-',
    'D_2^*(2460)0': 'D_2*0', 'Dbar_2^*(2460)0': 'anti-D_2*0', 'Dbar_2^*(2462)0': 'anti-D_2*0',
    'D_s()+': 'D_s+', 'D_s()-': 'D_s-', 'D_s^*()+': 'D_s*+', 'D_s^*()-': 'D_s*-',
    # D_s0(2317) and D_s0^*(2317) are one state; D_sJ(2457) is the D_s1(2460) (evt.pdl D_s1, 20433);
    # D_s1(2536) is evt.pdl D'_s1 (10433); D_sJ(2573) is the D_s2*.
    'D_s0(2317)+': 'D_s0*+', 'D_s0(2317)-': 'D_s0*-', 'D_s0^*(2317)+': 'D_s0*+',
    'D_sJ(2457)+': 'D_s1+', 'D_sJ(2457)-': 'D_s1-',
    'D_s1(2536)+': "D'_s1+", 'D_sJ(2573)+': 'D_s2*+',
    # charmonium
    'J/psi(1S)': 'J/psi', 'J/psi': 'J/psi', 'psi(2S)': 'psi(2S)', 'psi(3770)': 'psi(3770)',
    'psi(4040)': 'psi(4040)', 'psi(4160)': 'psi(4160)', 'psi(4415)': 'psi(4415)',
    'eta_c': 'eta_c', 'eta_c(1S)': 'eta_c', 'eta_c(2S)': 'eta_c(2S)',
    'chi_c0': 'chi_c0', 'chi_c0()': 'chi_c0', 'chi_c2()': 'chi_c2', 'chi_c1': 'chi_c1', 'chi_c1(1P)': 'chi_c1', 'chi_c2': 'chi_c2',
    'h_c': 'h_c', 'h_c(1P)': 'h_c', 'chi_c1(3872)': 'X_1(3872)',
    'T_c_cbar_1(4430)+': 'Z(4430)+',
    # baryons
    'p': 'p+', 'pbar': 'anti-p-', 'nbar': 'anti-n0',
    'Lambda': 'Lambda0', 'Lambda0': 'Lambda0', 'Lambdabar': 'anti-Lambda0',
    'Lambdabar0': 'anti-Lambda0', 'Sigma0': 'Sigma0', 'Sigmabar0': 'anti-Sigma0',
    'Delta++': 'Delta++', 'Delta0': 'Delta0', 'Deltabar0': 'anti-Delta0',
    'Delta(1232)++': 'Delta++', 'Delta+': 'Delta+', 'Omega-': 'Omega-',
    'Lambda_c()+': 'Lambda_c+', 'Lambdabar_c()-': 'anti-Lambda_c-', 'Lambda_c()-': 'anti-Lambda_c-',
    'Sigma_c(2455)++': 'Sigma_c++', 'Sigma_c(2455)0': 'Sigma_c0',
    'Sigmabar_c(2455)--': 'anti-Sigma_c--', 'Sigmabar_c(2455)0': 'anti-Sigma_c0',
    'Sigmabar_c(2520)--': 'anti-Sigma_c*--', 'Sigmabar_c(2520)0': 'anti-Sigma_c*0',
    'Xibar_c()0': 'anti-Xi_c0', 'Xibar_c()-': 'anti-Xi_c-', 'Xi_c()0': 'Xi_c0',
    'Omega_c()0': 'Omega_c0',
    'Lambdabar_c(2593)-': 'anti-Lambda_c(2593)-', 'Lambdabar_c(2625)-': 'anti-Lambda_c(2625)-',
    'Lambda(1520)': 'Lambda(1520)0', 'Lambdabar(1520)': 'anti-Lambda(1520)0',
}
_KS_TOKENS = ('K0S', 'KS0', 'K0L')
_LEPTON_Q3 = {_L_PLUS: 3, _L_MINUS: -3, _NU_L: 0}


# --------------------------------------------------------------------------------------------
# PDG cache
# --------------------------------------------------------------------------------------------
def build_pdg_table(out_path=DEFAULT_PDG_JSON):
    """Dump every branching-fraction node of B0 (S042) and B+ (S041) to one JSON file.

    All levels are written, including the indented nodes that the API hides by default
    (`include_subdecays=True`; B -> D l nu is one of them). Needs the `pdg` package.
    Returns the dict that was written (see `load_pdg_table`).
    """
    import warnings
    import pdg                                     # lazy: only this function needs it
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        api = pdg.connect()
        nodes = []
        for name, (base, mother_pdg) in PDG_PARENTS.items():
            part = api.get_particle_by_name(name)
            seen = set()
            bfs = list(part.exclusive_branching_fractions(include_subdecays=True))
            bfs += list(part.inclusive_branching_fractions(include_subdecays=True))
            for bf in bfs:
                if bf.baseid in seen:
                    continue
                seen.add(bf.baseid)
                s = bf.best_summary()
                rec = dict(pdgid=bf.baseid, parent=name, parent_pdgid=base, mother_pdg=mother_pdg,
                           description=bf.description, level=int(bf.subdecay_level),
                           data_type=bf.data_type, mode_number=bf.mode_number,
                           value=None, error_positive=None, error_negative=None, is_limit=None,
                           is_upper_limit=None, confidence_level=None, scale_factor=None,
                           value_type=None)
                if s is not None:
                    rec.update(value=s.value, error_positive=s.error_positive,
                               error_negative=s.error_negative, is_limit=bool(s.is_limit),
                               is_upper_limit=bool(s.is_upper_limit),
                               confidence_level=s.confidence_level, scale_factor=s.scale_factor,
                               value_type=s.value_type)
                nodes.append(rec)
        meta = dict(edition=str(api.edition), data_release_timestamp=str(api.info('data_release_timestamp')),
                    citation=str(api.info('citation')), license=str(api.info('license')),
                    package_version=str(pdg.__version__), parents={k: v[0] for k, v in PDG_PARENTS.items()},
                    note='Written by scripts/tools/pdg_match.py build_pdg_table. One record per '
                         'branching-fraction node, all sub-decay levels. pdgid is stable between '
                         'editions, mode_number is not.')
    out = dict(meta=meta, nodes=nodes)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    return out


def load_pdg_table(path=DEFAULT_PDG_JSON):
    """Read the cached PDG table: {'meta': {...}, 'nodes': [ {pdgid, parent, description, value,
    error_positive, error_negative, is_limit, confidence_level, scale_factor, value_type,
    data_type, level, ...}, ... ]}. Does not need the `pdg` package."""
    with open(path) as f:
        out = json.load(f)
    if 'nodes' not in out or 'meta' not in out:
        raise ValueError(f'{path}: not a PDG table written by build_pdg_table')
    return out


def load_exceptions(path=DEFAULT_EXCEPTIONS):
    """Read the exceptions file (a JSON list; see the module docstring). A missing file is an
    empty list."""
    if path is None or not os.path.exists(path):
        return []
    with open(path) as f:
        exc = json.load(f)
    if not isinstance(exc, list):
        raise ValueError(f'{path}: the exceptions file must be a JSON list')
    known = ('exclude_node', 'fold', 'no_fold', 'force_class', 'rewrite', 'isospin', 'direct_line',
             'extra_uncertainty', 'half_limit')
    for e in exc:
        if e.get('action') not in known:
            raise ValueError(f'{path}: unknown action in {e}; known: {known}')
        if not e.get('reason'):
            raise ValueError(f'{path}: entry without a reason: {e}')
        if e['action'] == 'force_class':
            if 'mother' not in e or 'key' not in e or 'class' not in e:
                raise ValueError(f'{path}: force_class needs mother, key and class: {e}')
        elif e['action'] == 'isospin':
            if any(x not in e for x in ('mother', 'key', 'from_mother', 'from_key')):
                raise ValueError(f'{path}: isospin needs mother, key, from_mother and from_key: {e}')
            if {int(e['mother']), int(e['from_mother'])} != {511, 521}:
                raise ValueError(f'{path}: isospin relates a B0 (511) line and a B+ (521) line: {e}')
        elif e['action'] == 'half_limit':
            if 'mother' not in e or 'key' not in e:
                raise ValueError(f'{path}: half_limit needs mother and key: {e}')
        elif e['action'] == 'extra_uncertainty':
            if 'mother' not in e or 'key' not in e or not float(e.get('relative', 0)) > 0:
                raise ValueError(f'{path}: extra_uncertainty needs mother, key and relative > 0: {e}')
        elif 'pdgid' not in e:
            raise ValueError(f'{path}: {e["action"]} needs a pdgid: {e}')
        if e['action'] == 'direct_line' and ('mother' not in e or 'key' not in e):
            raise ValueError(f'{path}: direct_line needs pdgid, mother and key: {e}')
        if e['action'] in ('fold', 'no_fold') and not e.get('states'):
            raise ValueError(f'{path}: {e["action"]} needs states: {e}')
        if e['action'] == 'rewrite' and '-->' not in str(e.get('description', '')):
            raise ValueError(f'{path}: rewrite needs a description: {e}')
    return exc


# --------------------------------------------------------------------------------------------
# Text parser
# --------------------------------------------------------------------------------------------
def _stem(token):
    """Loose form of a PDG name, used only to find the mother of a sub-decay among the daughters
    of the first part: mass, 'bar' and the marks ^ * _ ' { } are removed."""
    s = re.sub(r'\([^)]*\)', '', token)
    return re.sub(r"[\^\*_'{}]", '', s.replace('bar', ''))


def _tokens(text):
    """Names of one side of a decay: (list of (PDG token, evt name), reason if it failed)."""
    out = []
    for tok in text.split():
        mult = 1
        m = re.match(r'^(\d)([A-Za-z].*)$', tok)          # '3pi+' = pi+ pi+ pi+
        if m and m.group(2) in PDG_NAMES:
            mult, tok = int(m.group(1)), m.group(2)
        if tok not in PDG_NAMES:
            return None, f'unknown name {tok!r}'
        out.extend([(tok, PDG_NAMES[tok])] * mult)
    if not out:
        return None, 'empty'
    return out, ''


def _q3(names, pdl):
    return sum(_LEPTON_Q3[n] if n in _LEPTON_Q3 else pdl.charge3[pdl.name_to_pdg[n]] for n in names)


def parse_description(desc, pdl=None):
    """Parse the text of a PDG branching-fraction node.

    Returns a dict:
      kind   : 'plain', 'product', 'sum', 'nonres' or 'unkeyed'.
      parent : 'B0' or 'B+'.
      main   : list of alternatives (one, or several for a sum); each is a tuple of evt.pdl names.
               Lepton nodes carry the placeholders '@l+', '@l-', '@nu' (see `node_variants`).
      subs   : for a product: list of (evt name of the mother, list of alternative daughter
               tuples). The mother is one of the names in `main`.
      n_ks   : number of K_S0 / K_L0 written in the text.
      lepton : True if the node is written with "lepton".
      reason : why the node is unkeyed.
    The charge of every decay is checked; a node that fails is unkeyed.
    """
    pdl = pdl or dt.load_pdl()
    out = dict(kind='unkeyed', parent='', main=[], subs=[], n_ks=0, lepton=False, reason='')
    if '-->' not in desc:
        out['reason'] = 'no arrow'
        return out
    lhs, rhs = [s.strip() for s in desc.split('-->', 1)]
    out['parent'] = lhs
    if lhs not in PDG_PARENTS:
        out['reason'] = f'unknown parent {lhs!r}'
        return out
    q_parent = pdl.charge3[PDG_PARENTS[lhs][1]]
    out['n_ks'] = sum(rhs.split().count(t) for t in _KS_TOKENS)

    if re.search(r'nonresonant|\(NR\)|NR\(', rhs):
        out['kind'] = 'nonres'
        return out
    if '[' in rhs or ']' in rhs:
        out['reason'] = 'bracket syntax (D decay or partial wave)'
        return out

    # split off the sub-decays: " x B(X --> ...)" or ", X --> ..."
    sub_texts = []
    if re.search(r'x\s*B\(', rhs):
        parts = re.split(r'\s*x\s*B\(', rhs)
        main_text = parts[0]
        for p in parts[1:]:
            p = p.strip()
            if not p.endswith(')'):
                out['reason'] = 'unbalanced product'
                return out
            sub_texts.append(p[:-1])
    elif ',' in rhs:
        parts = [p.strip() for p in rhs.split(',')]
        main_text, sub_texts = parts[0], parts[1:]
    else:
        main_text = rhs
    if '-->' in main_text or any('-->' not in s for s in sub_texts):
        out['reason'] = 'unknown syntax'
        return out

    def alternatives(text, q_expected):
        alts = []
        for a in text.split(' + '):
            toks, why = _tokens(a)
            if toks is None:
                return None, why
            if _q3([n for _, n in toks], pdl) != q_expected:
                return None, 'charge not conserved'
            alts.append(toks)
        return alts, ''

    main, why = alternatives(main_text, q_parent)
    if main is None:
        out['reason'] = why
        return out
    if any(len(a) < 2 for a in main):
        out['reason'] = 'fewer than two daughters'
        return out
    if sub_texts and len(main) > 1:
        out['reason'] = 'sum with sub-decays'
        return out

    subs = []
    for s in sub_texts:
        m_tok, s_rhs = [x.strip() for x in s.split('-->', 1)]
        hits = [n for t, n in main[0] if _stem(t) == _stem(m_tok)]
        if len(hits) != 1:
            out['reason'] = f'mother of the sub-decay {m_tok!r} not found once in the first part'
            return out
        alts, why = alternatives(s_rhs, _q3(hits, pdl))
        if alts is None:
            out['reason'] = f'sub-decay: {why}'
            return out
        subs.append((hits[0], [tuple(n for _, n in a) for a in alts]))

    out['main'] = [tuple(n for _, n in a) for a in main]
    out['subs'] = subs
    out['lepton'] = any(n in _LEPTON_Q3 for a in out['main'] for n in a)
    out['kind'] = 'product' if subs else ('sum' if len(main) > 1 else 'plain')
    return out


def node_variants(parsed):
    """Lepton flavors of a parsed node: [('', main, subs)] or the 'e' and the 'mu' version.
    With a positive lepton the neutrino is nu_l, with a negative one anti-nu_l."""
    if not parsed['lepton']:
        return [('', parsed['main'], parsed['subs'])]
    out = []
    for lep in ('e', 'mu'):
        def sub(alt):
            nu = f'nu_{lep}' if _L_PLUS in alt else f'anti-nu_{lep}'
            return tuple({_L_PLUS: f'{lep}+', _L_MINUS: f'{lep}-', _NU_L: nu}.get(n, n) for n in alt)
        out.append((lep, [sub(a) for a in parsed['main']], parsed['subs']))
    return out


# --------------------------------------------------------------------------------------------
# Keys with collapsed neutral kaons
# --------------------------------------------------------------------------------------------
def collapse_key(codes):
    """Comparison key of a final state: sorted PDG codes with K0, anti-K0, K_S0 and K_L0 replaced
    by one symbol (311)."""
    return tuple(sorted(KSYM if int(c) in _K0_CODES else int(c) for c in codes))


def _loose_key(codes):
    """As `collapse_key`, with the K*0 flavor ignored too."""
    return tuple(sorted(313 if c == -313 else c for c in collapse_key(codes)))


def conjugate_key(key, pdl):
    """Charge-conjugate final state, in the canonical order of `decay_table.mode_key`."""
    return dt.mode_key([pdl.conjugate(int(c)) for c in key])


# --------------------------------------------------------------------------------------------
# Sub-decay tables of the decay file
# --------------------------------------------------------------------------------------------
def _read_widths(path):
    """name -> (width in GeV, c*tau) from an evt.pdl file."""
    out = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 11 and t[0] == 'add' and t[2] == 'Particle':
                out[t[3]] = (float(t[6]), float(t[10]))
    return out


class DecayGraph:
    """The decay file as a graph: who decays to what, with which probability.

    dec_all : `decay_table.parse_dec(mothers=None)`.
    Attributes
    ----------
    blocks : name -> list of (probability, daughter names). Alias particles (K*S, K*L, K*0T) keep
             their own tables. The probabilities are normalised to the sum of the block.
    code   : name -> PDG code (aliases resolved).
    broad  : names that are folded by default: width >= `BROAD_WIDTH`, lifetime 0, and a decay
             table in the file, except `NO_FOLD_DEFAULT`.
    broad_codes : their PDG codes.
    """

    def __init__(self, dec_all, pdl, broad_width=BROAD_WIDTH, max_depth=MAX_DEPTH):
        self.pdl = pdl
        self.max_depth = max_depth
        alias = dec_all.attrs.get('aliases', {})

        def base(n):
            seen = set()
            while n in alias and n not in seen:
                seen.add(n)
                n = alias[n]
            return n
        self.blocks, self.code = {}, {}
        for mother, g in dec_all.groupby('mother', sort=False):
            if mother in dt.B_MOTHERS:
                continue
            tot = g['bf'].sum()
            if tot <= 0:
                continue
            self.blocks[mother] = [(bf / tot, None if py else tuple(names))
                                   for bf, names, py in zip(g['bf'], g['names'], g['is_pythia']) if bf > 0]
            self.code[mother] = int(g['mother_pdg'].iloc[0])
        for names in dec_all['names']:
            for n in names:
                if n not in self.code:
                    self.code[n] = pdl.name_to_pdg[base(n)]
        for n, c in pdl.name_to_pdg.items():
            self.code.setdefault(n, c)
        widths = _read_widths(pdl.path)
        self.broad = set()
        for n in self.blocks:
            w, ctau = widths.get(base(n), (0.0, 1.0))
            if w >= broad_width and ctau == 0 and base(n) not in NO_FOLD_DEFAULT:
                self.broad.add(n)
        self.broad_codes = frozenset(self.code[n] for n in self.broad)
        self._memo = {}

    def default_fold(self, content=()):
        """Codes that may be folded for a node whose text names the codes `content`."""
        return self.broad_codes - frozenset(int(c) for c in content)

    def _leaf(self, name):
        c = self.code[name]
        return KSYM if c in _K0_CODES else c

    def decay_once(self, name, fold, depth=0):
        """Distribution {collapsed key: probability} of the final states of `name`, decayed once
        and then folded as far as `fold` allows."""
        memo_key = (name, fold, depth)
        if memo_key in self._memo:
            return self._memo[memo_key]
        out = {}
        for p, names in self.blocks.get(name, ()):
            if names is None:                              # quark-level line: unknown final state
                out[(_OPAQUE,)] = out.get((_OPAQUE,), 0.0) + p
                continue
            for k, q in self.expand(names, fold, depth + 1).items():
                out[k] = out.get(k, 0.0) + p * q
        self._memo[memo_key] = out
        return out

    def expand(self, names, fold, depth=0):
        """Distribution {collapsed key: probability} of the final states of a list of particle
        names, where every particle whose code is in `fold` is replaced by its decays."""
        result = {(): 1.0}
        for n in names:
            if depth < self.max_depth and self.code[n] in fold and n in self.blocks:
                d = self.decay_once(n, fold, depth)
            else:
                d = {(self._leaf(n),): 1.0}
            new = {}
            for k1, p1 in result.items():
                for k2, p2 in d.items():
                    p = p1 * p2
                    if p < 1e-12:
                        continue
                    k = tuple(sorted(k1 + k2))
                    new[k] = new.get(k, 0.0) + p
            result = new
        return result

    def expand_codes(self, codes, fold):
        """`expand` for PDG codes (the particle of the code, not an alias)."""
        return self.expand([self.pdl.name(c) for c in codes], fold)


# --------------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------------
def make_nodes(pdg_table, pdl, exceptions=()):
    """Parse every node of the cached PDG table ('rewrite' exceptions replace the text first).

    Returns a list of dicts, one per node and lepton flavor:
      uid, pdgid, mother (511 / 521), kind, description, level,
      targets  : collapsed keys of the alternatives (plain: one),
      names    : the alternatives as evt names,
      subs     : [(mother name, [alternative name tuples])] for product nodes,
      content  : PDG codes named in the text,
      n_bodies : number of daughters of the first part,
      value, err (mean of the two errors), is_limit; value and err include the factor 2 of a
                 single K_S0 / K_L0,
      multi_ks : two or more K_S0 / K_L0 in the text (no c),
      usable   : keyed, with a value, and not multi_ks.
    """
    nodes = []
    rewrite = {e['pdgid']: e['description'] for e in exceptions if e['action'] == 'rewrite'}
    for rec in pdg_table['nodes']:
        p = parse_description(rewrite.get(rec['pdgid'], rec['description']), pdl)
        value = rec.get('value')
        ep, en = rec.get('error_positive'), rec.get('error_negative')
        err = np.nan if value is None else 0.5 * ((ep or 0.0) + (en or 0.0))
        ks = 2.0 if p['n_ks'] == 1 else 1.0
        base = dict(pdgid=rec['pdgid'], mother=PDG_PARENTS[rec['parent']][1], kind=p['kind'],
                    description=rec['description'], level=rec.get('level', 0), reason=p['reason'],
                    value=np.nan if value is None else value * ks, err=err * ks,
                    is_limit=bool(rec.get('is_limit')), multi_ks=p['n_ks'] >= 2,
                    rewritten=rec['pdgid'] in rewrite,
                    data_type=rec.get('data_type', ''))
        if p['kind'] in ('unkeyed', 'nonres'):
            nodes.append(dict(base, uid=rec['pdgid'], targets=[], names=[], subs=[], content=(),
                              n_bodies=0, usable=False))
            continue
        for lep, main, subs in node_variants(p):
            code = lambda n: pdl.name_to_pdg[n]
            content = {code(n) for a in main for n in a}
            content |= {code(n) for _, alts in subs for a in alts for n in a}
            nodes.append(dict(base, uid=rec['pdgid'] + (f'[{lep}]' if lep else ''),
                              targets=[collapse_key(code(n) for n in a) for a in main],
                              names=main, subs=subs, content=tuple(sorted(content)),
                              n_bodies=max(len(a) for a in main),
                              usable=value is not None and p['n_ks'] < 2))
    return nodes


def _new_result():
    return dict(pdg_class='none', how='', pdgids=[], bf_pdg=np.nan, bf_pdg_err=np.nan, c=1.0,
                sigma_c=np.nan, flags=set(), f_res_mc=np.nan, f_res_pdg=np.nan, res_node='',
                p_node=np.nan, c_prod=np.nan, sigma_c_prod=np.nan, chi2_prod=np.nan, n_prod=0,
                prod_pdgids=[], alt_pdgids=[], unclosed_nodes=[], twin_pdgids=[], direct_node='', extra_rel=0.0,
                fed_nodes=[], node_info=[],
                _diff=-np.inf)


def _sub_signature(node):
    """Sub-decays of a product node: per sub-mother, the set of alternative daughter sets."""
    return tuple(sorted((m, frozenset(tuple(sorted(a)) for a in alts)) for m, alts in node['subs']))


def _independent_products(meas):
    """Reduce the measured product nodes of one line, [(node, pred)], to independent ones:
    (a) of nodes with the same sub-decays (one measurement listed twice), keep the most precise;
    (b) drop a node whose alternatives are a strict part of another node's (a component of a PDG
    sum node such as 'D_s1 --> D*0 K+ + D*+ K0'); the sum node is kept."""
    def rel(item):
        n = item[0]
        return n['err'] / n['value'] if n['value'] > 0 else np.inf
    best = {}
    for it in meas:
        s = _sub_signature(it[0])
        if s not in best or rel(it) < rel(best[s]):
            best[s] = it

    def part_of(s, t):
        return s != t and len(s) == len(t) and all(ms == mt and a <= b for (ms, a), (mt, b) in zip(s, t))
    return [it for s, it in best.items() if not any(part_of(s, t) for t in best)]


def _match_block(mother, lines, nodes, graph, exceptions=(), preset=None):
    """Match the exclusive lines of one B block with the PDG nodes of that B.

    lines : rows of `parse_dec` for this mother (exclusive lines only).
    preset : {collapsed key: (c, sigma_c, PDG identifiers)} for the lines that take the c of their
             isospin partner in the other B block ('isospin' exceptions).
    Returns (results, node_table):
      results    : {collapsed key: result dict} for every final state of the block;
      node_table : DataFrame, one row per node.
    """
    pdl = graph.pdl
    excluded = {e['pdgid'] for e in exceptions if e['action'] == 'exclude_node'}
    fold_add, fold_rm = {}, {}
    for e in exceptions:
        if e['action'] in ('fold', 'no_fold'):
            codes = {pdl.name_to_pdg[s] for s in e['states']}
            (fold_add if e['action'] == 'fold' else fold_rm).setdefault(e['pdgid'], set()).update(codes)

    # 'direct_line' exceptions: lines that stand for the final state of a node (the decay file
    # routes the whole final state through one resonance), treated as its direct line
    direct_extra = {}
    for e in exceptions:
        if e['action'] == 'direct_line' and int(e['mother']) == mother:
            direct_extra.setdefault(e['pdgid'], set()).add(collapse_key(e['key']))

    def is_direct(node, k):
        return k in node['targets'] or k in direct_extra.get(node['pdgid'], ())

    def fold_of(node):
        f = graph.default_fold(node['content'])
        return (f | frozenset(fold_add.get(node['pdgid'], ()))) - frozenset(fold_rm.get(node['pdgid'], ()))

    # final states of the block (collapsed keys) and their BF
    line_names = [tuple(n) for n in lines['names']]
    line_bf = lines['bf'].to_numpy(dtype=float)
    line_ckey = [collapse_key(d) for d in lines['daughters']]
    cmode_bf = {}
    for k, bf in zip(line_ckey, line_bf):
        cmode_bf[k] = cmode_bf.get(k, 0.0) + bf
    results = {k: _new_result() for k in cmode_bf}

    # feeders of a set of targets under a folding rule: {collapsed key of the line: sum BF x P}
    index_cache = {}

    def index_for(fold):
        if fold not in index_cache:
            idx = {}
            for i, names in enumerate(line_names):
                if line_bf[i] <= 0:
                    continue
                for t, p in graph.expand(names, fold).items():
                    idx.setdefault(t, []).append((i, p))
            index_cache[fold] = idx
        return index_cache[fold]

    def feeders_of(node):
        idx = index_for(fold_of(node))
        out = {}
        for t in set(node['targets']):
            for i, p in idx.get(t, ()):
                out[line_ckey[i]] = out.get(line_ckey[i], 0.0) + line_bf[i] * p
        return out

    mine = [n for n in nodes if n['mother'] == mother]
    rows = {n['uid']: dict(uid=n['uid'], pdgid=n['pdgid'], mother=mother, kind=n['kind'],
                           description=n['description'], level=n['level'], value=n['value'],
                           err=n['err'], is_limit=n['is_limit'], multi_ks=n['multi_ks'],
                           n_bodies=n['n_bodies'], bf_mc=np.nan, ratio=np.nan, n_feeders=0,
                           role='', c_assigned=np.nan, f_res_mc=np.nan, f_res_pdg=np.nan,
                           bf_mc_weighted=np.nan, closure=np.nan, closure_pull=np.nan,
                           reason=n['reason']) for n in mine}

    def flag_exception(keys):
        for k in keys:
            if k in results:
                results[k]['flags'].add('exception_applied')

    final_state = [n for n in mine if n['kind'] in ('plain', 'sum')]
    product = [n for n in mine if n['kind'] == 'product']
    feed = {}
    for n in final_state:
        f = feeders_of(n)
        if n['pdgid'] in excluded:
            rows[n['uid']]['role'] = 'excluded'
            flag_exception(f)
            continue
        if n['pdgid'] in fold_add or n['pdgid'] in fold_rm or n['rewritten']:
            flag_exception(f)
        if n['multi_ks']:
            rows[n['uid']]['role'] = 'multi_ks'
            for k in f:
                if f[k] / cmode_bf[k] >= P_MIN:
                    results[k]['flags'].add('multi_ks')
        feed[n['uid']] = f
        tot = sum(f.values())
        rows[n['uid']].update(bf_mc=tot, n_feeders=len(f),
                              ratio=n['value'] / tot if tot > 0 and n['usable'] else np.nan)

    measured = [n for n in final_state if n['uid'] in feed and n['usable'] and not n['is_limit']
                and feed[n['uid']]]

    def assign(k, how, node, c, sigma, p_node=np.nan):
        r = results[k]
        r.update(pdg_class='measured', how=how, pdgids=[node['pdgid']], bf_pdg=node['value'],
                 bf_pdg_err=node['err'], c=c, sigma_c=sigma, p_node=p_node)

    # lines fixed by an 'isospin' exception: the c of the partner line of the other B block
    for k, (c, sigma, ids, inherited) in (preset or {}).items():
        if cmode_bf.get(k, 0.0) <= 0:
            raise ValueError(f'exceptions: isospin target {k} of {mother} is not a line with BF > 0')
        results[k].update(pdg_class='measured', how='isospin', pdgids=list(ids),
                          bf_pdg=c * cmode_bf[k], bf_pdg_err=sigma * cmode_bf[k], c=c, sigma_c=sigma)
        results[k]['flags'].update(('isospin', 'exception_applied', *inherited))

    # 1. exact nodes: one target, fed by the line of that final state alone
    exact = [n for n in measured if len(n['targets']) == 1 and set(feed[n['uid']]) == {n['targets'][0]}]
    exact.sort(key=lambda n: (n['err'] / n['value'] if n['value'] > 0 else np.inf, n['uid']))
    exact_uids = set()
    for n in exact:
        k = n['targets'][0]
        exact_uids.add(n['uid'])
        rows[n['uid']]['role'] = 'exact'
        if results[k]['how'] == 'isospin':
            raise ValueError(f'exceptions: isospin target {k} of {mother} has its own PDG entry {n["pdgid"]}')
        if results[k]['how']:                         # a more precise node of the same final state
            results[k]['pdgids'].append(n['pdgid'])
            continue
        c = n['value'] / cmode_bf[k]
        assign(k, 'exact', n, c, n['err'] / cmode_bf[k], 1.0)
        rows[n['uid']]['c_assigned'] = c

    # product nodes: prediction of the decay file = BF_line x BF_sub
    prod = {}
    for n in product:
        k = n['targets'][0]
        if n['pdgid'] in excluded:
            rows[n['uid']]['role'] = 'excluded'
            flag_exception([k])
            continue
        if cmode_bf.get(k, 0.0) <= 0:
            continue
        if n['rewritten'] or n['pdgid'] in fold_add or n['pdgid'] in fold_rm:
            flag_exception([k])
        fold = fold_of(n)
        bf_sub = 1.0
        for m_name, alts in n['subs']:
            d = graph.decay_once(m_name, fold)
            bf_sub *= sum(d.get(collapse_key(pdl.name_to_pdg[x] for x in a), 0.0) for a in alts)
        pred = cmode_bf[k] * bf_sub
        rows[n['uid']].update(bf_mc=pred, n_feeders=1,
                              ratio=n['value'] / pred if pred > 0 and n['usable'] else np.nan)
        if n['multi_ks']:
            rows[n['uid']]['role'] = 'multi_ks'
            results[k]['flags'].add('multi_ks')
            continue
        if not n['usable'] or pred <= 0:
            continue
        rows[n['uid']]['role'] = 'product'
        prod.setdefault(k, []).append((n, pred))

    def assign_products():
        for k, lst in prod.items():
            r = results[k]
            meas = _independent_products([(n, pred) for n, pred in lst if not n['is_limit']])
            kept = {n['uid'] for n, _ in meas}
            for n, pred in lst:
                if not n['is_limit'] and n['uid'] not in kept:
                    rows[n['uid']]['role'] = 'duplicate'
            for n, pred in lst:
                if n['is_limit'] and pred > n['value'] * (1 + REL_TOL):
                    r['flags'].add('above_limit')
            if meas:
                cs = np.array([n['value'] / pred for n, pred in meas])
                ss = np.array([n['err'] / pred for n, pred in meas])
                if np.all(ss > 0):
                    w = 1.0 / ss ** 2
                    c, sigma = float((w * cs).sum() / w.sum()), float(1.0 / np.sqrt(w.sum()))
                    chi2 = float((w * (cs - c) ** 2).sum())
                else:
                    c, sigma, chi2 = float(cs.mean()), np.nan, np.nan
                r.update(c_prod=c, sigma_c_prod=sigma, chi2_prod=chi2, n_prod=len(meas),
                         prod_pdgids=[n['pdgid'] for n, _ in meas])
                if r['how'] == 'isospin':
                    raise ValueError(f'exceptions: isospin target {k} of {mother} has its own product entry')
                if not r['how'] and k not in forced:
                    best = min(meas, key=lambda x: x[0]['err'] / x[0]['value'] if x[0]['value'] > 0 else np.inf)[0]
                    r.update(pdg_class='measured', how='product', pdgids=[n['pdgid'] for n, _ in meas],
                             bf_pdg=best['value'], bf_pdg_err=best['err'], c=c, sigma_c=sigma)

    def assign_product_limits():
        for k, lst in prod.items():
            r = results[k]
            if r['how'] or any(not n['is_limit'] for n, _ in lst):
                continue
            n, pred = min(lst, key=lambda x: x[0]['value'] / x[1])
            r.update(pdg_class='limit', pdgids=[m['pdgid'] for m, _ in lst], bf_pdg=n['value'],
                     bf_pdg_err=0.0)

    # own upper limits of a line, as a bound on its c: plain limit nodes of its final state
    # (value / BF_MC) and product limit nodes (value / (BF_line x BF_sub of the decay file))
    cap, cap_id = {}, {}

    def add_cap(k, value, node):
        if value < cap.get(k, np.inf):
            cap[k], cap_id[k] = value, node['pdgid']
    for n in final_state:
        if n['uid'] in feed and n['usable'] and n['is_limit'] and len(n['targets']) == 1:
            k = n['targets'][0]
            if cmode_bf.get(k, 0.0) > 0:
                add_cap(k, n['value'] / cmode_bf[k], n)
    for k, lst in prod.items():
        for n, pred in lst:
            if n['is_limit']:
                add_cap(k, n['value'] / pred, n)

    # forced classes, before the totals are split: a forced line takes no c from a total and
    # enters its remainder like any line without a c (1 +- 1)
    forced = set()
    for e in exceptions:
        if e['action'] != 'force_class' or int(e['mother']) != mother:
            continue
        k = collapse_key(e['key'])
        if k not in results:
            raise ValueError(f'exceptions: no final state {e["key"]} of {mother}')
        r = results[k]
        if r['how'] and r['how'] != 'isospin' and e['class'] != 'measured':
            r.update(how='', c=1.0, sigma_c=np.nan, pdgids=[], bf_pdg=np.nan, bf_pdg_err=np.nan,
                     p_node=np.nan)
        r['pdg_class'] = e['class']
        r['flags'].add('exception_applied')
        forced.add(k)

    # 'half_limit': a line that the decay file puts above its own PDG limit takes c = limit/2 with
    # an uncertainty of limit/2 (the range [0, limit] as a flat-ish prior)
    for e in exceptions:
        if e['action'] != 'half_limit' or int(e['mother']) != mother:
            continue
        k = collapse_key(e['key'])
        if k not in results or k not in cap:
            raise ValueError(f'exceptions: half_limit on {e["key"]} of {mother}, which has no own limit')
        if results[k]['how']:
            raise ValueError(f'exceptions: half_limit on {e["key"]} of {mother}, which has a c already')
        if not cap[k] < 1.0 - REL_TOL:
            raise ValueError(f'exceptions: half_limit on {e["key"]} of {mother}: the decay file is not above the limit')
        r = results[k]
        r.update(pdg_class='limit', how='half_limit', pdgids=[cap_id[k]], c=0.5 * cap[k], sigma_c=0.5 * cap[k],
                 bf_pdg=cap[k] * cmode_bf[k], bf_pdg_err=0.0)
        r['flags'].add('exception_applied')

    # extra relative uncertainty on a c (e.g. from a sub-decay BF the decay file fixes at 100 %, or
    # two measurements in conflict); optional 'flag' is added to the line
    extra_done = set()

    def apply_extra(final):
        for i, e in enumerate(exceptions):
            if e['action'] != 'extra_uncertainty' or int(e['mother']) != mother or i in extra_done:
                continue
            k = collapse_key(e['key'])
            if k not in results:
                raise ValueError(f'exceptions: extra_uncertainty on {e["key"]} of {mother}: no such line')
            r = results[k]
            if not r['how']:
                if final:
                    raise ValueError(f'exceptions: extra_uncertainty on {e["key"]} of {mother}, which has no c')
                continue
            r['sigma_c'] = float(np.hypot(r['sigma_c'], float(e['relative']) * r['c']))
            r['extra_rel'] = float(np.hypot(r['extra_rel'], float(e['relative'])))
            r['flags'].add('exception_applied')
            if e.get('flag'):
                r['flags'].add(str(e['flag']))
            extra_done.add(i)

    # final-state nodes, the most exclusive first. Only the DIRECT lines of a node (their final
    # state is the node's own; in the decay file the PHSP line meant as the non-resonant part)
    # take a c from it: the remainder of the total after all other lines, clamped at 0. Lines with
    # a c are subtracted with c and sigma_c; lines without one with c = 1 +- 1; lines without one
    # that the decay file puts above their own limit with limit/2 +- limit/2. The total is not
    # exactly the sum of its parts (interference): F_INTERFERENCE of the subtracted part is added
    # to the uncertainty. A line that reaches the node through a resonance takes nothing from it.
    def node_pass():
        rest_nodes = [n for n in measured if n['uid'] not in exact_uids]
        # nodes with a 'direct_line' exception first: their stand-in line is a resonance line that
        # can also feed smaller nodes, where it must enter with its c
        rest_nodes.sort(key=lambda n: (n['pdgid'] not in direct_extra, n['n_bodies'],
                                       n['err'] / n['value'] if n['value'] > 0 else np.inf, n['uid']))

        def is_free(n, k, f):
            return (not results[k]['how'] and k not in forced and f[k] / cmode_bf[k] >= P_MIN
                    and is_direct(n, k))
        cand, used_nb = {}, {}                      # line -> {pdgid: n_bodies} of the nodes it could take
        for n in rest_nodes:
            f = feed[n['uid']]
            for k in f:
                if is_free(n, k, f):
                    cand.setdefault(k, {})[n['pdgid']] = n['n_bodies']
        for n in rest_nodes:
            f = feed[n['uid']]
            free = [k for k in f if is_free(n, k, f)]
            if not free:
                rows[n['uid']]['role'] = 'covered'
                continue
            for k in free:
                if k not in n['targets']:
                    results[k]['flags'].add('exception_applied')
            num, var, sub = n['value'], n['err'] ** 2, 0.0
            for k in f:
                if k in free:
                    continue
                r = results[k]
                if not r['how'] and cap.get(k, np.inf) < 1.0 - REL_TOL:
                    part, s2 = 0.5 * cap[k] * f[k], (0.5 * cap[k] * f[k]) ** 2
                elif r['how'] and np.isfinite(r['sigma_c']):
                    part, s2 = r['c'] * f[k], (r['sigma_c'] * f[k]) ** 2
                elif r['how']:
                    part, s2 = r['c'] * f[k], 0.0
                else:                               # no c of its own: c = 1 +- 1
                    part, s2 = f[k], f[k] ** 2
                num -= part
                var += s2
                sub += part
            den = sum(f[k] for k in free)
            c_raw = num / den
            sigma = float(np.hypot(np.sqrt(var), F_INTERFERENCE * sub)) / den
            c = max(c_raw, 0.0)
            rows[n['uid']].update(role='node', c_assigned=c)
            for k in free:
                assign(k, 'node', n, c, sigma, f[k] / cmode_bf[k])
                used_nb[k] = n['n_bodies']
                if c_raw < 0:
                    results[k]['flags'].add('remainder_clamped')
                if sigma > c:
                    results[k]['flags'].add('remainder_unstable')
        # the choice between nodes of the same size rests on the order above: flag it
        for k, nb in used_nb.items():
            used = results[k]['pdgids'][0]
            alt = sorted(p for p, m in cand.get(k, {}).items() if p != used and m == nb)
            if alt:
                results[k]['flags'].add('node_ambiguous')
                results[k]['alt_pdgids'] = alt

    # 2 and 3. A line with product nodes has a resonance-explicit measurement of its own: it is
    # fixed by them before the remainders of the final-state nodes are formed.
    assign_products()
    apply_extra(final=False)                    # lines with an exact, product or isospin c: before the
    node_pass()                                 # pass, so that remainders use the widened sigma_c
    assign_product_limits()

    # limits and zero-BF final states: plain nodes equal to the final state
    for n in final_state:
        if n['uid'] not in feed or not n['usable'] or len(n['targets']) != 1:
            continue
        k = n['targets'][0]
        if k not in results:
            continue
        r = results[k]
        if n['is_limit'] and cmode_bf[k] > n['value'] * (1 + REL_TOL):
            r['flags'].add('above_limit')
        if r['how'] or r['pdg_class'] == 'measured':
            continue
        if n['is_limit']:
            if r['pdg_class'] == 'limit' and r['bf_pdg'] <= n['value']:
                r['pdgids'].append(n['pdgid'])
                continue
            r.update(pdg_class='limit', pdgids=r['pdgids'] + [n['pdgid']], bf_pdg=n['value'],
                     bf_pdg_err=0.0)
            rows[n['uid']]['role'] = 'limit'
        elif cmode_bf[k] <= 0:
            # measured in PDG, zero BF in the decay file: no c can be formed
            r.update(pdg_class='measured', pdgids=[n['pdgid']], bf_pdg=n['value'], bf_pdg_err=n['err'])

    # resonance content of the nodes with three or more bodies
    plain_meas = [n for n in measured if len(n['targets']) == 1]
    prod_meas = [n for n in product if n['usable'] and not n['is_limit'] and n['pdgid'] not in excluded]
    for n in measured:
        if n['n_bodies'] < 3:
            continue
        f = feed[n['uid']]
        tot = sum(f.values())
        if tot <= 0:
            continue
        fold = fold_of(n)
        targets = set(n['targets'])
        direct = sum(v for k, v in f.items() if k in targets)
        f_mc = 1.0 - direct / tot

        def reach(names):
            d = graph.expand(names, fold)
            return sum(d.get(t, 0.0) for t in targets)
        subs = []                                       # (node, names of its final state, P)
        for m in plain_meas:
            if m['uid'] == n['uid'] or m['n_bodies'] >= n['n_bodies'] or m['targets'][0] in targets:
                continue
            if not (set(m['content']) & fold):
                continue
            p = reach(m['names'][0])
            if p > 0:
                subs.append((m, m['names'][0], p))
        plain_mains = {m['targets'][0] for m, _, _ in subs}
        for m in prod_meas:
            if m['targets'][0] in plain_mains or m['targets'][0] in targets:
                continue
            if any(len(alts) != 1 for _, alts in m['subs']):
                continue
            names = list(m['names'][0])
            for m_name, alts in m['subs']:
                names.remove(m_name)
                names.extend(alts[0])
            p = reach(names)
            if p > 0:
                subs.append((m, m['names'][0], p))
        # keep the sub-nodes that feed no other sub-node, so that nothing is counted twice
        kept = []
        for m, names, p in subs:
            inside = False
            for m2, _, _ in subs:
                if m2 is m or m2['kind'] != 'plain' or m2['n_bodies'] <= m['n_bodies']:
                    continue
                if graph.expand(names, fold_of(m2)).get(m2['targets'][0], 0.0) > 0:
                    inside = True
                    break
            if not inside:
                kept.append((m, p))
        f_pdg = sum(m['value'] * p for m, p in kept) / n['value'] if n['value'] > 0 else np.nan
        rows[n['uid']].update(f_res_mc=f_mc, f_res_pdg=f_pdg)
        if len(f) == 1 and not kept:                   # one line, no resonant sub-node: nothing to compare
            continue
        diff = min(f_pdg, 1.0) - f_mc              # PDG sub-nodes can add up to more than the node
        for k in f:
            if f[k] / cmode_bf[k] < P_MIN:
                continue
            r = results[k]
            if diff > r['_diff']:
                r.update(f_res_mc=f_mc, f_res_pdg=f_pdg, res_node=n['pdgid'], _diff=diff)
            if diff > RES_MISMATCH:
                r['flags'].add('resonance_mismatch')

    # extra relative uncertainty on the c of the lines fixed by the node pass (the others got it
    # before the pass, see apply_extra)
    apply_extra(final=True)

    # `unmeasured`: a line without a c (class none) that feeds a measured final-state node. The
    # node measures the sum, not this line: c = 1 with an uncertainty of 100 %.
    for n in measured:
        for k, v in feed[n['uid']].items():
            r = results[k]
            if r['how'] or v / cmode_bf[k] < P_MIN:
                continue
            if r['pdg_class'] == 'none':
                r['flags'].add('unmeasured')
                r['sigma_c'] = UNMEASURED_SIGMA
            elif r['pdg_class'] == 'limit' and cap.get(k, np.inf) < 1.0 - REL_TOL:
                r['flags'].add('limit_capped')      # counted as limit/2 +- limit/2 in the remainders

    # own limits against the corrected BF: max(1, c) x BF_MC above any limit of the line
    for k, c_max in cap.items():
        if max(1.0, results[k]['c']) > c_max * (1 + REL_TOL):
            results[k]['flags'].add('above_limit')

    # closure of every measured final-state node with the final c of its feeding lines
    for n in measured:
        f = feed[n['uid']]
        w = sum(results[k]['c'] * v for k, v in f.items())
        pull = (w - n['value']) / n['err'] if n['err'] > 0 else np.nan
        rows[n['uid']].update(bf_mc_weighted=w, closure=w / n['value'] if n['value'] > 0 else np.nan,
                              closure_pull=pull)
        info = (f"{n['pdgid']}: PDG {n['value']:.3g} +- {n['err']:.2g}, decay file {sum(f.values()):.3g}, "
                f"after c {w:.3g} ({pull:+.1f} sigma)" if np.isfinite(pull) else
                f"{n['pdgid']}: PDG {n['value']:.3g}, decay file {sum(f.values()):.3g}, after c {w:.3g}")
        for k, v in f.items():
            if v / cmode_bf[k] >= P_MIN and n['pdgid'] not in results[k]['fed_nodes']:
                results[k]['fed_nodes'] = results[k]['fed_nodes'] + [n['pdgid']]
                results[k]['node_info'] = results[k]['node_info'] + [info]
        if np.isfinite(pull) and abs(pull) > PULL_REVIEW:
            for k, v in f.items():
                if v / cmode_bf[k] >= P_MIN:
                    results[k]['flags'].add('node_not_closed')
                    if n['pdgid'] not in results[k]['unclosed_nodes']:
                        results[k]['unclosed_nodes'] = results[k]['unclosed_nodes'] + [n['pdgid']]

    for k, r in results.items():
        r['bf_mc_sum'] = cmode_bf[k]
        del r['_diff']
    # feeders of each node, for the review: (collapsed key of the line, BF x P, BF of the line)
    for uid, f in feed.items():
        rows[uid]['feeders'] = sorted(((k, v, cmode_bf[k]) for k, v in f.items()), key=lambda x: -x[1])
    return results, pd.DataFrame(list(rows.values()))


# --------------------------------------------------------------------------------------------
# Mode table
# --------------------------------------------------------------------------------------------
TABLE_COLUMNS = ['mother_pdg', 'key', 'names', 'n_daughters', 'is_sl', 'tau', 'n_charm', 'bf_mc',
                 'pdg_class', 'how', 'pdgids', 'bf_pdg', 'bf_pdg_err', 'c', 'sigma_c', 'pull',
                 'f_res_mc', 'f_res_pdg', 'flags', 'res_node', 'p_node', 'bf_mc_sum', 'c_prod',
                 'sigma_c_prod', 'chi2_prod', 'n_prod', 'prod_pdgids', 'alt_pdgids', 'unclosed_nodes',
                 'twin_pdgids', 'extra_rel', 'dec_lines', 'fed_nodes', 'node_info']


def _flag_twins(results, pdl):
    """`flavor_twin`: a 'none' final state with BF > 0 that equals a measured or limit final state
    of the same block once the K0 / K*0 flavor is ignored, or after charge conjugation. The twin's
    PDG identifiers go to `twin_pdgids`; c is not changed."""
    index = {}
    def ids(r):                                     # entries of a line, or of the node it is the direct line of
        if r['pdg_class'] in ('measured', 'limit') and r['pdgids']:
            return r['pdgids']
        return [r['direct_node']] if r['direct_node'] else []
    for k, r in results.items():
        if ids(r):
            index.setdefault(_loose_key(k), []).append(k)
    for k, r in results.items():
        if r['pdg_class'] != 'none' or r['bf_mc_sum'] <= 0 or r['direct_node']:
            continue
        twins = [t for lk in {_loose_key(k), _loose_key(conjugate_key(k, pdl))}
                 for t in index.get(lk, ()) if t != k]
        if twins:
            r['flags'].add('flavor_twin')
            r['twin_pdgids'] = sorted({p for t in twins for p in ids(results[t])})


def build_mode_pdg_table(mode_table=None, pdg_table=None, exceptions=None, dec_all=None, pdl=None,
                         return_nodes=False):
    """PDG class and correction of every exclusive final state of the four B blocks.

    Parameters (None reads the defaults of the repository)
    ----------
    mode_table : `decay_table.ModeTable` of the four B blocks.
    pdg_table  : output of `load_pdg_table`.
    exceptions : list of exception entries (see the module docstring), or None to read
                 `DEFAULT_EXCEPTIONS`.
    dec_all    : `decay_table.parse_dec(mothers=None)`: every block of the decay file.
    return_nodes : also return the node table (see below).

    Returns a DataFrame with one row per (mother_pdg, key) of `mode_table.modes`:
      names      : daughters in the order of the (first) decay-file line.
      n_daughters, is_sl, n_charm : as in the mode table.
      tau        : a tau among the daughters.
      bf_mc      : BF of this final state in its own block.
      pdg_class  : 'measured', 'limit', 'none' or 'inclusive'.
      how        : 'exact', 'node', 'product', 'isospin' (c of the isospin partner, by exception),
                   'half_limit' (limit/2 +- limit/2, by exception) or ''.
      pdgids     : PDG identifiers used, separated by ';'.
      bf_pdg, bf_pdg_err : value and uncertainty of the PDG node used (for several product nodes
                   the most precise one; for a limit the limit). With the factor 2 of a K_S0 node.
      c          : BF_PDG / BF_MC; 1 for the classes without a measurement.
      sigma_c    : uncertainty of c (c times the relative PDG uncertainty of what was used).
      pull       : (c - 1) / sigma_c.
      f_res_mc, f_res_pdg, res_node : resonant fraction of the node `res_node` in the decay file
                   and in PDG (lower bound); of the nodes the mode feeds, the one with the largest
                   difference.
      flags      : names separated by ';': tau, resonance_mismatch, multi_ks, remainder_unstable,
                   above_limit (max(1, c) x BF_MC above an own limit), exception_applied,
                   block_asymmetric; and more: zero_bf (the line has BF 0 in the decay file;
                   c = 1), c_extreme (c outside [1/3, 3]), limit_capped (kept out of a node
                   remainder by its own limit), node_ambiguous, node_not_closed, flavor_twin
                   (see the module docstring), unmeasured (feeds a measured node but has no c:
                   c = 1, sigma_c = 1), isospin, remainder_clamped (the remainder of the total
                   was negative; c = 0).
      extra_rel  : relative uncertainty added by an extra_uncertainty exception.
      dec_lines  : line numbers of the final state in the decay file (its own block).
      fed_nodes  : measured PDG final-state entries that this line feeds (at least P_MIN of it).
      node_info  : for each of them, 'id: PDG value +- error, decay file sum, after c (pull)',
                   separated by ';'. The full entry, with every feeding line, is in the node report
                   (`node_report`, b_mode_pdg_nodes.csv).
      alt_pdgids : node_ambiguous: the other nodes of the same size the line was free in.
      unclosed_nodes : node_not_closed: the nodes fed by the line that do not close (|pull| > 3).
      twin_pdgids : flavor_twin: PDG identifiers of the twin final state.
      p_node     : share of the final state that feeds the node used (1 for exact).
      bf_mc_sum  : BF_MC used for c: summed over the K_S0 / K_L0 / K0 versions of the mode, in
                   the B0 or B+ block.
      c_prod, sigma_c_prod, chi2_prod, n_prod, prod_pdgids : inverse-variance mean over the
                   product nodes of the mode, stored also when `how` is 'exact' or 'node'.
    `.attrs['pdg_meta']` is the metadata of the PDG table.

    With return_nodes=True the result is (table, nodes). `nodes` has one row per PDG node and
    lepton flavor: uid, pdgid, mother, kind, description, value and err (with the K_S0 factor),
    is_limit, multi_ks, n_bodies, bf_mc (BF_MC of the node: the sum over the lines that feed it;
    for a product node BF_line x BF_sub), ratio (value / bf_mc), n_feeders, role ('exact', 'node',
    'covered' = no direct line was left to fix, 'capped' (unused) = the free lines were all kept out
    by their own limits, 'product', 'duplicate' = product node dropped as a repeat of another,
    'limit', 'multi_ks', 'excluded'), c_assigned, f_res_mc, f_res_pdg, bf_mc_weighted (sum of
    c x BF x P over the feeding lines), closure (bf_mc_weighted / value), closure_pull ((weighted
    - value) / PDG error), reason (why a node is unkeyed).
    """
    pdl = pdl or (mode_table.pdl if mode_table is not None else dt.load_pdl())
    if mode_table is None:
        mode_table = dt.build_mode_table(pdl=pdl)
    if dec_all is None:
        dec_all = dt.parse_dec(mothers=None, pdl=pdl)
    if pdg_table is None:
        pdg_table = load_pdg_table()
    if exceptions is None:
        exceptions = load_exceptions()
    graph = DecayGraph(dec_all, pdl)
    nodes = make_nodes(pdg_table, pdl, exceptions)

    dec_b = mode_table.dec
    results, node_tables = {}, {}
    block_lines = {m: dec_b[(dec_b['mother_pdg'] == m) & ~dec_b['is_pythia']] for m in (511, 521)}
    for mother in (511, 521):
        if len(block_lines[mother]):
            results[mother], node_tables[mother] = _match_block(mother, block_lines[mother], nodes, graph,
                                                                 exceptions)
    # 'isospin' exceptions: the line takes c and sigma_c of its partner in the other block, which
    # must have a c of its own (exact or product); the block of the line is then matched again
    preset = {}
    for e in exceptions:
        if e['action'] != 'isospin':
            continue
        src = results.get(int(e['from_mother']), {}).get(collapse_key(e['from_key']))
        if src is None or src['how'] not in ('exact', 'product'):
            raise ValueError(f'exceptions: isospin source {e["from_key"]} of {e["from_mother"]} has no exact '
                             f'or product c')
        # BF(B+) = TAU_RATIO x BF(B0) for the corrected values; c refers to the line's own BF_MC
        own = results.get(int(e['mother']), {}).get(collapse_key(e['key']))
        if own is None or not own['bf_mc_sum'] > 0:
            raise ValueError(f'exceptions: isospin target {e["key"]} of {e["mother"]} is not a line with BF > 0')
        scale = (1.0 / TAU_RATIO if int(e['mother']) == 511 else TAU_RATIO) * src['bf_mc_sum'] / own['bf_mc_sum']
        preset.setdefault(int(e['mother']), {})[collapse_key(e['key'])] = (src['c'] * scale, src['sigma_c'] * scale,
                                                                             src['pdgids'],
                                                                             src['flags'] & {'measurement_conflict'})
    for mother, pre in preset.items():
        results[mother], node_tables[mother] = _match_block(mother, block_lines[mother], nodes, graph,
                                                             exceptions, preset=pre)
    for mother in results:
        _flag_twins(results[mother], pdl)
    node_tables = [node_tables[m] for m in (511, 521) if m in node_tables]
    loose = {m: {} for m in results}
    for m, res in results.items():
        for k in res:
            loose[m].setdefault(_loose_key(k), []).append(k)
    kind_of = {n['pdgid']: n['kind'] for n in nodes}
    # summed BF of every block per collapsed and per loose key, and the loose keys of the anti-B
    # blocks that hold a final state without a conjugate (alias quirks such as J/psi K*0 in anti-B0)
    csum, lsum, lasym = {}, {}, set()
    for mother, key, bf in zip(mode_table.modes['mother_pdg'], mode_table.modes['key'], mode_table.modes['bf_mc']):
        mother = int(mother)
        ck = (mother, collapse_key(key))
        csum[ck] = csum.get(ck, 0.0) + float(bf)
        lk = (mother, _loose_key(key))
        lsum[lk] = lsum.get(lk, 0.0) + float(bf)
        if mother < 0 and (-mother, conjugate_key(key, pdl)) not in mode_table.lookup:
            lasym.add(lk)

    def differ(a, b):
        return abs(a - b) > 1e-6 * max(abs(a), abs(b))

    rows = []
    modes = mode_table.modes
    for mother, key, bf, nd, is_sl, n_charm, incl, orders, dec_lines in zip(
            modes['mother_pdg'], modes['key'], modes['bf_mc'], modes['n_daughters'], modes['is_sl'],
            modes['n_charm'], modes['is_inclusive'], modes['orders'], modes['lines']):
        mother = int(mother)
        b = abs(mother)
        asym = (-mother, conjugate_key(key, pdl)) not in mode_table.lookup
        r = None
        bf_mismatch = False
        if b in results:
            bkey = key if mother > 0 else conjugate_key(key, pdl)
            if mother > 0 or (b, bkey) in mode_table.lookup:
                r = results[b].get(collapse_key(bkey))
                if mother < 0 and r is not None:
                    # the conjugate exists: is it written with the same BF?
                    own = csum[(mother, collapse_key(key))]
                    lk = (mother, _loose_key(key))
                    alias_quirk = lk in lasym and not differ(lsum[lk], lsum.get((b, _loose_key(bkey)), 0.0))
                    bf_mismatch = differ(own, r['bf_mc_sum']) and not alias_quirk
            else:
                cands = loose[b].get(_loose_key(bkey), [])
                if cands:
                    r = results[b][max(cands, key=lambda k: results[b][k]['bf_mc_sum'])]
        if r is None:
            r = dict(_new_result(), bf_mc_sum=np.nan)
            r.pop('_diff')
        flags = set(r['flags'])
        if asym or bf_mismatch:
            flags.add('block_asymmetric')
        if bf_mismatch and r['how'] and not (r['how'] == 'node' and kind_of.get(r['pdgids'][0]) == 'sum'):
            # a c formed with the BF of the other block does not apply here
            r = dict(r, how='', c=1.0, sigma_c=np.nan, p_node=np.nan, alt_pdgids=[],
                     bf_mc_sum=csum[(mother, collapse_key(key))])
            flags -= {'remainder_unstable', 'remainder_clamped', 'node_ambiguous'}
        tau = any(abs(c) == 15 for c in key)
        if tau:
            flags.add('tau')
        if bf == 0:
            flags.add('zero_bf')
        row = dict(mother_pdg=mother, key=key, names=pdl.names(orders[0]), n_daughters=int(nd),
                   is_sl=bool(is_sl), tau=tau, n_charm=int(n_charm), bf_mc=float(bf),
                   dec_lines=' '.join(str(x) for x in sorted(dec_lines)))
        row.update({c: r[c] for c in ('pdg_class', 'how', 'bf_pdg', 'bf_pdg_err', 'c', 'sigma_c',
                                      'f_res_mc', 'f_res_pdg', 'res_node', 'p_node', 'bf_mc_sum',
                                      'c_prod', 'sigma_c_prod', 'chi2_prod', 'n_prod', 'extra_rel')})
        row['pdgids'] = ';'.join(r['pdgids'])
        for c in ('prod_pdgids', 'alt_pdgids', 'unclosed_nodes', 'twin_pdgids', 'fed_nodes', 'node_info'):
            row[c] = ';'.join(r[c])
        if incl:
            row.update(pdg_class='inclusive', how='', c=1.0, sigma_c=np.nan)
        if row['how'] and not (1.0 / C_EXTREME <= row['c'] <= C_EXTREME):
            flags.add('c_extreme')
        s, c = row['sigma_c'], row['c']
        row['pull'] = (c - 1.0) / s if row['how'] and np.isfinite(s) and s > 0 else np.nan
        row['flags'] = ';'.join(sorted(flags))
        rows.append(row)
    out = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    out.attrs['pdg_meta'] = dict(pdg_table.get('meta', {}))
    if return_nodes:
        return out, (pd.concat(node_tables, ignore_index=True) if node_tables else pd.DataFrame())
    return out


def node_report(nodes, table, pdl=None):
    """One row per measured or limit PDG entry of B0 / B+ that some decay-file line feeds: the
    reference for the columns `fed_nodes`, `node_info` and `unclosed_nodes` of the mode table.

    Columns: pdgid, uid (with [e] / [mu] for lepton entries), mother, description, kind, role,
    value, err, is_limit, n_bodies, bf_mc (decay-file sum), bf_mc_weighted (after c), closure,
    closure_pull, c_assigned, n_feeders, and `feeders`: one text item per feeding line,
    'names [decay-file lines] BFxP=...; class/how c=... +- ...', separated by ' | '."""
    pdl = pdl or dt.load_pdl()
    look = {}
    for r in table.itertuples():
        if int(r.mother_pdg) in (511, 521):
            look.setdefault((int(r.mother_pdg), collapse_key(r.key)), r)
    out = []
    for n in nodes.itertuples():
        feeders = n.feeders if isinstance(n.feeders, list) else []
        if int(n.mother) not in (511, 521) or not feeders:
            continue
        items = []
        for k, v, bf in feeders:
            r = look.get((int(n.mother), k))
            name = ' '.join('K0' if c == KSYM else pdl.name(c) for c in k)
            if r is None:
                items.append(f'{name} BFxP={v:.3g}')
                continue
            sig = '' if not np.isfinite(r.sigma_c) else f' +- {r.sigma_c:.2g}'
            items.append(f'{name} [{r.dec_lines}] BFxP={v:.3g}; {r.pdg_class}/{r.how or "-"} c={r.c:.3g}{sig}')
        out.append(dict(pdgid=n.pdgid, uid=n.uid, mother=int(n.mother), description=n.description,
                        kind=n.kind, role=n.role, value=n.value, err=n.err, is_limit=n.is_limit,
                        n_bodies=n.n_bodies, bf_mc=n.bf_mc, bf_mc_weighted=n.bf_mc_weighted,
                        closure=n.closure, closure_pull=n.closure_pull, c_assigned=n.c_assigned,
                        n_feeders=len(feeders), feeders=' | '.join(items)))
    return pd.DataFrame(out)


def table_lookup(table):
    """{(mother_pdg, key): row position} of a mode table, for repeated lookups."""
    return {(int(m), tuple(k)): i for i, (m, k) in enumerate(zip(table['mother_pdg'], table['key']))}


def pdg_info(table, mother_pdg, key, lookup=None):
    """Row of the mode table for a final state (Series), or None.

    key : PDG codes of the daughters, in any order. lookup : `table_lookup(table)`, to avoid
    rebuilding it at every call."""
    lookup = table_lookup(table) if lookup is None else lookup
    i = lookup.get((int(mother_pdg), dt.mode_key(key)))
    return None if i is None else table.iloc[i]


def has_flag(table, flag):
    """Boolean array: rows of a mode or review table that carry `flag`."""
    return np.array([flag in str(f).split(';') for f in table['flags']], dtype=bool)


def write_table(table, path):
    """Write a mode or review table as CSV; the key becomes a string of PDG codes."""
    out = table.copy()
    out['key'] = [' '.join(str(c) for c in k) for k in out['key']]
    out.to_csv(path, index=False, float_format='%.6g')


def read_table(path):
    """Read a table written by `write_table` (the key is a tuple again)."""
    out = pd.read_csv(path, keep_default_na=False, na_values=[''])
    out['key'] = [tuple(int(c) for c in str(k).split()) for k in out['key']]
    for c in ('how', 'pdgids', 'flags', 'res_node', 'prod_pdgids', 'tables_top95', 'alt_pdgids',
              'unclosed_nodes', 'twin_pdgids', 'dec_lines', 'fed_nodes', 'node_info'):
        if c in out:
            out[c] = out[c].fillna('')
    return out


# --------------------------------------------------------------------------------------------
# Review report
# --------------------------------------------------------------------------------------------
def _region(tag):
    if '_wc_' in tag or tag.endswith('_wc'):
        return 'WC'
    if tag.endswith('_SR'):
        return 'SR'
    if tag.endswith('_combCR'):
        return 'combCR'
    return tag


SUMMARY_CLASSES = ['excl_sl', 'had_exact', 'had_node', 'had_product', 'had_isospin', 'had_limit', 'had_none',
                   'inclusive', 'pythia_0c', 'pythia_1c', 'pythia_2c']


def review_report(table, sample_csvs=None, mode_table=None):
    """Modes to check by hand, and the composition of the samples by PDG class.

    table : output of `build_mode_pdg_table`.
    sample_csvs : 'bbbar_modes_<tag>.csv' files; None takes all of them in `CSV_DIR`.

    Returns (review, summary).
    review : one row per exclusive final state that is among the modes covering 95 % of the
        exclusive hadronic yield of some table (file, component, side, B type), or has
        |pull| > 3, or carries a flag (other than zero_bf). Conjugate final states are merged into the B0 / B+ row.
        Columns: the main columns of the mode table, and
          max_share     : largest share of the exclusive yield over all tables,
          max_share_table : that table,
          n_top95       : number of tables in which the mode is among the top 95 % hadronic modes,
          why           : 'top95', 'pull', 'flag' (several joined by ';').
        Sorted by max_share.
    summary : one row per (region, component, side) with the yield in percent by class
        (`SUMMARY_CLASSES`), e and mu, run1 and run2, charged and neutral B summed. Weight: sum_w.
    """
    if sample_csvs is None:
        sample_csvs = sorted(glob.glob(os.path.join(CSV_DIR, 'bbbar_modes_*.csv')))
    mode_table = mode_table or dt.build_mode_table()
    pdl = mode_table.pdl
    lookup = {(int(m), tuple(k)): i for i, (m, k) in enumerate(zip(table['mother_pdg'], table['key']))}
    cls, how = table['pdg_class'].to_numpy(), table['how'].to_numpy()

    max_share, max_table, n_top = {}, {}, {}
    summary = {}
    for path in sample_csvs:
        tag = os.path.basename(path)[len('bbbar_modes_'):-len('.csv')]
        region = _region(tag)
        for (comp, side, b_abs), t in dt.load_mode_tables(path).items():
            raw = t[dt.RAW_COLUMNS].to_numpy()
            res = dt.classify_decays(raw, b_abs, mode_table)
            w = t['sum_w'].to_numpy(dtype=float)
            acc = summary.setdefault((region, comp, side), dict.fromkeys(SUMMARY_CLASSES, 0.0))
            y_all, y_had = {}, {}
            gen, fl, nfsr = res['gen_class'].to_numpy(), res['b_flavor'].to_numpy(), res['n_fsr'].to_numpy()
            n_charm = res['n_charm'].to_numpy()
            for i in range(len(t)):
                if gen[i] != dt.GEN_EXCLUSIVE:
                    acc[f'pythia_{min(int(n_charm[i]), 2)}c'] += w[i]
                    continue
                d = [int(x) for x in raw[i] if int(x) != dt.EMPTY]
                key = dt.mode_key(d[:len(d) - int(nfsr[i])])
                flavors = (int(fl[i]),) if fl[i] != 0 else (b_abs, -b_abs)
                j = next((lookup[(f, key)] for f in flavors if (f, key) in lookup), None)
                if j is None:
                    raise ValueError(f'{path}: exclusive decay {key} is not in the mode table')
                if cls[j] == 'inclusive':
                    c = 'inclusive'
                elif table['is_sl'].iat[j]:
                    c = 'excl_sl'
                else:
                    # 'measured' without a c (BF 0, or BF differs from the conjugate block): no c
                    c = {'measured': 'had_' + (how[j] or 'none'), 'limit': 'had_limit', 'none': 'had_none'}[cls[j]]
                acc[c] += w[i]
                # merge the conjugate into the B0 / B+ row
                m = int(table['mother_pdg'].iat[j])
                if m < 0:
                    j = lookup.get((-m, conjugate_key(key, pdl)), j)
                y_all[j] = y_all.get(j, 0.0) + w[i]
                if c.startswith('had_'):
                    y_had[j] = y_had.get(j, 0.0) + w[i]
            tot = sum(y_all.values())
            name = f'{tag}:{comp}:{side}:{b_abs}'
            for j, y in y_all.items():
                if tot > 0 and y / tot > max_share.get(j, -1.0):
                    max_share[j], max_table[j] = y / tot, name
            tot_had, run = sum(y_had.values()), 0.0
            for j, y in sorted(y_had.items(), key=lambda kv: -kv[1]):
                if tot_had <= 0 or run >= YIELD_COVER * tot_had:
                    break
                run += y
                n_top[j] = n_top.get(j, 0) + 1

    rows = []
    pull = table['pull'].to_numpy()
    flags = table['flags'].to_numpy()
    for j in range(len(table)):
        why = []
        if j in n_top:
            why.append('top95')
        if np.isfinite(pull[j]) and abs(pull[j]) > PULL_REVIEW:
            why.append('pull')
        if set(flags[j].split(';')) - {'', 'zero_bf'}:
            why.append('flag')
        if not why:
            continue
        m = int(table['mother_pdg'].iat[j])
        if m < 0 and (-m, conjugate_key(table['key'].iat[j], pdl)) in lookup and j not in max_share:
            continue                                   # listed with its B0 / B+ conjugate
        r = table.iloc[j]
        rows.append(dict(mother_pdg=m, key=r['key'], names=r['names'], pdg_class=r['pdg_class'],
                         how=r['how'], pdgids=r['pdgids'], bf_mc=r['bf_mc'], bf_pdg=r['bf_pdg'],
                         c=r['c'], sigma_c=r['sigma_c'], pull=r['pull'], f_res_mc=r['f_res_mc'],
                         f_res_pdg=r['f_res_pdg'], c_prod=r['c_prod'], flags=r['flags'],
                         alt_pdgids=r['alt_pdgids'], unclosed_nodes=r['unclosed_nodes'],
                         twin_pdgids=r['twin_pdgids'],
                         max_share=max_share.get(j, 0.0), max_share_table=max_table.get(j, ''),
                         n_top95=n_top.get(j, 0), why=';'.join(why)))
    review = pd.DataFrame(rows)
    if len(review):
        review = review.sort_values(['max_share', 'bf_mc'], ascending=False, kind='stable').reset_index(drop=True)

    srows = []
    for (region, comp, side), acc in summary.items():
        tot = sum(acc.values())
        srows.append(dict(region=region, component=comp, side=side, sum_w=tot,
                          **{c: 100.0 * acc[c] / tot if tot > 0 else np.nan for c in SUMMARY_CLASSES}))
    order = {'SR': 0, 'combCR': 1, 'WC': 2}
    summary = pd.DataFrame(srows)
    if len(summary):
        summary = summary.sort_values(by=['region', 'component', 'side'],
                                      key=lambda s: s.map(order) if s.name == 'region' else s,
                                      kind='stable').reset_index(drop=True)
    return review, summary


# --------------------------------------------------------------------------------------------
# Isospin partners
# --------------------------------------------------------------------------------------------
def isospin_partner_code(code):
    """PDG code of the particle with u and d exchanged, or None if the rule does not cover it
    (baryons, unknown codes). Leptons, the photon, neutral light mesons, s-sbar, charmonium and
    D_s states are their own partners; a charged light meson changes sign; a meson with one
    u or d and one s, c or b quark changes that light quark (K+ <-> K0, D+ <-> D0, ...)."""
    a = abs(int(code))
    if a in (11, 12, 13, 14, 15, 16, 22):
        return int(code)
    if a < 100 or (a // 1000) % 10 != 0:               # not a meson
        return None
    q2, q3 = (a // 100) % 10, (a // 10) % 10
    if q2 in (3, 4, 5) and q3 in (1, 2):
        return int(np.sign(code)) * (a + (10 if q3 == 1 else -10))
    if q2 in (1, 2) and q3 in (1, 2):
        return -int(code) if q2 != q3 else int(code)
    if q2 == q3 or (q2 in (4, 5) and q3 == 3):         # s-sbar, c-cbar, b-bbar, D_s, B_s
        return int(code)
    return None


def isospin_partner_keys(key, pdl, charge3):
    """Collapsed keys that are the u <-> d image of a collapsed final state and have the total
    charge `charge3` (in units of e/3). The collapsed neutral kaon (K0, anti-K0, K_S0, K_L0) can
    come from a K+ or a K-, so there may be more than one candidate."""
    options = []
    for c in key:
        if c == KSYM:
            options.append((321, -321))
            continue
        p = isospin_partner_code(c)
        if p is None or p not in pdl.charge3:
            return []
        options.append((KSYM,) if abs(p) == 311 else (p,))
    out = set()
    stack = [()]
    for opt in options:
        stack = [s + (o,) for s in stack for o in opt]
    for cand in stack:
        if sum(0 if c == KSYM else pdl.charge3[c] for c in cand) == charge3:
            out.add(tuple(sorted(cand)))
    return sorted(out)


def isospin_report(table, pdl=None):
    """Pairs of B0 and B+ lines related by isospin, with their PDG comparison.

    Covered are the transitions that conserve isospin, where the partner rates are equal up to
    the lifetime ratio: semileptonic decays to charm (b -> c l nu) and b -> c cbar s decays (two
    charm hadrons or a charmonium, with a D_s or a kaon). The partner is the final state with u
    and d exchanged. One row per pair:
      names_b0, names_bp, bf_mc_b0, bf_mc_bp : the two lines and their decay-file BF (summed
                 over the neutral-kaon versions);
      r_mc     : BF_MC(B+) / BF_MC(B0) / TAU_RATIO; 1 if the decay file respects isospin;
      how_b0, c_b0, sigma_b0, how_bp, c_bp, sigma_bp : the PDG comparison of each line;
      status   : 'both' (each has an exact, product or node c of its own), 'only_b0',
                 'only_bp', 'neither';
      r_pdg, r_pdg_err : for 'both', (c x BF_MC)(B+) / (c x BF_MC)(B0) / TAU_RATIO, i.e. the
                 ratio of the PDG values over the isospin expectation, and its uncertainty;
      pull     : (BF(B+) - TAU_RATIO x BF(B0)) / uncertainty of the difference, with the PDG
                 values as c x BF_MC (the ratio is not Gaussian when one value is poorly known);
      c_other  : for 'only_b0' / 'only_bp', the c the unmeasured line would take by isospin;
      kind     : 'sl' or 'ccs'.
    """
    pdl = pdl or dt.load_pdl()
    rows = {m: {} for m in (511, 521)}
    for r in table.itertuples():
        if r.mother_pdg in rows and r.bf_mc > 0 and r.pdg_class != 'inclusive':
            rows[r.mother_pdg].setdefault(collapse_key(r.key), r)

    def kind(r):
        key = [c for c in r.key]
        if r.is_sl:
            return 'sl' if r.n_charm >= 1 else ''
        charm = sum(dt.is_open_charm(c) for c in key) >= 2 or any(dt.is_charmonium(c) for c in key)
        strange = any((abs(c) // 1000) % 10 == 0 and abs(c) > 100 and
                      {(abs(c) // 100) % 10, (abs(c) // 10) % 10} in ({3, 1}, {3, 2}, {4, 3}) for c in key)
        return 'ccs' if charm and strange else ''

    own = ('exact', 'product', 'node')
    out = []
    for k0, r0 in rows[511].items():
        kd = kind(r0)
        if not kd:
            continue
        partners = [k for k in isospin_partner_keys(k0, pdl, 3) if k in rows[521]]
        if len(partners) != 1:
            continue
        rp = rows[521][partners[0]]
        m0, mp = r0.how in own, rp.how in own
        rec = dict(kind=kd, names_b0=r0.names, names_bp=rp.names, bf_mc_b0=r0.bf_mc_sum, bf_mc_bp=rp.bf_mc_sum,
                   r_mc=rp.bf_mc_sum / r0.bf_mc_sum / TAU_RATIO, how_b0=r0.how, c_b0=r0.c, sigma_b0=r0.sigma_c,
                   how_bp=rp.how, c_bp=rp.c, sigma_bp=rp.sigma_c,
                   status='both' if m0 and mp else 'only_b0' if m0 else 'only_bp' if mp else 'neither',
                   r_pdg=np.nan, r_pdg_err=np.nan, pull=np.nan, c_other=np.nan)
        if m0 and mp:
            bp, b0 = rp.c * rp.bf_mc_sum, r0.c * r0.bf_mc_sum * TAU_RATIO
            sig_p, sig_0 = rp.sigma_c, r0.sigma_c
            common = min(getattr(rp, 'extra_rel', 0.0) or 0.0, getattr(r0, 'extra_rel', 0.0) or 0.0)
            if common > 0:                          # an extra uncertainty common to both lines cancels
                sig_p = np.sqrt(max(sig_p ** 2 - (common * rp.c) ** 2, 0.0))
                sig_0 = np.sqrt(max(sig_0 ** 2 - (common * r0.c) ** 2, 0.0))
            sp, s0 = sig_p * rp.bf_mc_sum, sig_0 * r0.bf_mc_sum * TAU_RATIO
            ratio = bp / b0 if b0 > 0 else np.nan
            err = ratio * np.hypot(sp / bp, s0 / b0) if bp > 0 and b0 > 0 else np.nan
            rec.update(r_pdg=ratio, r_pdg_err=err,
                       pull=(bp - b0) / np.hypot(sp, s0) if np.hypot(sp, s0) > 0 else np.nan)
        elif m0 or mp:
            src, dst = (r0, rp) if m0 else (rp, r0)
            # c that reproduces the isospin image of the measured BF, given the decay file
            scale = TAU_RATIO if m0 else 1.0 / TAU_RATIO
            rec['c_other'] = src.c * src.bf_mc_sum * scale / dst.bf_mc_sum
        out.append(rec)
    return pd.DataFrame(out)


def class_shares(table, mother_pdg):
    """Summed BF_MC of one B block by class and by `how` (fractions of all decays of that B)."""
    t = table[(table['mother_pdg'] == mother_pdg) & (table['bf_mc'] > 0)]
    label = np.where(t['pdg_class'] == 'measured', 'measured_' + t['how'], t['pdg_class'])
    return t.groupby(label)['bf_mc'].sum().to_dict()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--rebuild-pdg' in argv or not os.path.exists(DEFAULT_PDG_JSON):
        build_pdg_table(DEFAULT_PDG_JSON)
        print(f'wrote {DEFAULT_PDG_JSON}')
    pdl = dt.load_pdl()
    mode_table = dt.build_mode_table(pdl=pdl)
    table, nodes = build_mode_pdg_table(mode_table, pdl=pdl, return_nodes=True)
    write_table(table, DEFAULT_TABLE_CSV)
    print(f'wrote {DEFAULT_TABLE_CSV}: {len(table)} final states')
    nrep = node_report(nodes, table, pdl)
    nrep.to_csv(DEFAULT_NODES_CSV, index=False, float_format='%.6g')
    print(f'wrote {DEFAULT_NODES_CSV}: {len(nrep)} PDG entries fed by decay-file lines')
    for m in (511, 521):
        sh = class_shares(table, m)
        print(f'  {pdl.name(m)}: BF by class: ' + ', '.join(f'{k} {100 * v:.2f} %' for k, v in sorted(sh.items())))
    unk = nodes[(nodes['kind'] == 'unkeyed') & (nodes['value'] > 1e-4) & ~nodes['is_limit']]
    print(f'  PDG nodes: {nodes["pdgid"].nunique()}, by kind: {nodes.groupby("kind")["pdgid"].nunique().to_dict()}; '
          f'unkeyed with a value above 1e-4: {len(unk)}')
    review, summary = review_report(table, mode_table=mode_table)
    write_table(review, DEFAULT_REVIEW_CSV)
    print(f'wrote {DEFAULT_REVIEW_CSV}: {len(review)} modes to review '
          f'({int((review["n_top95"] > 0).sum())} in the top 95 % of some table)')
    with pd.option_context('display.width', 250, 'display.float_format', '{:.1f}'.format):
        print(summary.to_string(index=False))
    iso = isospin_report(table, pdl)
    iso.to_csv(DEFAULT_ISOSPIN_CSV, index=False, float_format='%.6g')
    print(f'wrote {DEFAULT_ISOSPIN_CSV}: {len(iso)} isospin pairs, by status: '
          f'{iso.groupby("status").size().to_dict()}')
    return table, review, summary


if __name__ == '__main__':
    main()
