#!/usr/bin/env python3
"""Sanity checks of the output of b_decay_truth_basf2.py (piece 3, point 3.2).

Usage (any environment with uproot, numpy, pandas, e.g. the basf2 light-2609-luna environment):
    python3 check_truth_file.py test_mixed.root [test_charged.root ...]

Prints, per file:
  * entries per tree, B entries per event (expected: 2 in mixed or charged MC, B0-type in mixed,
    B+-type in charged),
  * the B PDG codes found, the range of nMCDaughters (expected >= 2: this tests the mixing question),
    how many B's have more daughters than the 12 stored slots,
  * the PHOTOS flags found (expected 0 and 1, -1 for empty slots),
  * for the charm trees: how many entries have a B ancestor, and whether every ancestorBIndex is the
    genParticleID of a B of the same event (expected: all; this tests the index question),
  * a few example B decays.
"""
import sys

import numpy as np
import pandas as pd
import uproot

EVT = ['__experiment__', '__run__', '__event__']


def check(path):
    print(f'\n===== {path}')
    f = uproot.open(path)
    trees = {k.split(';')[0]: f[k] for k in f.keys() if f[k].classname.startswith('TTree')}
    for name, t in trees.items():
        print(f'  tree {name:18s} {t.num_entries:9d} entries')
    bs = []
    for name in ('B0', 'Bplus'):
        if name in trees:
            df = trees[name].arrays(library='pd')
            df['tree'] = name
            bs.append(df)
    b = pd.concat(bs, ignore_index=True)
    per_evt = b.groupby(EVT).size()
    print(f'  B entries per event: {per_evt.value_counts().sort_index().to_dict()}')
    print(f'  B PDG codes: {b.PDG.astype(int).value_counts().to_dict()}')
    print(f'  nMCDaughters: min {int(b.nMCDaughters.min())}, max {int(b.nMCDaughters.max())}; '
          f'with fewer than 2: {(b.nMCDaughters < 2).sum()}; more than 12 (truncated): {(b.nMCDaughters > 12).sum()}')
    ph = pd.concat([b[f'd{i}_photos'] for i in range(12)]).astype(int).value_counts().to_dict()
    print(f'  PHOTOS flags over all slots: {ph}')
    keys = set(zip(*[b[c] for c in EVT], b.genParticleID.astype(int)))
    for name, t in trees.items():
        if not name.startswith('charm_'):
            continue
        c = t.arrays(library='pd')
        if not len(c):
            print(f'  {name}: empty')
            continue
        has_b = c.ancestorBIndex >= 0
        found = [k in keys for k in zip(*[c[col] for col in EVT], c.ancestorBIndex.astype(int))]
        found = np.array(found) & has_b.to_numpy()
        print(f'  {name:18s} {len(c):8d} entries, with a B ancestor {has_b.sum():8d}, '
              f'ancestor found among the B trees {found.sum():8d} (should equal the previous number)')
    print('  example B decays (PDG: daughters, PHOTOS photons marked with *):')
    for _, r in b.head(8).iterrows():
        d = [f"{int(r[f'd{i}_PDG'])}{'*' if r[f'd{i}_photos'] == 1 else ''}"
             for i in range(12) if r[f'd{i}_PDG'] != -1]
        print(f'    {int(r.PDG):6d}: {" ".join(d)}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        check(p)
