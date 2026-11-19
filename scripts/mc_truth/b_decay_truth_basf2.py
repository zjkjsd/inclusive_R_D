#!/usr/bin/env python3
"""Generator truth of the B decays in official generic MC (piece 3, point 3.2 of the BBbar model).

Reads the MCParticles of existing MC16rd generic mdst files (no generation, no simulation, no
reconstruction) and writes one CSV row per B meson decay:

    exp, run, evt, b_pdg, daughters, n_photos, <counts of charm hadrons in the decay chain>

* `daughters`: PDG codes of the first-generation daughters in generator order, PHOTOS photons
  removed (space separated); `n_photos` counts the removed photons. This is the list that
  `scripts/tools/decay_table.classify_decays` uses to tell exclusive decay-file lines from PYTHIA.
* Charm counts are over the whole decay chain of that B (primary particles only), by species and
  charge sign, for ground-state charm hadrons and charmonium:
  D0, D0bar, Dp, Dm, Dsp, Dsm, Lcp, Lcm (Lambda_c), Xic (any Xi_c / Omega_c, signed sum of +/-),
  jpsi, psi2s, chic0, chic1, chic2, etac, and n_charm_quarks (net count of c and cbar quarks in the
  ground-state hadrons, charmonium counted as 2).
* A B that oscillated is written once, at the B that decays (a B whose only daughter is a B is
  skipped). `b_pdg` is the flavor at decay.

Usage (basf2 release-08 environment, e.g. on KEKCC):

    basf2 b_decay_truth_basf2.py -- --out truth_mixed.csv  file1.mdst.root file2.mdst.root ...
    basf2 b_decay_truth_basf2.py -- --out truth_charged.csv --max-events 100000 'path/*.mdst.root'

About 1e5 events each of mixed (B0 B0bar) and charged (B+ B-) generic MC is enough.

Written 2026-11-19 without access to basf2: NOT TESTED. Run it on one file with --max-events 100
first and check the CSV by eye (see the checks printed at the end).
"""
import argparse
import csv
import glob

import basf2
from ROOT import Belle2

B_CODES = {511, 521}
CHARM = {421: 'D0', -421: 'D0bar', 411: 'Dp', -411: 'Dm', 431: 'Dsp', -431: 'Dsm',
         4122: 'Lcp', -4122: 'Lcm'}
XIC = {4132, 4232, 4332}                       # Xi_c0, Xi_c+, Omega_c0 (signed)
CHARMONIUM = {443: 'jpsi', 100443: 'psi2s', 10441: 'chic0', 20443: 'chic1', 445: 'chic2', 441: 'etac'}
COLUMNS = ['exp', 'run', 'evt', 'b_pdg', 'daughters', 'n_photos'] + sorted(set(CHARM.values())) + \
    ['Xic', 'jpsi', 'psi2s', 'chic0', 'chic1', 'chic2', 'etac', 'n_charm_quarks']
PHOTOS = Belle2.MCParticle.c_IsPHOTOSPhoton


class BDecayTruth(basf2.Module):
    def __init__(self, out_path):
        super().__init__()
        self.out_path = out_path
        self.n_b = 0

    def initialize(self):
        self.mcp = Belle2.PyStoreArray('MCParticles')
        self.meta = Belle2.PyStoreObj('EventMetaData')
        self.f = open(self.out_path, 'w', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=COLUMNS)
        self.w.writeheader()

    def _chain_counts(self, b):
        counts = dict.fromkeys(COLUMNS[6:], 0)
        stack = [d for d in b.getDaughters()]
        while stack:
            p = stack.pop()
            if not p.isPrimaryParticle():           # Geant4 secondaries
                continue
            code = p.getPDG()
            if code in CHARM:
                counts[CHARM[code]] += 1
                counts['n_charm_quarks'] += 1
            elif abs(code) in XIC:
                counts['Xic'] += 1 if code > 0 else -1
                counts['n_charm_quarks'] += 1
            elif code in CHARMONIUM:
                counts[CHARMONIUM[code]] += 1
                counts['n_charm_quarks'] += 2
            stack.extend(d for d in p.getDaughters())
        return counts

    def event(self):
        meta = self.meta.obj()
        for p in self.mcp:
            if abs(p.getPDG()) not in B_CODES or not p.isPrimaryParticle():
                continue
            daus = [d for d in p.getDaughters()]
            if len(daus) == 1 and abs(daus[0].getPDG()) in B_CODES:   # mixing: keep the decaying B
                continue
            kept = [d.getPDG() for d in daus if not d.hasStatus(PHOTOS)]
            row = dict(exp=meta.getExperiment(), run=meta.getRun(), evt=meta.getEvent(),
                       b_pdg=p.getPDG(), daughters=' '.join(map(str, kept)),
                       n_photos=len(daus) - len(kept))
            row.update(self._chain_counts(p))
            self.w.writerow(row)
            self.n_b += 1

    def terminate(self):
        self.f.close()
        basf2.B2INFO(f'b_decay_truth: wrote {self.n_b} B decays to {self.out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', help='mdst files or glob patterns')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-events', type=int, default=0)
    a = ap.parse_args()
    files = sorted(f for pat in a.files for f in (glob.glob(pat) or [pat]))
    path = basf2.create_path()
    path.add_module('RootInput', inputFileNames=files, branchNames=['MCParticles', 'EventMetaData'])
    path.add_module(BDecayTruth(a.out))
    basf2.process(path, max_event=a.max_events)
    print(basf2.statistics)
    print('Checks: about 2 B rows per event; in mixed MC b_pdg is +-511, in charged MC +-521; '
          'daughters lists like "-413 211" (B0 -> D*- pi+); n_photos mostly 0 or 1.')


if __name__ == '__main__':
    main()
