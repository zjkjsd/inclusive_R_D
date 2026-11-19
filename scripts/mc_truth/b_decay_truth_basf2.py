# -*- coding: utf-8 -*-
"""Generator truth of the B decays in official generic MC (piece 3, point 3.2 of the BBbar model).

Reads only the MC truth (MCParticles) of MC16rd generic mdst files: no generation, no simulation,
no reconstruction. Writes one ROOT file with two trees:

* trees "B0" and "Bplus": one entry per generated B0 / B+ (and conjugates) that decays,
    PDG, genParticleID (MCParticles array index), nMCDaughters,
    d{i}_PDG and d{i}_photos for i = 0 .. N_DAU-1: the first-generation daughters in generator order
    and whether each is a PHOTOS photon (-1 when the B has fewer daughters).
  The daughter list is what `scripts/tools/decay_table.classify_decays` uses to tell exclusive
  decay-file lines from PYTHIA (PHOTOS photons removed).
* one tree per charm species ("charm_D0", "charm_Dplus", "charm_D_splus", ...): one entry per
    generated D0, D+, D_s+, Lambda_c+, Xi_c0, Xi_c+, Omega_c0, J/psi, psi(2S), chi_c0, chi_c1, chi_c2,
    eta_c (and antiparticles), with PDG and ancestorBIndex (array index of its first B ancestor, -1 if
    none). Joining ancestorBIndex to genParticleID of the B trees (same event) gives the charm
    content of each B decay chain.

Event identifiers (__experiment__, __run__, __event__) are added by variablesToNtuple.

Usage, locally (basf2 light-2609-luna):
    basf2 b_decay_truth_basf2.py -i <mdst file(s)> -o truth_mixed.root -n 100
On the grid (gbasf2): the input dataset is given to gbasf2 (it overrides the input files); the output
file name is the -o argument (default b_decay_truth.root).

About 1e5 events each of mixed and charged MC16rd generic MC are enough.
Written 2026-11-19 against the light-2609-luna sources (modularAnalysis.fillParticleListFromMC,
variables mcDaughter, nMCDaughters, mcPhotos, genParticleID, ancestorBIndex); not run here (no basf2
in this environment): test on a few events first.
"""
import argparse

import basf2 as b2
import modularAnalysis as ma
from variables import variables as vm

N_DAU = 12          # first-generation daughter slots (nMCDaughters tells whether more exist)

CHARM_LISTS = ['D0:gen', 'D+:gen', 'D_s+:gen', 'Lambda_c+:gen', 'Xi_c0:gen', 'Xi_c+:gen', 'Omega_c0:gen',
               'J/psi:gen', 'psi(2S):gen', 'chi_c0:gen', 'chi_c1:gen', 'chi_c2:gen', 'eta_c:gen']


def argparser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', nargs='*', default=[], help='input mdst file(s); empty on the grid')
    parser.add_argument('-o', '--output', default='b_decay_truth.root', help='output ROOT file')
    parser.add_argument('-n', '--max-events', type=int, default=0, help='events to process (0 = all)')
    return parser


if __name__ == '__main__':
    args = argparser().parse_args()
    main_path = b2.Path()
    ma.inputMdstList(filelist=args.input, path=main_path)

    # ---- generated B mesons (charge conjugates included); a B whose only daughter is a B (mixing
    # bookkeeping, if present) is skipped, so that each decay is written once, at the decaying B
    for blist in ('B0:gen', 'B+:gen'):
        ma.fillParticleListFromMC(blist, 'nMCDaughters > 1', skipNonPrimary=True, path=main_path)

    dau_vars = []
    for i in range(N_DAU):
        vm.addAlias(f'd{i}_PDG', f'ifNANgiveX(mcDaughter({i}, PDG), -1)')
        vm.addAlias(f'd{i}_photos', f'ifNANgiveX(mcDaughter({i}, mcPhotos), -1)')
        dau_vars += [f'd{i}_PDG', f'd{i}_photos']
    b_vars = ['PDG', 'genParticleID', 'nMCDaughters'] + dau_vars

    # ---- generated charm hadrons and charmonium, linked to their B ancestor
    charm_vars = ['PDG', 'genParticleID', 'ancestorBIndex']
    for clist in CHARM_LISTS:
        ma.fillParticleListFromMC(clist, '', skipNonPrimary=True, path=main_path)
        tree = 'charm_' + clist.split(':')[0].replace('+', 'plus').replace('/', '').replace('(', '').replace(')', '')
        ma.variablesToNtuple(clist, charm_vars, treename=tree, filename=args.output, path=main_path)

    ma.variablesToNtuple('B0:gen', b_vars, treename='B0', filename=args.output, path=main_path)
    ma.variablesToNtuple('B+:gen', b_vars, treename='Bplus', filename=args.output, path=main_path)

    b2.process(path=main_path, max_event=args.max_events)
    print(b2.statistics)
