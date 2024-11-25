# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool to prepare BDT training signal/bkg root files.

Usage: python3 2_Prepare_Training_Samples.py -d folder -i -o -l e (currently only support e)
                --mctype --reweight

Examples: python3 2_Prepare_Training_Samples.py -d Samples/Signal_MC14ri/MC14ri_sigDDst_bengal_e_2 \
         -i sigDDst_bengal_e_2_0.parquet -o MC14ri_signal.root -l e --mctype signal (--reweight 10)
         
         python3 2_Prepare_Training_Samples.py -d Samples/Generic_MC14ri/MC14ri_bbbar_bengal_e_1 \
         -i MC14ri_bbbar_0.root MC14ri_bbbar_1.root MC14ri_bbbar_2.root MC14ri_bbbar_3.root \
         MC14ri_bbbar_4.root -o MC14ri_DTCFake.root -l e --mctype bkg_DTC
"""

import argparse
from termcolor import colored
import root_pandas
import uproot
import pandas
import utilities as util

def argparser():
    """
    Parse options as command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    
    parser.add_argument('-d', "--dir",
                        action="store",
                        type=str,
                        default=None,
                        required=True,
                        help="relative path to the input root files")
    parser.add_argument('-i',"--input",
                        action="store", # if append, input = [ [-i 1 2], [-i 3 4] ]
                        nargs='+',
                        required=True,
                        help="List of input parquet or root files in the --dir filder")
    parser.add_argument('-o', "--output",
                        action="store",
                        type=str,
                        required=True,
                        help="Name of output files, parquet or root")
    parser.add_argument('-l', "--lmode",
                        action="store",
                        type=str,
                        required=True,
                        choices=['e', 'mu'],
                        help="Lepton mode, e or mu")
    parser.add_argument("--mctype",
                        action="store",
                        type=str,
                        required=True,
                        choices=['signal','bkg_DTC', 'bkg_B', 'bkg_continuum'],
                        help="Type of the training sample")
    parser.add_argument("--sampling",
                        action="store",
                        type=int,
                        required=False,
                        help="Randomly sample the data")
    parser.add_argument("--reweight",
                        action="store",
                        type=int,
                        required=False,
                        help="Replace the __weight__ column")

    return parser


if __name__ == "__main__":

    args = argparser().parse_args()
    # read parquets as pandas dataframe
    samples = []

    Dstst_e_nu_selection = f'DecayMode=={util.DecayMode["all_Dstst_e_nu"].value} and \
                        D_mcPDG*e_mcPDG==411*11 and e_genMotherPDG==B0_mcPDG and \
        ((B0_mcErrors<64 and B0_mcPDG*D_mcPDG==-511*411) or (B0_mcErrors<512 and abs(B0_mcPDG)==521))'
    
    Dstst_tau_nu_selection = f'DecayMode=={util.DecayMode["all_Dstst_tau_nu"].value} and \
                        D_mcPDG*e_mcPDG==411*11 and e_mcPDG*e_genMotherPDG==11*15 and \
        ((B0_mcErrors<64 and B0_mcPDG*D_mcPDG==-511*411) or (B0_mcErrors<512 and abs(B0_mcPDG)==521))'
    
    signals_selection = 'B0_mcPDG*D_mcPDG==-511*411 and D_mcPDG*e_mcPDG==411*11 and e_mcPDG*e_genMotherPDG==11*15'
    norms_selection = 'B0_mcPDG*D_mcPDG==-511*411 and D_mcPDG*e_mcPDG==411*11 and e_genMotherPDG==B0_mcPDG'


    for file in args.input:
        filename = f'{args.dir}/{file}'
        print(colored(f'Loading {file}', 'blue'))
        if file.endswith('parquet'):
            data = pandas.read_parquet(filename, engine='pyarrow')
        elif file.endswith('root'):
            data = root_pandas.read_root(filename)

        if args.mctype=='signal':
            sig_D_tau_nu=data.query(f'DecayMode=={util.DecayMode["sig_D_tau_nu"].value} and \
            B0_mcErrors<32 and {signals_selection}').copy()
            sig_D_tau_nu['Signal'] = True
            samples.append(sig_D_tau_nu)

        if args.mctype=='bkg_DTC':
            bkg_fakeD = data.query('abs(D_mcPDG)!=411 and B0_mcErrors!=512 and B0_isContinuumEvent!=1').copy()
            bkg_fakeTracksClusters = data.query('B0_mcErrors==512 and B0_isContinuumEvent!=1').copy()
            bkg_fakeD['Signal'] = False
            bkg_fakeTracksClusters['Signal'] = False
            samples.append(pandas.concat([bkg_fakeD, bkg_fakeTracksClusters]))
            

        if args.mctype=='bkg_B':
            bkg_combinatorial = data.query('B0_mcPDG==300553 and abs(D_mcPDG)==411 and B0_mcErrors!=512 and B0_isContinuumEvent!=1').copy()
            bkg_sigOtherBDTaudecay = data.query(f'(DecayMode=={util.DecayMode["bkg"].value} or \
            DecayMode=={util.DecayMode["sig_D_mu_nu"].value} or DecayMode=={util.DecayMode["sig_Dst_mu_nu"].value} or \
            DecayMode=={util.DecayMode["all_Dstst_mu_nu"].value}) and B0_mcPDG!=300553 and \
            abs(D_mcPDG)==411 and B0_mcErrors!=512 and B0_isContinuumEvent!=1').copy()
            
            bkg_combinatorial['Signal'] = False
            bkg_sigOtherBDTaudecay['Signal'] = False
            samples.append(pandas.concat([bkg_combinatorial, bkg_sigOtherBDTaudecay]))
            
        if args.mctype=='bkg_continuum':
            samples.append(data)

    df_merged = pandas.concat(samples)
    
    if args.sampling:
        df_merged = df_merged.sample(n=args.sampling, random_state=0)
    if args.reweight:
        df_merged['__weight__'] = args.reweight
        
    size = len(df_merged)
    treename = 'B0'
    output_name = f'reweight{args.reweight}_'+args.output if args.reweight else args.output
    print(colored(f'Saving {output_name} {size=} to folder BDTs', 'magenta'))
    
    if args.output.endswith('parquet'):
        df_merged.to_parquet(f'BDTs/{output_name}', engine="pyarrow", index=False)
    elif args.output.endswith('root'):
        with uproot.recreate(f'BDTs/{output_name}') as file:
            file['B0'] = df_merged
