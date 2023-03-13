# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool to add Signal column to train/test samples.

Usage: python3 5_Add_Labels_or_combine.py -d folder -i -f (-s)

Example: python3 5_Add_Labels_or_combine.py -d BDTs -i MC14ri_taupair.root -f label (-s)


basf2_mva_expert --identifiers CS/MVA1_FastBDT.xml DTCFake/MVA2_1_FastBDT.xml BFake/MVA2_2_FastBDT.xml \
                 --treename B0 --datafiles MC14ri_taupair.root --outputfile MC14ri_taupair_applied.root
                 
basf2_mva_expert --identifiers BDTs/CS/MVA1_FastBDT.xml BDTs/DTCFake/MVA2_1_FastBDT.xml \
                BDTs/BFake/MVA2_2_FastBDT.xml --treename B0 \
                --datafiles Samples/Generic_MC14ri/MC14ri_taupair_e_bengal_1/grid_MC_e.root \
                --outputfile Samples/Generic_MC14ri/MC14ri_taupair_e_bengal_1/grid_MC_e_applied.root
"""

import argparse
import root_pandas
import uproot
import pandas
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from termcolor import colored


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
                        action="store",
                        nargs='+',
                        required=True,
                        help="List of input root files in the --dir filder")
    parser.add_argument('-c',"--columnname",
                        action="store",
                        type=str,
                        default = 'Signal',
                        required=False,
                        help="name of the training label column")
    parser.add_argument('-s', "--signal",
                        action="store_true",
                        help="True: signal, False: bkg")
    parser.add_argument('-f',"--function",
                        action="store",
                        type=str,
                        required=True,
                        choices=['label', 'combine','rename'],
                        help="Function of this script, adding label column or \
                              combine spectators and BDT outputs or \
                              rename the BDT outputs columns")



    return parser


if __name__ == "__main__":

    # read command line optionsa
    args = argparser().parse_args()
    
    for file in tqdm(args.input, desc=colored(f'{args.function}', 'blue')):
        file_path = f'{args.dir}/{file}'
        if args.function=='combine':
            applied_file_path = f"{args.dir}/{file.strip('root').strip('.')}_applied.root"
            df_BDTout = root_pandas.read_root(applied_file_path, 
                columns=['BDTs__slCS__slMVA1_FastBDT__ptxml', 'BDTs__slDTCFake__slMVA2_1_FastBDT__ptxml', 
                         'BDTs__slBFake__slMVA2_2_FastBDT__ptxml'])
            
            df_BDTout.rename(columns={"BDTs__slCS__slMVA1_FastBDT__ptxml": "CS__slMVA1_FastBDT__ptxml", 
                            "BDTs__slDTCFake__slMVA2_1_FastBDT__ptxml": "DTCFake__slMVA2_1_FastBDT__ptxml",
                            'BDTs__slBFake__slMVA2_2_FastBDT__ptxml':'BFake__slMVA2_2_FastBDT__ptxml'},
                            inplace=True)
            
            df_spectators = root_pandas.read_root(file_path, 
                columns=['__weight__','D_CMS_p', 'e_CMS_p', 'B0_CMS3_weMissM2', 'B0_CMS3_weQ2lnuSimple'])
            
            df = pandas.concat([df_spectators, df_BDTout], axis=1)
            ## [ERROR] There is a mix of float and basf2 variable types (double, int, bool)
            ## Error in <TChain::SetBranchAddress>: The pointer type given "Float_t" (5) does not correspond 
            ## to the type needed "Double_t" (8) by the branch: __weight__
            for col in df.columns:
                df[col] = np.float64(df[col])
            df[args.columnname] = args.signal
            with uproot.recreate(applied_file_path) as file:
                file['B0'] = df
        elif args.function=='label':
            df = root_pandas.read_root(file_path)
            df[args.columnname] = args.signal
            with uproot.recreate(file_path) as file:
                file['B0'] = df
