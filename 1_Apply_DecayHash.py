# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool to apply offline cuts (and DecayHash) to Ntuples and save/split them to one/multiple parquet/root files.

Usage: python3 1_Apply_DecayHash.py -d folder -i -o -l -n (--nohash)

Example: python3 1_Apply_DecayHash.py -d Samples/Signal_MC14ri/MC14ri_sigDDst_bengal_e_2 \
         -i sigDDst_bengal_e_2.root -o sigDDst_bengal_e_2.parquet -l e -n 1 --mctype signal (--nohash)
"""

import argparse
import decayHash
from decayHash import DecayHashMap
import ROOT
import uproot
import pandas
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from termcolor import colored
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
                        help="List of input root files in the --dir filder")
    parser.add_argument('-o', "--output",
                        action="store",
                        type=str,
                        required=True,
                        help="name of output files")
    parser.add_argument('-l', "--lmode",
                        action="store",
                        type=str,
                        required=True,
                        choices=['e', 'mu'],
                        help="Lepton mode, e or mu")
    parser.add_argument("--mctype",
                        action="store",
                        type=str,
                        default='generic',
                        choices=['generic', 'signal'],
                        help="generic MC or signal MC")
    parser.add_argument('-n', "--nchunk",
                        action="store",
                        type=int,
                        required=True,
                        help="Number of output files from one input file")
    parser.add_argument("--nohash",
                        action="store_true",
                        help="Not to apply decayhash")


    return parser

def apply_decayHash(df, filename, args):
    # import channel maps and add DecayHash column
    hash_modes = util.mode_dict[args.lmode]

    def found(modes,row):
        for mode in modes:
            # check the decay chain for the reconstructed B meson only
            if mode.startswith(str(int(row['B0_mcPDG']))):
                decaytree = ROOT.Belle2.DecayTree(mode)
                if hashmap2.get_original_decay(row["B0_DecayHash"],row["B0_DecayHashEx"]).find_decay(decaytree):
                    return True
            else:
                continue
        return False

    def decay_mode(row):
        # return bkg for mis-reconstructed events
        if abs(int(row['B0_mcPDG'])) not in [511, 521]:
            return util.DecayMode['bkg'].value
        if abs(int(row['D_mcPDG']))!=411:
            return util.DecayMode['bkg'].value
        if abs(int(row[f'{args.lmode}_genMotherPDG'])) not in [15, 511, 521]:
            return util.DecayMode['bkg'].value
        if abs(int(row[f'{args.lmode}_mcPDG'])) not in [11, 13]:
            return util.DecayMode['bkg'].value
        
        # return signal modes
        sig_mode_list = ['sig_D_tau_nu', 'sig_Dst_tau_nu','all_Dstst_tau_nu']
        if abs(int(row[f'{args.lmode}_genMotherPDG']))==15:
            for name,modes in hash_modes.items():
                if name not in sig_mode_list:
                    continue
                if found(modes,row):
                    return util.DecayMode[name].value
            return util.DecayMode['bkg'].value
        
        norm_mode_list = ['sig_D_e_nu','sig_Dst_e_nu', 'all_Dstst_e_nu',
                          'sig_D_mu_nu', 'sig_Dst_mu_nu', 'all_Dstst_mu_nu']
        if abs(int(row[f'{args.lmode}_genMotherPDG'])) in [511, 521]:
            for name,modes in hash_modes.items():
                if name not in norm_mode_list:
                    continue
                if found(modes,row):
                    return util.DecayMode[name].value
            return util.DecayMode['bkg'].value

    decayhash=f'{args.dir}/hashmap_{filename}'
    if filename.endswith('parquet'):
        decayhash=f'{args.dir}/hashmap_{filename.strip("parquet")}root'
    hashmap2 = DecayHashMap(decayhash, removeRadiativeGammaFlag=True)

    print(colored('Appending Decayhash column to the dataframe', 'magenta'))
    df['DecayMode'] = df.progress_apply(decay_mode, axis=1)


if __name__ == "__main__":

    # read command line optionsa
    args = argparser().parse_args()
    
    # apply offline cuts
    cut = 'D_vtxReChi2<13 and B0_vtxReChi2<14 and -3.2<B0_deltaE<0 and \
           5<B0_roeMbc_my_mask and 4.3<B0_CMS2_weMbc and B0_CMS_E<5.4 and \
           -5<B0_roeDeltae_my_mask<2 and -3<B0_CMS0_weDeltae<2 and \
           abs(B0_roeCharge_my_mask)<3 and \
           0.2967<B0_Lab5_weMissPTheta<2.7925 and 0.2967<B0_Lab6_weMissPTheta<2.7925 and \
           0<B0_TagVReChi2<100 and 0<B0_TagVReChi2IP<100'
    # lepton veto (nElectrons90+nMuons90)==1; e_p>0.2, mu_p>0.6
    
    if args.nohash:
        # Load Ntuples
        print(colored(f'Loading Ntuples', 'blue'))
        file_path_list = [f'{args.dir}/{file}' for file in args.input]
        with uproot.concatenate(file_path_list, library="np") as data_dict:
            df = pandas.DataFrame(data_dict)
        # Apply offline cuts
        df_cut = df.query(cut).copy()
        df_cut['B0_mcPDG'] = df_cut['B0_mcPDG'].fillna(0)
        print(colored(f'sample size decreases from {len(df)} to {len(df_cut)} after cut', 'blue'))
        df_merged = df_cut
        
    else:
        # Load one Ntuple at a time
        samples = []
        for filename in args.input:
            print(colored(f'Loading {filename}', 'blue'))
            file_location = f'{args.dir}/{filename}'
#             with uproot.concatenate([file_location], library="np") as data_dict:
#                 df = pandas.DataFrame(data_dict)
            if filename.endswith('parquet'):
                df = pandas.read_parquet(file_location, engine="pyarrow")
            else:
                with uproot.open(file_location)['B0'] as file:
                    df = pandas.DataFrame(file.arrays(library="np"))
            # Offline cuts
            df_cut = df.query(cut).copy()
            print(colored(f'sample size decreases from {len(df)} to {len(df_cut)} after cut', 'blue'))
            # Apply DecayHash
            df_cut['B0_mcPDG'] = df_cut['B0_mcPDG'].fillna(0)
            apply_decayHash(df_cut, filename, args)
            samples.append(df_cut)
        df_merged = pandas.concat(samples,ignore_index=True)
            
    df_merged['Signal'] = False
    df_merged.eval('B_D_ReChi2 = B0_vtxReChi2 + D_vtxReChi2', inplace=True)
    if args.lmode=='e':
        df_merged.eval('p_D_l = D_CMS_p + e_CMS_p', inplace=True)
    elif args.lmode=='mu':
        df_merged.eval('p_D_l = D_CMS_p + mu_CMS_p', inplace=True)
    
    # Apply BCS, should be after MVA cuts
    #print(colored('Selecting the Best Candidate', 'magenta'))
    #df_bestSelected=df_cut.loc[df_cut.groupby(['__experiment__','__run__','__event__','__production__']).B_D_ReChi2.idxmin()]
            
    for idx, chunk in enumerate(tqdm(np.array_split(df_merged, args.nchunk), desc =f"Saving output file(s)")):
        #chunk.reset_index().to_feather(f'{args.dir}/{filename}_{idx}.feather')
        if args.output.endswith('parquet'):
            outname = args.output.strip('parquet').strip('.')
            chunk.to_parquet(f'{args.dir}/{outname}_{idx}.parquet', engine="pyarrow", index=False)
        elif args.output.endswith('root'):
            outname = args.output.strip('root').strip('.')
            #chunk.to_root(f'{args.dir}/{outname}_{idx}.root','B0')
            with uproot.recreate(f'{args.dir}/{outname}_{idx}.root') as file:
                file['B0'] = chunk
