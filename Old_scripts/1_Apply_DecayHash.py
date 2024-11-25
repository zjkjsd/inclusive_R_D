# -*- coding: utf-8 -*-
# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool to apply offline cuts (and DecayHash) to Ntuples and save/split them to one/multiple parquet/root files.

Usage: python3 1_Apply_DecayHash.py -d folder -i -o -l -n (--nohash)

Example: python3 1_Apply_DecayHash.py -d Samples/Signal_MC14ri/MC14ri_sigDDst_bengal_e_2 \
         -i sigDDst_bengal_e_2.root -o parquet -l e --mctype signal (--nohash)
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
                        choices=['parquet', 'root'],
                        required=True,
                        help="Format of output files, parquet or root")
    parser.add_argument('-l', "--lmode",
                        action="store",
                        type=str,
                        required=True,
                        choices=['e', 'mu'],
                        help="Lepton mode, e or mu")
#     parser.add_argument("--mctype",
#                         action="store",
#                         type=str,
#                         default='generic',
#                         choices=['generic', 'signal'],
#                         help="generic MC or signal MC")
#     parser.add_argument('-n', "--nchunk",
#                         action="store",
#                         type=int,
#                         required=True,
#                         help="Number of output files from one input file")
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
            else:
                continue
        return False

    def decay_mode(row):
        # mcErrors is complicated and handled separately in util
        
        # return bkg for mis-reconstructed events
        if abs(int(row['D_mcPDG']))!=411 or int(row['D_mcErrors'])>=8:
            return util.DecayMode['bkg'].value
        if abs(int(row['B0_mcPDG'])) not in [511, 521]:
            return util.DecayMode['bkg'].value
        if abs(int(row[f'{args.lmode}_genMotherPDG'])) not in [15, 511, 521]:
            return util.DecayMode['bkg'].value
        if abs(int(row[f'{args.lmode}_mcPDG'])) not in [11, 13]:
            return util.DecayMode['bkg'].value
        
        # return signal modes
        for name,modes in hash_modes.items():
            if name=='sig_D_tau_nu':
                if abs(int(row[f'D_genMotherPDG']))==511 and abs(int(row[f'{args.lmode}_genMotherPDG']))==15: 
                # + e_GMPDG==D_MPDG==B_mcPDG
                    pass
                else:
                    continue
            if name=='sig_D_l_nu':
                if abs(int(row[f'D_genMotherPDG']))==511 and abs(int(row[f'{args.lmode}_genMotherPDG']))==511: 
                # + e_MPDG==D_MPDG==B_mcPDG
                    pass
                else:
                    continue
            if name=='sig_Dst_tau_nu':
                if abs(int(row[f'D_genMotherPDG']))==413 and abs(int(row[f'{args.lmode}_genMotherPDG']))==15: 
                # + e_GMPDG==D_GMPDG==B_mcPDG
                    pass
                else:
                    continue
            if name=='sig_Dst_l_nu':
                if abs(int(row[f'D_genMotherPDG']))==413 and abs(int(row[f'{args.lmode}_genMotherPDG']))==511: 
                # + e_MPDG==D_GMPDG==B_mcPDG
                    pass
                else:
                    continue
            if name in ['Dstst_tau_nu_mixed','Dstst_tau_nu_charged']:
                if abs(int(row[f'{args.lmode}_genMotherPDG']))==15:
                # + e_GMPDG==B_mcPDG
                    pass
                else:
                    continue
            if name in ['res_Dstst_l_nu_mixed','res_Dstst_l_nu_charged']:
                if abs(int(row[f'{args.lmode}_genMotherPDG'])) in [511,521]:
                # + e_MPDG==B_mcPDG
                    pass
                else:
                    continue
            if name in ['nonres_Dstst_l_nu_mixed','gap_Dstst_l_nu_mixed']:
                if (abs(int(row[f'D_genMotherPDG'])) in [413,511]) and abs(int(row[f'{args.lmode}_genMotherPDG']))==511: 
                # + e_MPDG==B_mcPDG
                    pass
                else:
                    continue
            if name=='nonres_Dstst_l_nu_charged':
                if (abs(int(row[f'D_genMotherPDG'])) in [413,521]) and abs(int(row[f'{args.lmode}_genMotherPDG']))==521: 
                # + e_MPDG==B_mcPDG
                    pass
                else:
                    continue
            
            if found(modes,row):
                return util.DecayMode[name].value
            else:
                continue
        return util.DecayMode['bkg'].value
    
#         # return signal modes
#         if abs(int(row[f'{args.lmode}_genMotherPDG']))==15:
#             sig_mode_list = ['sig_D_tau_nu', 'sig_Dst_tau_nu','Dstst_tau_nu_mixed','Dstst_tau_nu_charged']
#             for name,modes in hash_modes.items():
#                 if name not in sig_mode_list: 
#                     continue # only tau mode is considered here
#                 if found(modes,row):
#                     return util.DecayMode[name].value
#                 else:
#                     continue
#             return util.DecayMode['bkg'].value
        
#         if abs(int(row[f'{args.lmode}_genMotherPDG'])) in [511, 521]:
#             norm_mode_list = ['sig_D_l_nu','sig_Dst_l_nu', 'res_Dstst_l_nu_mixed',
#                               'nonres_Dstst_l_nu_mixed', 'gap_Dstst_l_nu_mixed',
#                               'res_Dstst_l_nu_charged','nonres_Dstst_l_nu_charged']
#             for name,modes in hash_modes.items():
#                 if name not in norm_mode_list: 
#                     continue # it has to be a ell mode
#                 if found(modes,row):
#                     return util.DecayMode[name].value
#                 else:
#                     continue
#             return util.DecayMode['bkg'].value
        
#         else:
#             return util.DecayMode['bkg'].value

    decayhash=f'{args.dir}/hashmap_{filename}'
    if filename.endswith('parquet'):
        decayhash=f'{args.dir}/hashmap_{filename.strip("parquet")}root'
        
    print(colored('Loading the Hash file', 'blue'))
    hashmap2 = DecayHashMap(decayhash, removeRadiativeGammaFlag=True)

    print(colored('Appending Decayhash column to the dataframe', 'magenta'))
    df['DecayMode'] = df.progress_apply(decay_mode, axis=1)


if __name__ == "__main__":

    # read command line optionsa
    args = argparser().parse_args()
    
    # apply offline cuts
    cut = 'D_vtxReChi2<13 and B0_vtxReChi2<14 and -3.2<B0_deltaE<0 and \
           5<B0_roeMbc_my_mask and 4.3<B0_CMS2_weMbc and \
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
        print(colored(f'sample size decreases from {len(df)} to {len(df_cut)} after cut', 'magenta'))
        df_cut['B0_mcPDG'] = df_cut['B0_mcPDG'].fillna(0)
        df_cut['D_mcPDG'] = df_cut['D_mcPDG'].fillna(0)
        df_merged = df_cut
        
    else:
        # Load one Ntuple at a time
        for filename in args.input:
            print(colored(f'Loading {filename}', 'blue'))
            file_location = f'{args.dir}/{filename}'
            if filename.endswith('parquet'):
                df = pandas.read_parquet(file_location, engine="pyarrow")
                del df['__MCDecayString__']
            else:
                with uproot.open(file_location)['B0'] as file:
                    df = pandas.DataFrame(file.arrays(entry_start=0,entry_stop=None,library="np",
                                                      filter_branch=lambda branch: branch.name != "__MCDecayString__"))
#             with uproot.concatenate([file_location], library="np") as data_dict:
#                 df = pandas.DataFrame(data_dict)

            # Offline cuts
            df_cut = df.query(cut).copy()
            print(colored(f'sample size decreases from {len(df)} to {len(df_cut)} after cut', 'magenta'))
            del df
            # Apply DecayHash
            df_cut['B0_mcPDG'] = df_cut['B0_mcPDG'].fillna(0)
            df_cut['D_mcPDG'] = df_cut['D_mcPDG'].fillna(0)
            apply_decayHash(df_cut, filename, args)
            
            df_cut['Signal'] = False
            df_cut.eval('B_D_ReChi2 = B0_vtxReChi2 + D_vtxReChi2', inplace=True)
            df_cut.eval(f'p_D_l = D_CMS_p + {args.lmode}_CMS_p', inplace=True)
            
#             print(df_cut.info(verbose=True))
    
    
#     Apply BCS, should be after MVA cuts
#     for idx, chunk in enumerate(tqdm(np.array_split(df_merged, args.nchunk), desc =f"Saving output file(s)")):
            
            if args.output=='parquet':
                outname = filename.strip('root')
                print(colored(f'Saving {outname}parquet', 'blue'))
                df_cut.to_parquet(f'{args.dir}/hashed_{outname}parquet', engine="pyarrow", index=False)
            elif args.output=='root':
                print(colored(f'Saving hashed_{filename}', 'blue'))
                with uproot.recreate(f'{args.dir}/hashed_{filename}') as file:
                    file['B0'] = df_cut
