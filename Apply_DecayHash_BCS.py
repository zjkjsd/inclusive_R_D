# +
# Usage: basf2 Apply_DecayHash_BCS.py folder Filename(w/o .root) eORmu nChunk

import sys
sys.path.append('/home/belle/zhangboy/R_D/')
from termcolor import colored
import utilities as util
import decayHash
from decayHash import DecayHashMap
import ROOT
import root_pandas
import pandas
import numpy as np
from tqdm import tqdm
tqdm.pandas()

# read in root-file as a pandas dataframe
folder = sys.argv[1]
filename = sys.argv[2]
eORmu = sys.argv[3]
nChunk = int(sys.argv[4])
decayhash=f'{folder}/hashmap_{filename}.root'
#data = uproot.open(filename)['B0'].arrays(library="pd")
print(colored('Loading Ntuple and Hashmap', 'blue'))
df = root_pandas.read_root(f'{folder}/{filename}.root',key='B0')
hashmap2 = DecayHashMap(decayhash, removeRadiativeGammaFlag=True)

# generic MC
#cut = 'D_vtxReChi2<13 and B0_vtxReChi2<14 and -3.2<B0_deltaE<0 and e_p>0.2 and \
#    5<B0_roeMbc_my_mask and 4.3<B0_CMS2_weMbc and B0_CMS_E<5.4 and \
#    -5<B0_roeDeltae_my_mask<2 and -3<B0_CMS0_weDeltae<2 and \
#    abs(B0_roeCharge_my_mask)<3 and \
#    0.2967<B0_Lab5_weMissPTheta<2.7925 and 0.2967<B0_Lab6_weMissPTheta<2.7925 and \
#    0<B0_TagVReChi2<100 and 0<B0_TagVReChi2IP<100'

# signal MC
cut = 'D_vtxReChi2<13 and B0_vtxReChi2<14 and -3.2<B0_deltaE<0 and e_p>0.2 and \
    5<B0_roeMbc_my_mask and 4.3<B0_CMS2_weMbc and B0_CMS_E<5.4 and \
    -5<B0_roeDeltae_my_mask<2 and -3<B0_CMS0_weDeltae<2 and \
    abs(B0_roeCharge_my_mask)<3 and (nElectrons90+nMuons90)==1 and \
    0.2967<B0_Lab5_weMissPTheta<2.7925 and 0.2967<B0_Lab6_weMissPTheta<2.7925 and \
    0<B0_TagVReChi2<100 and 0<B0_TagVReChi2IP<100'

data = df.query(cut).copy()

# fillna for B0_mcPDG and prepare BCS variables
data['B0_mcPDG'] = data['B0_mcPDG'].fillna(0)
data['isSignal'] = 1.0
data.eval('B_D_ReChi2 = B0_vtxReChi2 + D_vtxReChi2', inplace=True)
data.eval('p_D_l = D_CMS_p + e_CMS_p', inplace=True)

# import channel maps and add DecayHash column
hash_modes = util.mode_dict[eORmu]

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
    for name,modes in hash_modes.items():
        if found(modes,row):
            return name
    return 'bkg'

print(colored('Appending Decayhash column to the dataframe', 'magenta'))
data['DecayMode'] = data.progress_apply(decay_mode, axis=1)
print(colored('Selecting the Best Candidate', 'magenta'))
df_bestSelected=data.loc[data.groupby(['__experiment__','__run__','__event__','__production__']).B_D_ReChi2.idxmin()]

for idx, chunk in enumerate(tqdm(np.array_split(data, nChunk), desc ="Saving output files")):
    #chunk.reset_index().to_feather(f'{folder}/{filename}_{idx}.feather')
    chunk.to_parquet(f'{folder}/{filename}_{idx}.parquet', engine="pyarrow", index=False)
