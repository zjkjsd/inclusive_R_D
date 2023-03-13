# +
# Usage: basf2 Save_Templates 2d_2channels_workspace_3_0.json
import sys
sys.path.append('/home/belle/zhangboy/R_D/')
import utilities as util
from termcolor import colored
import json
import pandas
import numpy as np
from tqdm import tqdm
tqdm.pandas()

# read parquets as pandas dataframe
samples = {}
files = ['sigDDst', 'normDDst','bkgDststp_tau', 'bkgDstst0_tau','bkgDstst0_ell']

Dstst_e_nu_selection = 'DecayMode=="all_Dstst_e_nu" and D_mcPDG*e_mcPDG==411*11 and e_genMotherPDG==B0_mcPDG and \
    ((B0_mcErrors<64 and B0_mcPDG*D_mcPDG==-511*411) or (B0_mcErrors<512 and abs(B0_mcPDG)==521))'
Dstst_tau_nu_selection = 'DecayMode=="all_Dstst_tau_nu" and D_mcPDG*e_mcPDG==411*11 and e_mcPDG*e_genMotherPDG==11*15 and \
    ((B0_mcErrors<64 and B0_mcPDG*D_mcPDG==-511*411) or (B0_mcErrors<512 and abs(B0_mcPDG)==521))'
signals_selection = 'B0_mcPDG*D_mcPDG==-511*411 and D_mcPDG*e_mcPDG==411*11 and e_mcPDG*e_genMotherPDG==11*15'
norms_selection = 'B0_mcPDG*D_mcPDG==-511*411 and D_mcPDG*e_mcPDG==411*11 and e_genMotherPDG==B0_mcPDG'


for file_name in tqdm(files, desc=colored('Loading parquets', 'blue')):
    filename=f'/home/belle/zhangboy/R_D/Samples/Signal_MC14ri/MC14ri_{file_name}_bengal_e_2/{file_name}_bengal_e_2_0.parquet'
    data = pandas.read_parquet(filename, engine="pyarrow")
    
    # Signal components
    if file_name == files[0]:
        sig_D_tau_nu=data.query(f'DecayMode=="sig_D_tau_nu" and B0_mcErrors<32 and {signals_selection}').copy()
        sig_Dst_tau_nu=data.query(f'DecayMode=="sig_Dst_tau_nu" and B0_mcErrors<64 and {signals_selection}').copy()
        samples[r'$D\tau\nu$'] = sig_D_tau_nu
        samples[r'$D^\ast\tau\nu$'] = sig_Dst_tau_nu
        
    if file_name == files[1]:
        sig_D_e_nu=data.query(f'DecayMode=="sig_D_e_nu" and B0_mcErrors<16 and {norms_selection}').copy()
        sig_Dst_e_nu=data.query(f'DecayMode=="sig_Dst_e_nu" and B0_mcErrors<64 and {norms_selection}').copy()
        Dstst_e_nu_p=data.query(Dstst_e_nu_selection).copy()
        samples[r'$D\ell\nu$'] = sig_D_e_nu
        samples[r'$D^\ast\ell\nu$'] = sig_Dst_e_nu
        
    if file_name == files[2]:
        Dstst_tau_nu_p=data.query(Dstst_tau_nu_selection).copy()
        
    if file_name == files[3]:
        Dstst_tau_nu_0=data.query(Dstst_tau_nu_selection).copy()
        samples[r'$D^{\ast\ast}\tau\nu$'] = pandas.concat([Dstst_tau_nu_p,Dstst_tau_nu_0])
        
    if file_name == files[4]:
        Dstst_e_nu_0=data.query(Dstst_e_nu_selection).copy()
        samples[r'$D^{\ast\ast}\ell\nu$'] = pandas.concat([Dstst_e_nu_p,Dstst_e_nu_0])
        

workspace = sys.argv[1]
workspace_file = f'/home/belle/zhangboy/R_D/Samples/Signal_MC14ri/{workspace}'
cut='B0_roeMbc_my_mask>4'
xedges = np.linspace(-2, 10, 48) # -7.5 for weMiss2, -2 for weMiss3, -2.5 for weMiss4
yedges = np.linspace(0.4, 4.6, 42)
variable_x = 'B0_CMS3_weMissM2'
variable_y = 'p_D_l'


i = 0

for name, sample in samples.items():
    (counts, xedges, yedges) = np.histogram2d(sample.query(cut)[variable_x], 
                                              sample.query(cut)[variable_y],
                                              bins=[xedges, yedges])
    counts = counts.T
    

    size = np.sum(counts)
#    print(colored(f'{Total =}, {Recorded =}, {Total - Recorded =}', 'blue'))
    print(colored(f'Template {name=}, {size=}', 'magenta'))    
    
    with open(workspace_file, 'r+') as f:
        data = json.load(f)
        data['channels'][0]['samples'][i]['name'] = name
        data['channels'][0]['samples'][i]['data'] = counts.ravel().tolist()
        # counts.ravel()/.reshape(-1) returns a view, counts.flatten() returns a copy (slower)
        f.seek(0)        # <--- should reset file position to the beginning.
        json.dump(data, f, indent=4)
        f.truncate()     # remove remaining part

    i += 1
