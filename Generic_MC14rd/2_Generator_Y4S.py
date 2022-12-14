# +
import basf2 as b2
import modularAnalysis as ma
from variables import variables as vm
import variables.collections as vc
import variables.utils as vu
import vertex as vx
#import stdV0s
#from stdCharged import stdE, stdMu
#from variables.MCGenTopo import mc_gen_topo


analysis_gt = ma.getAnalysisGlobaltag()
b2.B2INFO(f"Appending analysis GT: {analysis_gt}")
b2.conditions.append_globaltag(analysis_gt)
    
# Define the path
main_path = b2.Path()

input_file = ['']
output_file = 'grid_MC_e.root'
output_hash = 'hashmap_grid_MC_e.root'

ma.inputMdstList(environmentType='default', filelist=input_file, path=main_path)

ma.fillParticleListFromMC('Upsilon(4S):MC', '',skipNonPrimaryDaughters=True, path=main_path)

# generate the decay string
main_path.add_module('ParticleMCDecayString', listName='Upsilon(4S):MC', fileName=output_hash)
vm.addAlias('DecayHash','extraInfo(DecayHash)')
vm.addAlias('DecayHashEx','extraInfo(DecayHashExtended)')

vm.addAlias('d0d0_mcPDG','daughter(0,daughter(0,mcPDG))')
vm.addAlias('d0d1_mcPDG','daughter(0,daughter(1,mcPDG))')
vm.addAlias('d1d0_mcPDG','daughter(1,daughter(0,mcPDG))')
vm.addAlias('d1d1_mcPDG','daughter(1,daughter(1,mcPDG))')
daughters_info=['d0d0_mcPDG','d0d1_mcPDG','d1d0_mcPDG','d1d1_mcPDG']
    
ma.variablesToNtuple('Upsilon(4S):MC', daughters_info, filename=output_file, treename='Y4S', path=main_path)

b2.process(path=main_path)
