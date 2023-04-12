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

#input_file = [''] # for grid
# the basf2 command line code is: 
# bsub -q l 'basf2 /current/directory/3_Reconstruction_e.py  D_tau/Dst_l'
#import sys
#decaymode = sys.argv[1]
#filename = sys.argv[2]
input_file = ['']
output_file = 'grid_MC_mu.root'
output_hash = 'hashmap_grid_MC_mu.root'

ma.inputMdstList(environmentType='default', filelist=input_file, path=main_path)

goodTrack = 'abs(dz)<4 and dr<2 and pt>0.1 and E<5.5 and thetaInCDCAcceptance and nCDCHits>0'

vm.addAlias("pionID_binary_noSVD", "binaryPID_noSVD(211, 321)")
vm.addAlias("kaonID_binary_noSVD", "binaryPID_noSVD(321, 211)")
ma.fillParticleList('pi+:mypi', cut=goodTrack + ' and pionID_binary_noSVD > 0.1', path=main_path)
ma.fillParticleList('K-:myk', cut=goodTrack + ' and kaonID_binary_noSVD > 0.1', path=main_path)

# ----------------------------------
# Fill example standard lepton list.
# ----------------------------------

# For electrons, we show the case in which a Bremsstrahlung correction
# is applied first to get the 4-momentum right,
# and the resulting particle list is passed as input to the stdE list creator.
ma.fillParticleList("e+:uncorrected",
                    cut="dr < 2 and abs(dz) < 4",  # NB: whichever cut is set here, will be inherited by the std electrons.
                    path=main_path)
ma.fillParticleList("gamma:bremsinput", cut="0.05<clusterE<2.5",
                    loadPhotonBeamBackgroundMVA=True, path=main_path)
ma.applyCuts('gamma:bremsinput','beamBackgroundSuppression>0.2', path=main_path)
ma.correctBremsBelle(outputListName="e+:corrected",
                     inputListName="e+:uncorrected",
                     gammaListName="gamma:bremsinput",
                     angleThreshold=0.15,
                     multiplePhotons=False,
                     path=main_path)
vm.addAlias("isBremsCorrected", "extraInfo(bremsCorrected)")

# global lepton id
ma.applyChargedPidMVA(['e+:corrected'], path=main_path, trainingMode=1, 
                      chargeIndependent=False, binaryHypoPDGCodes=(0, 0))
#ma.applyChargedPidMVA(['e+:corrected'], path=main_path, trainingMode=0,
#                      chargeIndependent=False, binaryHypoPDGCodes=(11, 211))


ma.fillParticleList("mu+:mymu",
                    cut="dr<2 and abs(dz)<4 and thetaInCDCAcceptance and inKLMAcceptance",
                    path=main_path)

ma.applyChargedPidMVA(['mu+:mymu'], path=main_path, trainingMode=1, 
                      chargeIndependent=False, binaryHypoPDGCodes=(0, 0))


# ------------------------------------------------------------
# Add extra cuts on the standard lepton lists and lepton veto.
# ------------------------------------------------------------

ma.applyCuts(f"e-:corrected", "pidChargedBDTScore(11, ALL)>0.9 and p>0.2 and thetaInCDCAcceptance and nCDCHits>0 and nPXDHits>0", path=main_path)
ma.applyCuts(f"mu-:mymu", "pidChargedBDTScore(13, ALL)>0.9", path=main_path)

# select events only containing 1 lepton
vm.addAlias('nElectrons90','nParticlesInList(e-:corrected)')
vm.addAlias('nMuons90','nParticlesInList(mu-:mymu)')
#ma.applyEventCuts('formula(nParticlesInList(e-:corrected) + nParticlesInList(mu-:mymu)) == 1', path=main_path)

#stdV0s.stdKshorts(path=main_path)

# Event Kinematics
ma.buildEventKinematics(fillWithMostLikely=True,path=main_path) 




# Reconstruct D
#Dcuts = '1.855 < M < 1.885'
Dcuts = '1.84 <M< 1.9'
ma.reconstructDecay('D+:K2pi -> K-:myk pi+:mypi pi+:mypi', cut=Dcuts, path=main_path)

ma.variablesToExtraInfo('D+:K2pi', variables={'M':'D_BFM','InvM':'D_BFInvM'},option=0, path=main_path)
vm.addAlias('BFM','extraInfo(D_BFM)')
vm.addAlias('BFInvM','extraInfo(D_BFInvM)')

Daughters_vars = []
for variable in ['kaonID_binary_noSVD','pionID_binary_noSVD','dr','dz','nCDCHits','nPXDHits','pValue']:
    vm.addAlias(f'K_{variable}', f'daughter(0, {variable})')
    vm.addAlias(f'pi1_{variable}', f'daughter(1, {variable})')
    vm.addAlias(f'pi2_{variable}', f'daughter(2, {variable})')
    Daughters_vars.append(f'K_{variable}')
    Daughters_vars.append(f'pi1_{variable}')
    Daughters_vars.append(f'pi2_{variable}')
    

# vertex fitting D, save vtx variables before the 2nd freefit
vx.treeFit('D+:K2pi', conf_level=0.00, updateAllDaughters=True, massConstraint=[], ipConstraint=False, path=main_path)
vm.addAlias('vtxChi2','extraInfo(chiSquared)')
vm.addAlias('vtxNDF','extraInfo(ndf)')
vm.addAlias('vtxReChi2','formula(vtxChi2/vtxNDF)')
vm.addAlias('flightDistanceSig','formula(flightDistance/flightDistanceErr)')
vm.addAlias('flightTimeSig','formula(flightTime/flightTimeErr)')
ma.variablesToExtraInfo('D+:K2pi', variables={'flightDistanceSig':'D_A1FflightDistanceSig'},option=1, path=main_path)
vm.addAlias('A1FflightDistanceSig_IP','extraInfo(D_A1FflightDistanceSig)')

vertex_vars = ['vtxReChi2','vtxNDF','flightDistanceSig','flightTimeSig',]

#ma.applyCuts('D+:K2pi', 'vtxReChi2<13', path=main_path)

# Reconstruct B
ma.reconstructDecay('anti-B0:Dl -> D+:K2pi mu-:mymu ?addbrems', cut='', path=main_path)
vx.treeFit('anti-B0:Dl', conf_level=0.00, updateAllDaughters=False, massConstraint=[], ipConstraint=True, path=main_path)

# Get the distance between vertices De/IP and D+
vm.addAlias('vtxDDSig', 'vertexDistanceOfDaughterSignificance(0,0)')
#vm.addAlias('vtxIPDSig', 'vertexDistanceOfDaughterSignificance(0)') this output the same distribution as above

# Calculate DOCA(D,l)
# ma.calculateDistance('anti-B0:Dl', 'anti-B0:Dl -> ^D+:K2pi ^mu-:mymu', "vertextrack", path=main_path)
vm.addAlias('DistanceSig', 'formula( extraInfo(CalculatedDistance) / extraInfo(CalculatedDistanceError) )')

def Distance_dic(target='sig_', kind='_vtx'):
    return {'extraInfo(CalculatedDistance)':f'{target}Distance{kind}',
            'extraInfo(CalculatedDistanceError)':f'{target}DistanceError{kind}',
            'DistanceSig':f'{target}DistanceSig{kind}',
#             'extraInfo(CalculatedDistanceVector_X)':f'{target}DistanceVector_X{kind}',
#             'extraInfo(CalculatedDistanceCovMatrixXZ)':f'{target}DistanceCovMatrixXZ{kind}',
            'daughterAngle(0, 1)':f'{target}daughterAngleLab{kind}',
            'useCMSFrame(daughterAngle(0, 1))':f'{target}daughterAngleCMS{kind}'}

# sigDl_DOCA_dic = Distance_dic('sig_','')
# sigDl_DOCA = []
# for key, value in sigDl_DOCA_dic.items():
#     vm.addAlias(value,key)
#     sigDl_DOCA.append(value)


# Calculate the lepton isolation, the alias will be f'{alias}{detector}'
# 3D distance (default).
#detectors = ['CDC','PID','ECL','KLM']
#ma.calculateTrackIsolation('e-:corrected', main_path,*detectors,
#                           alias="3Ddist")
# 2D distance on rho-phi plane (chord length).
#ma.calculateTrackIsolation('e-:corrected', main_path, *detectors, use2DRhoPhiDist=True,
#                           alias="2Ddist")

#LeptonIso_vars = [f"3Ddist{det}" for det in detectors]
#LeptonIso_vars += [f"2Ddist{det}" for det in detectors]

# MC Truth Matching
ma.matchMCTruth('anti-B0:Dl', path=main_path)

# generate the decay string
main_path.add_module('ParticleMCDecayString', listName='anti-B0:Dl', fileName=output_hash)
vm.addAlias('DecayHash','extraInfo(DecayHash)')
vm.addAlias('DecayHashEx','extraInfo(DecayHashExtended)')



#ma.applyEventCuts(cut='[genUpsilon4S(mcDaughter(0, mcDaughter(0, PDG)))==411] or \
#                    [genUpsilon4S(mcDaughter(0, mcDaughter(0, PDG)))==-411]', path=main_path)

# ma.applyCuts('anti-B0:Dl', 'vtxReChi2<14 and -3.2<deltaE<0 and CMS_E<5.4', path=main_path)



# build the ROE
ma.fillParticleList('pi+:all', '', path=main_path)
ma.tagCurlTracks('pi+:all', mcTruth=True, selectorType='mva', path=main_path)
vm.addAlias('isCurl', 'extraInfo(isCurl)')
vm.addAlias('isTruthCurl', 'extraInfo(isTruthCurl)')
vm.addAlias('truthBundleSize', 'extraInfo(truthBundleSize)')

ma.fillParticleList('gamma:all', '', loadPhotonBeamBackgroundMVA=True,loadPhotonHadronicSplitOffMVA=True, path=main_path)
ma.buildRestOfEvent('anti-B0:Dl', fillWithMostLikely=True,path=main_path)

loose_track = 'dr<10 and abs(dz)<20 and thetaInCDCAcceptance and E < 5.5' 
loose_gamma = "0.05< clusterE < 5.5"
tight_track = f'nCDCHits>=0 and thetaInCDCAcceptance and pValue>=0.0005 and \
                [pt<0.15 and formula(dr**2/36+dz**2/16)<1] or \
                [0.15<pt<0.25 and formula(dr**2/49+dz**2/64)<1] or \
                [0.25<pt<0.5 and formula(dr**2/49+dz**2/16)<1] or \
                [0.5<pt<1 and formula(dr**2/25+dz**2/36)<1] or \
                [pt>1 and formula(dr**2+dz**2)<1]'
tight_gamma = f'clusterE>0.05 and abs(clusterTiming)<formula(2*clusterErrorTiming) and abs(clusterTiming)<200 and \
                beamBackgroundSuppression>0.05 and hadronicSplitOffSuppression>0.1 and minC2TDist>25'
roe_mask1 = ('my_mask',  loose_track, loose_gamma)
ma.appendROEMasks('anti-B0:Dl', [roe_mask1], path=main_path)


# creates V0 particle lists and uses V0 candidates to update/optimize the Rest Of Event
ma.updateROEUsingV0Lists('anti-B0:Dl', mask_names='my_mask', default_cleanup=True, selection_cuts=None,
                         apply_mass_fit=True, fitter='treefit', path=main_path)

ma.updateROEMask("B0:Dl","my_mask",tight_track, tight_gamma, path=main_path)

# Load ROE as a particle and use a mask 'my_mask':
#ma.fillParticleListFromROE('B0:tagFromROE', '', maskName='my_mask',
#  sourceParticleListName='anti-B0:Dl', path=main_path)




roe_path = b2.Path()
deadEndPath = b2.Path()
ma.signalSideParticleFilter('anti-B0:Dl', '', roe_path, deadEndPath)
ma.discardFromROEMasks('pi+:all', ['my_mask'], '[isCurl==1 or pt==0 or E>5.5] and isInRestOfEvent==1', path=roe_path)


# Virtual particles are place holders for the calculateDistance module to work
# These distances below will be used to suppress the combinatorial bkg(in MVAs)
# DOCA between \ell and any track in the ROE
ma.fillParticleList('pi+:roe', 'isInRestOfEvent>0 and passesROEMask(my_mask)', path = roe_path)
ma.fillSignalSideParticleList('mu-:sig', 'anti-B0:Dl -> D+:K2pi ^mu-:mymu ?addbrems', path=roe_path)
ma.reconstructDecay('K_L0:virtual -> pi+:roe mu-:sig ?addbrems', cut='', path=roe_path)
vx.treeFit('K_L0:virtual', conf_level=-1, updateAllDaughters=False, massConstraint=[], ipConstraint=False, path=roe_path)
ma.calculateDistance('K_L0:virtual', 'K_L0:virtual -> ^pi+:roe ^mu-:sig ?addbrems', "2tracks", path=roe_path)

    
ma.rankByLowest('K_L0:virtual', 'DistanceSig', numBest=1, path=roe_path)
roel_DOCA_dis_dic = Distance_dic('roel_','_dis')
ma.variableToSignalSideExtraInfo('K_L0:virtual', roel_DOCA_dis_dic, path=roe_path)
ma.variableToSignalSideExtraInfo('K_L0:virtual', {'vtxReChi2':'roel_vtxReChi2_dis'}, path=roe_path)

roel_DOCA_Chi2 = []
    
for key, value in roel_DOCA_dis_dic.items():
    vm.addAlias(value, f'ifNANgiveX(extraInfo({value}), -1.0)')
    roel_DOCA_Chi2.append(value)
    
vm.addAlias('roel_vtxReChi2_dis', f'ifNANgiveX(extraInfo(roel_vtxReChi2_dis), -1.0)')
roel_DOCA_Chi2.append('roel_vtxReChi2_dis')

# lepton veto 2
ma.applyChargedPidMVA(['pi+:roe'], path=roe_path, trainingMode=1, 
                      chargeIndependent=False, binaryHypoPDGCodes=(0, 0))

ma.applyChargedPidMVA(['pi+:roe'], path=roe_path, trainingMode=1, 
                      chargeIndependent=False, binaryHypoPDGCodes=(0, 0))

# according to sphinx, call this in the main_path first to enable it in the roe_path
ma.variablesToEventExtraInfo('B0:Dl', {'p':'B0_p'}, option=0, path=main_path)

ma.copyList('pi+:roe2', 'pi+:roe', writeOut=False, path=roe_path)
ma.applyCuts(f"pi+:roe", "pidChargedBDTScore(11, ALL)>0.5 and p>0.2 and thetaInCDCAcceptance and nCDCHits>0 and nPXDHits>0", path=roe_path)
ma.applyCuts(f"pi+:roe2", "pidChargedBDTScore(13, ALL)>0.5 and thetaInCDCAcceptance and inKLMAcceptance", path=roe_path)

ma.variablesToEventExtraInfo('pi+:roe', {'pidChargedBDTScore(11, ALL)':'ROEeidBDT'}, option=1, path=roe_path)
ma.variablesToEventExtraInfo('pi+:roe2', {'pidChargedBDTScore(13, ALL)':'ROEmuidBDT'}, option=1, path=roe_path)

for var in ['ROEeidBDT','ROEmuidBDT']:
    vm.addAlias(var, f'ifNANgiveX(eventExtraInfo({var}),-1)')


# Save the ROE tracks to the Ntuple
#ma.matchMCTruth('pi+:roe', path=roe_path)

#ma.variablesToNtuple("pi+:roe", ["dr", "dz", "pt",'pValue','isSignal','mcPDG','mcErrors','isCloneTrack',
#                                 'passesROEMask(my_mask)','hasAncestorFromSignalSide',
#                                 'nCDCHits','ndf', 'isCurl','isTruthCurl',
#                                 #'chi2' is calculated from pValue and ndf but pValue==0 will cause error,
#                                ], 
#                     filename=output_file, treename='pi_roe', path=roe_path)

main_path.for_each('RestOfEvent', 'RestOfEvents', roe_path)




# ROE variables
roe_kinematics = ["roeE(my_mask)", "roeP(my_mask)", "roePx(my_mask)",
                  "roePy(my_mask)","roePz(my_mask)","roePt(my_mask)",]
roe_MC_kinematics = ['roeMC_E','roeMC_M','roeMC_P',
                     'roeMC_PTheta','roeMC_Pt',
                     'roeMC_Px','roeMC_Py','roeMC_Pz',]

roe_Mbc_Deltae = ["roeMbc(my_mask)", "roeM(my_mask)","roeDeltae(my_mask)",]

roe_E_Q = ['roeCharge(my_mask)', 'roeNeextra(my_mask)','roeEextra(my_mask)',]

roe_multiplicities = ["nROE_Charged(my_mask)",'nROE_ECLClusters(my_mask)',
                      'nROE_NeutralECLClusters(my_mask)','nROE_KLMClusters',
                      'nROE_NeutralHadrons(my_mask)',"nROE_Photons(my_mask)",
                      'nROE_Tracks(my_mask)','nROE_Composites(my_mask)',
                      'roeMC_MissFlags(my_mask)',]

vm.addAlias('nROE_e','nROE_Charged(my_mask, 11)')
vm.addAlias('nROE_mu','nROE_Charged(my_mask, 13)')
vm.addAlias('nROE_K','nROE_Charged(my_mask, 321)')
vm.addAlias('nROE_pi','nROE_Charged(my_mask, 211)')
roe_nCharged = ['nROE_e','nROE_mu','nROE_K','nROE_pi']



vm.addAlias('CMS0_weDeltae','weDeltae(my_mask,0)')
vm.addAlias('Lab1_weDeltae','weDeltae(my_mask,1)')
#Option for correctedB_deltae variable should only be 0/1 (CMS/LAB)
#Option for correctedB_mbc variable should only be 0/1/2 (CMS/LAB/CMS with factor)
vm.addAlias('CMS0_weMbc','weMbc(my_mask,0)')
vm.addAlias('Lab1_weMbc','weMbc(my_mask,1)')
vm.addAlias('CMS2_weMbc','weMbc(my_mask,2)')

vm.addAlias('CMS0_weMissM2','weMissM2(my_mask,0)')
vm.addAlias('CMS1_weMissM2','weMissM2(my_mask,1)')
vm.addAlias('CMS2_weMissM2','weMissM2(my_mask,2)')
vm.addAlias('CMS3_weMissM2','weMissM2(my_mask,3)')
vm.addAlias('CMS4_weMissM2','weMissM2(my_mask,4)')
vm.addAlias('Lab5_weMissM2','weMissM2(my_mask,5)')
vm.addAlias('Lab6_weMissM2','weMissM2(my_mask,6)')
vm.addAlias('CMS7_weMissM2','weMissM2(my_mask,7)')

#vm.addAlias('CMS0_weMissPTheta','weMissPTheta(my_mask,0)')
#vm.addAlias('CMS1_weMissPTheta','weMissPTheta(my_mask,1)')
#vm.addAlias('CMS2_weMissPTheta','weMissPTheta(my_mask,2)')
#vm.addAlias('CMS3_weMissPTheta','weMissPTheta(my_mask,3)')
#vm.addAlias('CMS4_weMissPTheta','weMissPTheta(my_mask,4)')
vm.addAlias('Lab5_weMissPTheta','weMissPTheta(my_mask,5)')
vm.addAlias('Lab6_weMissPTheta','weMissPTheta(my_mask,6)')
#vm.addAlias('CMS7_weMissPTheta','weMissPTheta(my_mask,7)')

vm.addAlias('CMS0_weQ2lnuSimple', 'weQ2lnuSimple(my_mask, 0)')
vm.addAlias('CMS1_weQ2lnuSimple', 'weQ2lnuSimple(my_mask, 1)')
vm.addAlias('CMS2_weQ2lnuSimple', 'weQ2lnuSimple(my_mask, 2)')
vm.addAlias('CMS3_weQ2lnuSimple', 'weQ2lnuSimple(my_mask, 3)')
vm.addAlias('CMS4_weQ2lnuSimple', 'weQ2lnuSimple(my_mask, 4)')
vm.addAlias('Lab5_weQ2lnuSimple', 'weQ2lnuSimple(my_mask, 5)')
vm.addAlias('Lab6_weQ2lnuSimple', 'weQ2lnuSimple(my_mask, 6)')
vm.addAlias('CMS7_weQ2lnuSimple', 'weQ2lnuSimple(my_mask, 7)')

we= ['CMS0_weDeltae','CMS0_weMbc','CMS2_weMbc','Lab5_weMissPTheta','Lab6_weMissPTheta']
we_vars = ['weMissM2', 'weQ2lnuSimple']
for var in we_vars:
    for i in range(8):
        if i in [5,6]:
            we.append(f'Lab{i}_{var}')
        elif i<8:
            we.append(f'CMS{i}_{var}')


# vm.addAlias('n_e','nDaughterCharged(11)')
# vm.addAlias('n_mu','nDaughterCharged(13)')
# vm.addAlias('n_K','nDaughterCharged(321)')
# vm.addAlias('n_pi','nDaughterCharged(211)')
# vm.addAlias('n_particle','nDaughterCharged()')
# tag_nParticle = ['n_particle','n_e','n_mu','n_K','n_pi',
#                  'nDaughterNeutralHadrons','nDaughterPhotons']



# Load missing momentum in the event and use a mask 'cleanMask':
#ma.fillParticleListFromROE('nu_e:missing', '', maskName='my_mask',
#                           sourceParticleListName='anti-B0:Dl', useMissing = True, path=main_path)

# fit B vertex on the tag-side
vx.TagV("anti-B0:Dl", fitAlgorithm="Rave", maskName='my_mask', path=main_path)

# Continuum Suppression
ma.buildContinuumSuppression(list_name="anti-B0:Dl", roe_mask="my_mask", path=main_path)

vm.addAlias('KSFWV1','KSFWVariables(et)')
vm.addAlias('KSFWV2','KSFWVariables(mm2)')
vm.addAlias('KSFWV3','KSFWVariables(hso00)')
vm.addAlias('KSFWV4','KSFWVariables(hso01)')
vm.addAlias('KSFWV5','KSFWVariables(hso02)')
vm.addAlias('KSFWV6','KSFWVariables(hso03)')
vm.addAlias('KSFWV7','KSFWVariables(hso04)')
vm.addAlias('KSFWV8','KSFWVariables(hso10)')
vm.addAlias('KSFWV9','KSFWVariables(hso12)')
vm.addAlias('KSFWV10','KSFWVariables(hso14)')
vm.addAlias('KSFWV11','KSFWVariables(hso20)')
vm.addAlias('KSFWV12','KSFWVariables(hso22)')
vm.addAlias('KSFWV13','KSFWVariables(hso24)')
vm.addAlias('KSFWV14','KSFWVariables(hoo0)')
vm.addAlias('KSFWV15','KSFWVariables(hoo1)')
vm.addAlias('KSFWV16','KSFWVariables(hoo2)')
vm.addAlias('KSFWV17','KSFWVariables(hoo3)')
vm.addAlias('KSFWV18','KSFWVariables(hoo4)')
vm.addAlias('CC1','CleoConeCS(1)')
vm.addAlias('CC2','CleoConeCS(2)')
vm.addAlias('CC3','CleoConeCS(3)')
vm.addAlias('CC4','CleoConeCS(4)')
vm.addAlias('CC5','CleoConeCS(5)')
vm.addAlias('CC6','CleoConeCS(6)')
vm.addAlias('CC7','CleoConeCS(7)')
vm.addAlias('CC8','CleoConeCS(8)')
vm.addAlias('CC9','CleoConeCS(9)')

CSVariables = [
    'isContinuumEvent',
    "R2",
    "thrustBm",
    "thrustOm",
    "cosTBTO",
    "cosTBz",
    "KSFWV1",
    "KSFWV2",
    "KSFWV3",
    "KSFWV4",
    "KSFWV5",
    "KSFWV6",
    "KSFWV7",
    "KSFWV8",
    "KSFWV9",
    "KSFWV10",
    "KSFWV11",
    "KSFWV12",
    "KSFWV13",
    "KSFWV14",
    "KSFWV15",
    "KSFWV16",
    "KSFWV17",
    "KSFWV18",
    "CC1",
    "CC2",
    "CC3",
    "CC4",
    "CC5",
    "CC6",
    "CC7",
    "CC8",
    "CC9",
]

# call flavor tagging
#ft.flavorTagger(['anti-B0:Dl'],path=main_path)

# perform best candidate selection
#b2.set_random_seed('Belle2')
#ma.rankByLowest('anti-B0:Dl',variable='vtxReChi2',numBest=1,path=main_path)
#vm.addAlias('Chi2_rank','extraInfo(vtxReChi2_rank)')




# Write variables to Ntuples
vm.addAlias('cos_pV','cosAngleBetweenMomentumAndVertexVector')
vm.addAlias('cos_pB','cosThetaBetweenParticleAndNominalB')
vm.addAlias('TagVReChi2','formula(TagVChi2/TagVNDF)')
vm.addAlias('TagVReChi2IP','formula(TagVChi2IP/TagVNDF)')


# Kinematic variables in CMS
cms_kinematics = vu.create_aliases(vc.kinematics, "useCMSFrame({variable})", "CMS")
roe_cms_kinematics = vu.create_aliases(roe_kinematics, "useCMSFrame({variable})", "CMS")
roe_cms_MC_kinematics = vu.create_aliases(roe_MC_kinematics, "useCMSFrame({variable})", "CMS")


# ma.applyCuts('anti-B0:Dl', '5<roeMbc(my_mask) and 4.3<B0_CMS2_weMbc and \
#               -5<B0_roeDeltae_my_mask<2 and -3<B0_CMS0_weDeltae<2 and \
#               abs(roeCharge(my_mask))<3 and \
#               0.2967<Lab5_weMissPTheta<2.7925 and 0.2967<Lab6_weMissPTheta<2.7925 and \
#               0<TagVReChi2<100 and 0<TagVReChi2IP<100', path=main_path)


# GenMCTagTool
GenMCTags=['B0Mode','Bbar0Mode','BminusMode','BplusMode',
           'DminusMode','DplusMode','TauminusMode','TauplusMode',]


b_vars = vu.create_aliases_for_selected(
    list_of_variables= cms_kinematics + vc.kinematics + vc.deltae_mbc + vc.inv_mass
    + roe_Mbc_Deltae + roe_cms_kinematics + roe_kinematics + roe_E_Q #+ roe_MC_kinematics
    + roe_multiplicities + roe_nCharged + CSVariables + we #+ roel_DOCA_Chi2
    + vertex_vars + vc.flight_info + vc.mc_flight_info
    + ['vtxDDSig','DecayHash','DecayHashEx','TagVReChi2','TagVReChi2IP', 'roel_DistanceSig_dis',
       'genMotherPDG','mcErrors', 'mcP','pErr','mcPDG','mcPhi','phi','phiErr'],
    decay_string='^anti-B0:Dl -> D+:K2pi mu-:mymu',
    prefix=['B0'])

D_vars = vu.create_aliases_for_selected(
    list_of_variables= cms_kinematics + vc.kinematics + vc.dalitz_3body + vc.inv_mass 
    + vc.flight_info + vc.mc_flight_info + Daughters_vars + vertex_vars
    + ['genMotherPDG','mcErrors', 'mcP','pErr','mcPDG','mcPhi','phi','phiErr',
       'dM','BFM','A1FflightDistanceSig_IP'],
    decay_string='anti-B0:Dl -> ^D+:K2pi mu-:mymu',
    prefix=['D'])

l_vars = vu.create_aliases_for_selected(
    list_of_variables= cms_kinematics + vc.kinematics# + electron_id_weights
    + ['genMotherPDG','mcErrors', 'mcP','pErr','mcPDG','mcPhi','phi','phiErr',
       'dM','isBremsCorrected','nPXDHits','isCloneTrack'],
    decay_string='anti-B0:Dl -> D+:K2pi ^mu-:mymu',
    prefix=['mu'])

#nu_vars = vu.create_aliases_for_selected(
#    list_of_variables= cms_kinematics + vc.inv_mass,
#    decay_string='^nu_e:missing',
#    prefix=['nu'])

#tag_vars = vu.create_aliases_for_selected(
#    list_of_variables= cms_kinematics + vc.kinematics + vc.inv_mass + tag_nParticle,
#    decay_string='^B0:tagFromROE',
#    prefix=['tag'])


candidate_vars = ['Ecms','nElectrons90','nMuons90','ROEeidBDT','ROEmuidBDT'] + b_vars + D_vars + l_vars

ma.variablesToNtuple('anti-B0:Dl', candidate_vars,
                     filename=output_file, treename='B0', path=main_path)

b2.process(path=main_path)


