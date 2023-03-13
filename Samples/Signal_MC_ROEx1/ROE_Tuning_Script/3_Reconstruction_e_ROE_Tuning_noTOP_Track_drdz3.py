# +
import basf2 as b2
import modularAnalysis as ma
from variables import variables as vm
import variables.collections as vc
import variables.utils as vu
import vertex as vx
import stdV0s
from variables.MCGenTopo import mc_gen_topo


import decayHash
from decayHash import DecayHashMap
import root_pandas
import pandas as pd
import numpy as np
import sys


#input_file = [''] # for grid
# the basf2 command line: 
# bsub -q l 'basf2 /current/directory/3_Reconstruction_e.py  D_tau/Dst_l 10k/50k'
        
decaymode = sys.argv[1]
filesize = sys.argv[2]
input_file = f'/home/belle/zhangboy/R_D/Signal_MC_ROEx1/B2{decaymode}_nu/MC/MC_e_{filesize}.root'
    
ncdchits_tuning = [0, 1, 20]
ncdchits_tuned = 0
p_t_tuning = [0, 0.025, 0.05]
p_t_tuned = 0
pvalue_tuning = [0, 0.0005, 0.005]
pvalue_tuned = 0.0005
dr1_tuning = 36 #[16, 25, 36] # 36, 64, 100
dz1_tuning = 225 #[64, 100, 225] # 225, 400, 625
dr2 = 64  #[36, 64, 100]
dz2 = 225 #[100, 225, 400]
dr3_tuning = [49]
dz3_tuning = [4, 9]
dr4 = 9   #[4, 9, 16]
dz4 = 16  #[9, 16, 25]
dr5 = 0.64#[0.36, 0.64, 1]
dz5 = 1   #[0.64, 1, 2.25]
bBS_tuning = [0.3,0.4,0.5,0.6]
bBS_tuned = 0.5
hSOS_tuning = [0.3,0.4,0.5,0.6]
hSOS_tuned = 0.6

x=0
small_dfs = []
for dr3 in dr3_tuning:
    for dz3 in dz3_tuning:
    
        # Define the path
        main_path = b2.Path()

        output_file = f'/home/belle/zhangboy/R_D/Signal_MC_ROEx1/B2{decaymode}_nu/Ntuples/MC_e_ROE_Legacy_noTOP_{filesize}_test_' + str(x) + '.root'
        output_hash = f'/home/belle/zhangboy/R_D/Signal_MC_ROEx1/B2{decaymode}_nu/Ntuples/hashmap_MC_e_ROE_Legacy_noTOP_{filesize}_test_' + str(x) + '.root'

        ma.inputMdstList(environmentType='default', filelist=input_file, path=main_path)

        goodTrack = 'abs(dz)<2.0 and dr<0.5 and thetaInCDCAcceptance and nCDCHits>0'

        ma.fillParticleList('pi+:mypi', cut=goodTrack + ' and pionID > 0.1', path=main_path)
        ma.fillParticleList('K-:myk', cut=goodTrack + ' and kaonID > 0.5', path=main_path)
        #stdCharged.stdE(listtype='UniformEff80', method='bdt', classification='binary', path=main_path)
        #stdCharged.stdMu(listtype='UniformEff80', method='bdt', classification='binary', path=main_path)
        ma.fillParticleList('e-:uncorrected', cut=goodTrack + ' and electronID_noTOP>0.9 and nPXDHits>0',path=main_path)
        ma.fillParticleList('mu-:mymu', cut=goodTrack + ' and muonID>0.9 and nPXDHits>0 and inKLMAcceptance',path=main_path)

        # apply Bremsstrahlung correction to electrons
        vm.addAlias(
            "goodFWDGamma", "passesCut(clusterReg == 1 and clusterE > 0.075)"
        )
        vm.addAlias(
            "goodBRLGamma", "passesCut(clusterReg == 2 and clusterE > 0.05)"
        )
        vm.addAlias(
            "goodBWDGamma", "passesCut(clusterReg == 3 and clusterE > 0.1)"
        )
        vm.addAlias(
            "goodGamma", "passesCut(goodFWDGamma or goodBRLGamma or goodBWDGamma)"
        )
        ma.fillParticleList("gamma:brems", "goodGamma",path=main_path)
        ma.correctBrems(outputList="e-:corrected", inputList="e-:uncorrected",
                        gammaList="gamma:brems", path=main_path)
        vm.addAlias("isBremsCorrected", "extraInfo(bremsCorrected)")

        #stdV0s.stdKshorts(path=main_path)

        # Event Kinematics
        ma.buildEventKinematics(fillWithMostLikely=True,path=main_path) 




        # Reconstruct D
        #Dcuts = '1.8 < M < 1.9'
        Dcuts = '1.855 <M< 1.885'
        ma.reconstructDecay('D+:K2pi -> K-:myk pi+:mypi pi+:mypi', cut=Dcuts, path=main_path)
        #ma.reconstructDecay('D+:K3pi -> K-:myk pi+:mypi pi+:mypi pi0:eff60_May2020Fit', cut=Dcuts,path=main_path)
        #ma.reconstructDecay('D+:Ks3pi -> K_S0:merged pi+:mypi pi+:mypi pi-:mypi', cut=Dcuts,path=main_path)
        #ma.copyLists('D+:myD',['D+:K2pi','D+:Ks3pi'],path=main_path) #'D+:K3pi'
        ma.variablesToExtraInfo('D+:K2pi', variables={'M':'D_BFM','InvM':'D_BFInvM'},option=0, path=main_path)
        vm.addAlias('BFM','extraInfo(D_BFM)')
        vm.addAlias('BFInvM','extraInfo(D_BFInvM)')

        # vertex fitting D
        vx.treeFit('D+:K2pi', conf_level=0.0, updateAllDaughters=True, massConstraint=['D+'], ipConstraint=False, path=main_path)
        vm.addAlias('vtxChi2','extraInfo(chiSquared)')
        vm.addAlias('vtxNDF','extraInfo(ndf)')
        vm.addAlias('vtxReChi2','formula(vtxChi2/vtxNDF)')

        # Reconstruct B
        ma.reconstructDecay('anti-B0:De -> D+:K2pi e-:corrected ?addbrems', cut='', path=main_path)
        vx.treeFit('anti-B0:De', conf_level=0.0, updateAllDaughters=False, massConstraint=[], ipConstraint=True, path=main_path)

        # Calculate the distance between vertices De and D+
        vm.addAlias('vtxDD', 'vertexDistanceOfDaughter(0, D+)')
        vm.addAlias('vtxDDSig', 'vertexDistanceOfDaughterSignificance(0, D+)')

        # Calculate DOCA(D,l)
        ma.calculateDistance('anti-B0:De', 'anti-B0:De -> ^D+:K2pi ^e-:corrected', "vertextrack", path=main_path)
        vm.addAlias('Distance', 'extraInfo(CalculatedDistance)')
        vm.addAlias('DistanceError', 'extraInfo(CalculatedDistanceError)')
        vm.addAlias('DistanceVector_X', 'extraInfo(CalculatedDistanceVector_X)')
        vm.addAlias('DistanceVector_Y', 'extraInfo(CalculatedDistanceVector_Y)')
        vm.addAlias('DistanceVector_Z', 'extraInfo(CalculatedDistanceVector_Z)')
        vm.addAlias('DistanceCovMatrixXX', 'extraInfo(CalculatedDistanceCovMatrixXX)')
        vm.addAlias('DistanceCovMatrixXY', 'extraInfo(CalculatedDistanceCovMatrixXY)')
        vm.addAlias('DistanceCovMatrixXZ', 'extraInfo(CalculatedDistanceCovMatrixXZ)')
        vm.addAlias('DistanceCovMatrixYX', 'extraInfo(CalculatedDistanceCovMatrixYX)')
        vm.addAlias('DistanceCovMatrixYY', 'extraInfo(CalculatedDistanceCovMatrixYY)')
        vm.addAlias('DistanceCovMatrixYZ', 'extraInfo(CalculatedDistanceCovMatrixYZ)')
        vm.addAlias('DistanceCovMatrixZX', 'extraInfo(CalculatedDistanceCovMatrixZX)')
        vm.addAlias('DistanceCovMatrixZY', 'extraInfo(CalculatedDistanceCovMatrixZY)')
        vm.addAlias('DistanceCovMatrixZZ', 'extraInfo(CalculatedDistanceCovMatrixZZ)')

        distance_vars = [
            'Distance',
            'DistanceError',
            'DistanceVector_X',
            'DistanceVector_Y',
            'DistanceVector_Z',
            'DistanceCovMatrixXX',
            'DistanceCovMatrixXY',
            'DistanceCovMatrixXZ',
            'DistanceCovMatrixYX',
            'DistanceCovMatrixYY',
            'DistanceCovMatrixYZ',
            'DistanceCovMatrixZX',
            'DistanceCovMatrixZY',
            'DistanceCovMatrixZZ'
        ]

        # MC Truth Matching
        ma.matchMCTruth('anti-B0:De', path=main_path)

        # generate the decay string
        main_path.add_module('ParticleMCDecayString', listName='anti-B0:De', fileName=output_hash)
        vm.addAlias('DecayHash','extraInfo(DecayHash)')
        vm.addAlias('DecayHashEx','extraInfo(DecayHashExtended)')



        #ma.applyEventCuts(cut='[genUpsilon4S(mcDaughter(0, mcDaughter(0, PDG)))==411] or \
        #                    [genUpsilon4S(mcDaughter(0, mcDaughter(0, PDG)))==-411]', path=main_path)



        # build the ROE
        ma.fillParticleList('gamma:all', '', loadPhotonBeamBackgroundMVA=True,loadPhotonHadronicSplitOffMVA=True, path=main_path)
        ma.buildRestOfEvent('anti-B0:De',fillWithMostLikely=True,path=main_path)
        roe_mask1 = ('my_mask',  f'nCDCHits>={ncdchits_tuned} and thetaInCDCAcceptance and pt>{p_t_tuned} and pValue>={pvalue_tuned} and E<5.5 and \
                    [pt<0.15 and formula(dr**2/{dr1_tuning}+dz**2/{dz1_tuning})<1] or \
                    [0.15<pt<0.25 and formula(dr**2/{dr2}+dz**2/{dz2})<1] or \
                    [0.25<pt<0.5 and formula(dr**2/{dr3}+dz**2/{dz3})<1] or \
                    [0.5<pt<1 and formula(dr**2/{dr4}+dz**2/{dz4})<1] or \
                    [pt>1 and formula(dr**2/{dr5}+dz**2/{dz5})<1]',
                    f'goodGamma and abs(clusterTiming)<clusterErrorTiming and \
                    [clusterE<0.1 and clusterReg==1 and beamBackgroundSuppression>0.2 and hadronicSplitOffSuppression>0 and minC2TDist>25] or \
                    [clusterE<0.1 and clusterReg==2 and beamBackgroundSuppression>0.2 and minC2TDist>25] or \
                    [0.1<clusterE<0.2 and clusterReg==1 and beamBackgroundSuppression>0.4 and minC2TDist>25 and clusterZernikeMVA>0.05] or \
                    [0.1<clusterE<0.2 and clusterReg==2 and beamBackgroundSuppression>0.4 and minC2TDist>25 and clusterZernikeMVA>0.05] or \
                    [0.1<clusterE<0.2 and clusterReg==3 and beamBackgroundSuppression>0.4 and minC2TDist>25 and clusterZernikeMVA>0.05] or \
                    [0.2<clusterE<0.5 and clusterReg==1 and beamBackgroundSuppression>0.4 and minC2TDist>20 and clusterZernikeMVA>0.05] or \
                    [0.2<clusterE<0.5 and clusterReg==2 and beamBackgroundSuppression>0.4 and minC2TDist>20 and clusterZernikeMVA>0.05] or \
                    [0.2<clusterE<0.5 and clusterReg==3 and beamBackgroundSuppression>0.4 and minC2TDist>20 and clusterZernikeMVA>0.05] or \
                    [clusterE>0.5 and clusterReg==1 and beamBackgroundSuppression>0.5] or \
                    [clusterE>0.5 and clusterReg==2 and beamBackgroundSuppression>0.5] or \
                    [clusterE>0.5 and clusterReg==3 and beamBackgroundSuppression>0.5]')




        ma.appendROEMasks('anti-B0:De', [roe_mask1], path=main_path)

        # creates V0 particle lists and uses V0 candidates to update/optimize the Rest Of Event
        ma.updateROEUsingV0Lists('anti-B0:De', mask_names='my_mask', default_cleanup=True, selection_cuts=None,
                                 apply_mass_fit=True, fitter='treefit', path=main_path)

        # Load ROE as a particle and use a mask 'my_mask':
        ma.fillParticleListFromROE('B0:tagFromROE', '', maskName='my_mask',
          sourceParticleListName='anti-B0:De', path=main_path)



        # ROE variables
        roe_kinematics = ["roeE(my_mask)", "roeP(my_mask)", "roePx(my_mask)",
                          "roePy(my_mask)","roePz(my_mask)","roePt(my_mask)",]
        roe_MC_kinematics = ['roeMC_E','roeMC_M','roeMC_P','roeMC_PTheta','roeMC_Pt','roeMC_Px','roeMC_Py','roeMC_Pz']

        roe_E_Q = ['roeCharge(my_mask)', 'roeNeextra(my_mask)','roeEextra(my_mask)',]

        roe_multiplicities = ["nROE_Charged(my_mask)",'nROE_ECLClusters(my_mask)',
                              'nROE_NeutralECLClusters(my_mask)','nROE_KLMClusters',
                              'nROE_NeutralHadrons(my_mask)',"nROE_Photons(my_mask)",
                              'nROE_Tracks(my_mask)',]

        vm.addAlias('nROE_e','nROE_Charged(my_mask, 11)')
        vm.addAlias('nROE_mu','nROE_Charged(my_mask, 13)')
        vm.addAlias('nROE_K','nROE_Charged(my_mask, 321)')
        vm.addAlias('nROE_pi','nROE_Charged(my_mask, 211)')
        roe_nCharged = ['nROE_e','nROE_mu','nROE_K','nROE_pi',]



        vm.addAlias('CMS_weDeltae','weDeltae(my_mask,0)')
        vm.addAlias('Lab_weDeltae','weDeltae(my_mask,1)')
        #Option for correctedB_deltae variable should only be 0/1 (CMS/LAB)
        #Option for correctedB_mbc variable should only be 0/1/2 (CMS/LAB/CMS with factor)
        vm.addAlias('CMS_weMbc','weMbc(my_mask,0)')
        vm.addAlias('Lab_weMbc','weMbc(my_mask,1)')
        vm.addAlias('CMS1_weMbc','weMbc(my_mask,2)')
        vm.addAlias('CMS_weMissM2','weMissM2(my_mask,0)')
        vm.addAlias('CMS1_weMissM2','weMissM2(my_mask,1)')
        vm.addAlias('CMS2_weMissM2','weMissM2(my_mask,2)')
        vm.addAlias('Lab_weMissM2','weMissM2(my_mask,5)')
        vm.addAlias('Lab_weMissPTheta','weMissPTheta(my_mask,5)')
        vm.addAlias('Lab1_weMissPTheta','weMissPTheta(my_mask,6)')

        we= ['CMS_weDeltae','CMS1_weMbc','CMS2_weMissM2','Lab_weMissPTheta','Lab1_weMissPTheta',]


        vm.addAlias('n_e','nDaughterCharged(11)')
        vm.addAlias('n_mu','nDaughterCharged(13)')
        vm.addAlias('n_K','nDaughterCharged(321)')
        vm.addAlias('n_pi','nDaughterCharged(211)')
        vm.addAlias('n_particle','nDaughterCharged()')
        tag_nParticle = ['n_particle','n_e','n_mu','n_K','n_pi',
                         'nDaughterNeutralHadrons','nDaughterPhotons']



        # Load missing momentum in the event and use a mask 'cleanMask':
        #ma.fillParticleListFromROE('nu_e:missing', '', maskName='my_mask',
        #                           sourceParticleListName='anti-B0:De', useMissing = True, path=main_path)

        # fit B vertex on the tag-side
        vx.TagV("anti-B0:De", fitAlgorithm="Rave", maskName='my_mask', path=main_path)

        # Continuum Suppression
        ma.buildContinuumSuppression(list_name="anti-B0:De", roe_mask="my_mask", path=main_path)

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
        #ft.flavorTagger(['anti-B0:De'],path=main_path)



        # Write variables to Ntuples
        vm.addAlias('cos_pV','cosAngleBetweenMomentumAndVertexVector')
        vm.addAlias('cos_pB','cosThetaBetweenParticleAndNominalB')
        vm.addAlias('TagVReChi2','formula(TagVChi2/TagVNDF)')
        vm.addAlias('TagVReChi2IP','formula(TagVChi2IP/TagVNDF)')

        # Kinematic variables in CMS
        cms_kinematics = vu.create_aliases(vc.kinematics, "useCMSFrame({variable})", "CMS")
        roe_cms_kinematics = vu.create_aliases(roe_kinematics, "useCMSFrame({variable})", "CMS")
        roe_cms_MC_kinematics = vu.create_aliases(roe_MC_kinematics, "useCMSFrame({variable})", "CMS")



        ma.applyCuts('anti-B0:De', '4.9<Mbc<5.3 and CMS_E<5.3 and 0.2967<Lab_weMissPTheta<2.7925 and 0<TagVReChi2<100 and 0<TagVReChi2IP<100 and R2<0.3', path=main_path)


        # perform best candidate selection
        #b2.set_random_seed('Belle2')
        #ma.rankByLowest('anti-B0:De',variable='vtxReChi2',numBest=1,path=main_path)
        #vm.addAlias('Chi2_rank','extraInfo(vtxReChi2_rank)')

        b_vars = vu.create_aliases_for_selected(
            list_of_variables= cms_kinematics + vc.kinematics + vc.deltae_mbc + vc.inv_mass +vc.mc_truth
            + ['dM','Q','dQ','recMissM2','cos_pV','cos_pB','vtxReChi2','vtxNDF','vtxDD', 
               'vtxDDSig','DecayHash','DecayHashEx',"roeMbc(my_mask)", "roeM(my_mask)","roeDeltae(my_mask)",]
            + roe_cms_kinematics + roe_kinematics + roe_MC_kinematics + roe_cms_MC_kinematics
            + roe_E_Q + roe_multiplicities + roe_nCharged + CSVariables + we
            + vc.tag_vertex + vc.mc_tag_vertex + distance_vars, # + ft.flavor_tagging
            decay_string='^anti-B0:De -> D+:K2pi e-:corrected',
            prefix=['B0'])

        D_vars = vu.create_aliases_for_selected(
            list_of_variables= cms_kinematics + vc.kinematics + vc.mc_truth
            + ['dM','BFM','BFInvM','vtxReChi2','vtxNDF',],
            decay_string='anti-B0:De -> ^D+:K2pi e-:corrected',
            prefix=['D'])

        e_vars = vu.create_aliases_for_selected(
            list_of_variables= cms_kinematics + vc.kinematics + vc.inv_mass + vc.mc_truth
            + ['dM','isBremsCorrected'],
            decay_string='anti-B0:De -> D+:K2pi ^e-:corrected',
            prefix=['e'])

        #nu_vars = vu.create_aliases_for_selected(
        #    list_of_variables= cms_kinematics + vc.inv_mass,
        #    decay_string='^nu_e:missing',
        #    prefix=['nu'])

        tag_vars = vu.create_aliases_for_selected(
            list_of_variables= cms_kinematics + vc.kinematics + vc.inv_mass + tag_nParticle,
            decay_string='^B0:tagFromROE',
            prefix=['tag'])

        candidate_vars = b_vars + D_vars + e_vars + tag_vars# + nu_vars
        event_vars=['Ecms', 'IPX', 'IPY', 'IPZ'] + vc.event_kinematics
        ma.variablesToNtuple('anti-B0:De', event_vars + candidate_vars, #+ mc_gen_topo(30),
                             filename=output_file, treename='B0', path=main_path)
        #ma.variablesToNtuple('','missingMass2OfEvent', filename=output_file, treename='event', path=main_path)

        b2.process(path=main_path)
        print(b2.statistics)


        # Offline analysis
        # DecayHash

        df = root_pandas.read_root(output_file,key='B0')
        hashmap = DecayHashMap(output_hash, removeRadiativeGammaFlag=False)
        hashmap2 = DecayHashMap(output_hash, removeRadiativeGammaFlag=True)


        mode_dict={}
        mode_dict['D_tau_nu']=['511 (-> -411 -15 (-> -11 12 -16) 16)','-511 (-> 411 15 (-> 11 -12 16) -16)',
                               '511 (-> -411 -15 (-> -13 14 -16) 16)','-511 (-> 411 15 (-> 13 -14 16) -16)']
        mode_dict['D_e_nu']=['511 (-> -411 -11 12)','-511 (-> 411 11 -12)']
        mode_dict['D_mu_nu']=['511 (-> -411 -13 14)','-511 (-> 411 13 -14)']
        mode_dict['Dst_tau_nu']=['511 (-> -413 -15 (-> -11 12 -16) 16)','-511 (-> 413 15 (-> 11 -12 16) -16)',
                                 '511 (-> -413 -15 (-> -13 14 -16) 16)','-511 (-> 413 15 (-> 13 -14 16) -16)']
        mode_dict['Dst_e_nu']=['511 (-> -413 -11 12)','-511 (-> 413 11 -12)']
        mode_dict['Dst_mu_nu']=['511 (-> -413 -13 14)','-511 (-> 413 13 -14)']
        mode_dict['Dstst_tau_nu']=['511 (-> -10413 -15 16)','-511 (-> 10413 15 -16)',
                                   '511 (-> -10411 -15 16)','-511 (-> 10411 15 -16)',
                                   '511 (-> -20413 -15 16)','-511 (-> 20413 15 -16)',
                                   '511 (-> -415 -15 16)',  '-511 (-> 415 15 -16)',
                                   '521 (-> -10423 -15 16)','-521 (-> 10423 15 -16)',
                                   '521 (-> -10421 -15 16)','-521 (-> 10421 15 -16)',
                                   '521 (-> -20423 -15 16)','-521 (-> 20423 15 -16)',
                                   '521 (-> -425 -15 16)',  '-521 (-> 425 15 -16)']

        mode_dict['Dstst_e_nu']=['511 (-> -10413 -11 12)','-511 (-> 10413 11 -12)',
                                 '511 (-> -10411 -11 12)','-511 (-> 10411 11 -12)',
                                 '511 (-> -20413 -11 12)','-511 (-> 20413 11 -12)',
                                 '511 (-> -415 -11 12)',  '-511 (-> 415 11 -12)',
                                 '521 (-> -10423 -11 12)','-521 (-> 10423 11 -12)',
                                 '521 (-> -10421 -11 12)','-521 (-> 10421 11 -12)',
                                 '521 (-> -20423 -11 12)','-521 (-> 20423 11 -12)',
                                 '521 (-> -425 -11 12)',  '-521 (-> 425 11 -12)',
                                 '511 (-> -411 221 -11 12)','-511 (-> 411 221 11 -12)',
                                 '511 (-> -411 111 -11 12)','-511 (-> 411 111 11 -12)',
                                 '511 (-> -411 111 111 -11 12)','-511 (-> 411 111 111 11 -12)',
                                 '511 (-> -411 -211 211 -11 12)','-511 (-> 411 211 -211 11 -12)',
                                 '511 (-> -413 221 -11 12)','-511 (-> 413 221 11 -12)',
                                 '511 (-> -413 111 -11 12)','-511 (-> 413 111 11 -12)',
                                 '511 (-> -413 111 111 -11 12)','-511 (-> 413 111 111 11 -12)',
                                 '511 (-> -413 -211 211 -11 12)','-511 (-> 413 211 -211 11 -12)',
                                 '511 (-> -421 -211 -11 12)','-511 (-> 421 211 11 -12)',
                                 '511 (-> -423 -211 -11 12)','-511 (-> 423 211 11 -12)']

        mode_dict['Dstst_mu_nu']=['511 (-> -10413 -13 14)','-511 (-> 10413 13 -14)',
                                  '511 (-> -10411 -13 14)','-511 (-> 10411 13 -14)',
                                  '511 (-> -20413 -13 14)','-511 (-> 20413 13 -14)',
                                  '511 (-> -415 -13 14)',  '-511 (-> 415 13 -14)',
                                  '521 (-> -10423 -13 14)','-521 (-> 10423 13 -14)',
                                  '521 (-> -10421 -13 14)','-521 (-> 10421 13 -14)',
                                  '521 (-> -20423 -13 14)','-521 (-> 20423 13 -14)',
                                  '521 (-> -425 -13 14)',  '-521 (-> 425 13 -14)',
                                  '511 (-> -411 221 -13 14)','-511 (-> 411 221 13 -14)',
                                  '511 (-> -411 111 -13 14)','-511 (-> 411 111 13 -14)',
                                  '511 (-> -411 111 111 -13 14)','-511 (-> 411 111 111 13 -14)',
                                  '511 (-> -411 -211 211 -13 14)','-511 (-> 411 211 -211 13 -14)',
                                  '511 (-> -413 221 -13 14)','-511 (-> 413 221 13 -14)',
                                  '511 (-> -413 111 -13 14)','-511 (-> 413 111 13 -14)',
                                  '511 (-> -413 111 111 -13 14)','-511 (-> 413 111 111 13 -14)',
                                  '511 (-> -413 -211 211 -13 14)','-511 (-> 413 211 -211 13 -14)',
                                  '511 (-> -421 -211 -13 14)','-511 (-> 421 211 13 -14)',
                                  '511 (-> -423 -211 -13 14)','-511 (-> 423 211 13 -14)']



        def found(modes,row):
            for mode in modes:
                decaytree = decayHash.Belle2.DecayTree(mode)
                if hashmap2.get_original_decay(row["B0_DecayHash"],row["B0_DecayHashEx"]).find_decay(decaytree):
                    return True
            return False

        def decay_mode(row):
            for name,modes in mode_dict.items():
                if found(modes,row):
                    return name
            return 'bkg'


        # Calculate MM2 and cos_D_l

        def withROE_mm2_2(data):
            # Energy
            E_B = data.Ecms.mean()/2
            E_Y = data.D_CMS_E + data.e_CMS_E
            Mbc_roe = data.B0_roeMbc_my_mask
            # Calculating M_Y^2
            p_Yx = data.D_CMS_px + data.e_CMS_px
            p_Yy = data.D_CMS_py + data.e_CMS_py
            p_Yz = data.D_CMS_pz + data.e_CMS_pz
            p_Y2 = p_Yx**2 + p_Yy**2 + p_Yz**2
            m_Y2 = E_Y**2 - p_Y2
            # dot product
            p_xdot = data.B0_CMS_roePx_my_mask * p_Yx
            p_ydot = data.B0_CMS_roePy_my_mask * p_Yy
            p_zdot = data.B0_CMS_roePz_my_mask * p_Yz
            p_dot = p_xdot + p_ydot + p_zdot
            # Calculating the final quantities
            withROE_missing_m2 = Mbc_roe**2 + m_Y2 - 2*E_B*E_Y - 2*p_dot
            return withROE_missing_m2


        def cos_D_l(df):
            dot_product=df.D_CMS_px*df.e_CMS_px+df.D_CMS_py*df.e_CMS_py+df.D_CMS_pz*df.e_CMS_pz
            magnitude=df.D_CMS_p*df.e_CMS_p
            cos=dot_product/magnitude
            return cos


        df['DecayMode'] = df.apply(decay_mode, axis=1).astype('category')
        df['MM2'] = withROE_mm2_2(df)
        df['cos_D_l'] = cos_D_l(df)
        df['B0_mcPDG'] = df['B0_mcPDG'].fillna(0)
        df['B0_isSignal'] = df['B0_isSignal'].fillna(-1)
        df['D_isSignal'] = df['D_isSignal'].fillna(-1)
        df['e_isSignal'] = df['e_isSignal'].fillna(-1)

        df.eval('B_D_ReChi2 = B0_vtxReChi2 + 2 * D_vtxReChi2', inplace=True)
        df.eval('p_D_l = D_CMS_p + e_CMS_p', inplace=True)


        # Best Candidate Selection
        df_bestSelected=df.loc[df.groupby(['__experiment__','__run__','__event__','__production__']).B_D_ReChi2.idxmin()]


        # Classification
        # B decay
        B_meson='B0_mcPDG==511 or B0_mcPDG==-511 or B0_mcPDG==521 or B0_mcPDG==-521'
    #    tau_modes = 'DecayMode=="D_tau_nu" or DecayMode=="Dst_tau_nu" or DecayMode=="Dstst_tau_nu"'
        e_modes = 'DecayMode=="D_e_nu" or DecayMode=="Dst_e_nu" or DecayMode=="Dstst_e_nu"'
    #    mu_modes = 'DecayMode=="D_mu_nu" or DecayMode=="Dst_mu_nu" or DecayMode=="Dstst_mu_nu"'
    #    bkg_misIdAndTwoD = 'DecayMode=="bkg"'
        # Not B decay
        bkg_combinatorial_oneBsignal = 'B0_mcPDG==300553 and DecayMode!="bkg"'
    #    bkg_combinatorial_noBsignal = 'B0_mcPDG==300553 and DecayMode=="bkg"'
    #    bkg_BDaughterDecay = 'B0_mcPDG!=511 and B0_mcPDG!=-511 and B0_mcPDG!=521 and B0_mcPDG!=-521 \
    #    and B0_mcPDG!=300553 and B0_mcPDG!=0 and B0_isContinuumEvent!=1'
        bkg_misId2='B0_mcPDG==0'
    #    bkg_continuum = 'B0_isContinuumEvent==1'

        df_B_mother = df_bestSelected.query(B_meson)
    #    df_tau = df_B_mother.query(tau_modes)
        df_e = df_B_mother.query(e_modes)
    #    df_mu = df_B_mother.query(mu_modes)
    #    df_bkg_misIdAndTwoD = df_B_mother.query(bkg_misIdAndTwoD)
        # Not B decay
        df_bkg_comb_oneBsignal = df_bestSelected.query(bkg_combinatorial_oneBsignal)
    #    df_bkg_comb_noBsignal = df_bestSelected.query(bkg_combinatorial_noBsignal)
    #    df_bkg_BDaughterDecay = df_bestSelected.query(bkg_BDaughterDecay)
        df_bkg_misId2 = df_bestSelected.query(bkg_misId2)
    #    df_bkg_continuum = df_bestSelected.query(bkg_continuum)

        D_e_nu=df_e.query('DecayMode=="D_e_nu"')
    #    D_mu_nu=df_mu.query('DecayMode=="D_mu_nu"')
    #    D_tau_nu=df_tau.query('DecayMode=="D_tau_nu"')
    #    Dst_e_nu=df_e.query('DecayMode=="Dst_e_nu"')
    #    Dst_mu_nu=df_mu.query('DecayMode=="Dst_mu_nu"')
    #    Dst_tau_nu=df_tau.query('DecayMode=="Dst_tau_nu"')
    #    Dstst_e_nu=df_e.query('DecayMode=="Dstst_e_nu"')
    #    Dstst_mu_nu=df_mu.query('DecayMode=="Dstst_mu_nu"')
    #    Dstst_tau_nu=df_tau.query('DecayMode=="Dstst_tau_nu"')


        # Calculate signal and background statistics and store them in the small_df as performance indicators


        cuts = {'roeMbc528_':'B0_roeMbc_my_mask>5.28','roeMbc526_':'B0_roeMbc_my_mask>5.26'}
        variables = ['B0_roeDeltae_my_mask','B0_CMS2_weMissM2']
        d = {}
        for j in cuts:
            components = {'Denu_':D_e_nu.query(cuts[j]), 'comb_':df_bkg_comb_oneBsignal.query(cuts[j]),
                          'misId_':df_bkg_misId2.query(cuts[j])}
            for i in components:
                d[i+j+'count']=components[i][variables[0]].count()
                d[i+j+'roeDeltae_mean']=components[i][variables[0]].mean()
                d[i+j+'roeDeltae_std']=components[i][variables[0]].std()
                d[i+j+'MM2_mean']=components[i][variables[1]].mean()
                d[i+j+'MM2_std']=components[i][variables[1]].std()


        # store the mask tuning parameters in the small_df
        #d['bBS'] = bBS
        #d['hSOS'] = hSOS
        d['ncdchits'] = ncdchits_tuned
        d['p_t'] = p_t_tuned
        d['pvalue'] = pvalue_tuned
        d['dr1'] = dr1_tuning
        d['dz1'] = dz1_tuning
        d['dr3'] = dr3
        d['dz3'] = dz3

        small_df = pd.DataFrame(data=d,index=[x])
        small_dfs.append(small_df)
        x+=1
        
large_df = pd.concat(small_dfs)
large_df.to_csv(f'/home/belle/zhangboy/R_D/Signal_MC_ROEx1/B2{decaymode}_nu/Ntuples/MaskTuning_Track_drdz3_4.csv')


