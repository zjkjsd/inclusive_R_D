# -*- coding: utf-8 -*-
# +
import basf2 as b2
import modularAnalysis as ma
from variables import variables as vm
import variables.collections as vc
import variables.utils as vu
import vertex as vx
import argparse

def argparser():
    """
    Parse options as command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-pdt', "--printMCDecayTree",action="store_true",
                        help="print the MC truth decay tree for each event")
    return parser


if __name__ == "__main__":
    
    args = argparser().parse_args()


    analysis_gt = ma.getAnalysisGlobaltag()
    b2.B2INFO(f"Appending analysis GT: {analysis_gt}")
    b2.conditions.append_globaltag(analysis_gt)
    b2.conditions.prepend_globaltag('pid_nn_release06_Kpi')
    b2.conditions.prepend_globaltag('chargedpidmva_rel6_v5')

    # b2.conditions.prepend_globaltag('data_beam_conditions_proc13prompt')
    # b2.conditions.prepend_globaltag('data_trigger_conditions_proc13')

    # Define the path
    main_path = b2.Path()

    input_file = '/group/belle2/dataprod/MC/MC15ri/mixed/sub00/mdst_000001_prod00024821_task10020000001.root'
    output_file = 'MC_e_control.root'

    ma.inputMdstList(filelist=input_file, path=main_path)

    goodTrack = 'abs(dz)<2 and dr<0.5 and thetaInCDCAcceptance and nCDCHits>0'

    # kinematic vars in CMS frame
    cms_kinematics = vu.create_aliases(vc.kinematics, "useCMSFrame({variable})", "CMS")
    # cms_mc_kinematics = vu.create_aliases(vc.mc_kinematics, "useCMSFrame({variable})", "CMS")
    # cms_momentum_uncertainty = vu.create_aliases(vc.momentum_uncertainty, "useCMSFrame({variable})", "CMS")

    ma.fillParticleList('pi+:mypi', cut=goodTrack + ' and nSVDHits>0 and pionIDNN > 0.1', path=main_path)
    ma.fillParticleList('K-:myk', cut=goodTrack + ' and nPXDHits>0 and kaonIDNN > 0.9', path=main_path)

    # ----------------------------------
    # Fill example standard lepton list.
    # ----------------------------------

    # For electrons, we show the case in which a Bremsstrahlung correction
    # is applied first to get the 4-momentum right,
    # and the resulting particle list is passed as input to the stdE list creator.
    ma.fillParticleList("e+:uncorrected",
                        cut=goodTrack + ' and nPXDHits>0',  # NB: whichever cut is set here, will be inherited by the std electrons.
                        path=main_path)

    ma.fillParticleList('gamma:all', '', path=main_path)
    ma.getBeamBackgroundProbability('gamma:all','MC15ri', path=main_path)
    ma.getFakePhotonProbability('gamma:all','MC15ri', path=main_path)

    ma.fillParticleList("gamma:bremsinput", cut="0.05<clusterE<2.5",path=main_path)
    ma.applyCuts('gamma:bremsinput','beamBackgroundSuppression>0.2', path=main_path)
    ma.correctBremsBelle(outputListName="e+:corrected",
                         inputListName="e+:uncorrected",
                         gammaListName="gamma:bremsinput",
                         angleThreshold=0.15,
                         multiplePhotons=True,
                         path=main_path)
    # ma.correctBrems(outputList="e+:corrected",
    #                 inputList="e+:uncorrected",
    #                 gammaList="gamma:bremsinput", 
    #                 maximumAcceptance=3.0, 
    #                 multiplePhotons=False, 
    #                 usePhotonOnlyOnce=True, 
    #                 path=main_path)
    vm.addAlias("isBremsCorrected", "extraInfo(bremsCorrected)")

    ma.fillParticleList("mu+:mymu",
                        cut=goodTrack + " and nPXDHits>0 and inKLMAcceptance",
                        path=main_path)

    ma.applyChargedPidMVA(['e+:corrected', 'mu+:mymu'], path=main_path, trainingMode=1, 
                          chargeIndependent=False, binaryHypoPDGCodes=(0, 0))
    #ma.applyChargedPidMVA(['e+:corrected'], path=main_path, trainingMode=0,
    #                      chargeIndependent=False, binaryHypoPDGCodes=(11, 211))

    # ------------------------------------------------------------
    # Add extra cuts on the standard lepton lists and lepton veto.
    # ------------------------------------------------------------
    vm.addAlias('eID','pidChargedBDTScore(11, ALL)')
    ma.applyCuts(f"e-:corrected", "eID>0.9 and p>0.2", path=main_path)
    ma.applyCuts(f"mu-:mymu", "muonID_noSVD>0.9", path=main_path) #pidChargedBDTScore(13, ALL)>0.9

    # select events only containing 1 lepton
    ma.applyEventCuts('formula(nParticlesInList(e-:corrected) + nParticlesInList(mu-:mymu)) == 1', path=main_path)

    # Event Kinematics
    ma.buildEventKinematics(fillWithMostLikely=True,path=main_path) 


    # --------------------------------------------------------
    # Reconstruct D, 1 sigma == 0.005, mean==1.87, vertex fit
    # --------------------------------------------------------
    DMcut1 = '[1.78 <M< 1.96]'
    DMcut2 = '[1.79 <M< 1.82 or 1.92 <M< 1.95 or 1.855 <M< 1.885]'
    ma.reconstructDecay('D+:K2pi -> K-:myk pi+:mypi pi+:mypi', cut=DMcut1, path=main_path)

    ma.variablesToExtraInfo('D+:K2pi', variables={'M':'D_BFM','InvM':'D_BFInvM'},option=0, path=main_path)
    vm.addAlias('BFM','extraInfo(D_BFM)')
    vm.addAlias('BFInvM','extraInfo(D_BFInvM)')

    D_Daughters_vars = []
    pid_vars = ['p','cosTheta','theta','charge','PDG','mcPDG']
    for variable in ['pionIDNN','kaonIDNN','mcErrors','pt']+pid_vars:
        vm.addAlias(f'K_{variable}', f'daughter(0, {variable})')
        vm.addAlias(f'pi1_{variable}', f'daughter(1, {variable})')
        vm.addAlias(f'pi2_{variable}', f'daughter(2, {variable})')
        D_Daughters_vars.append(f'K_{variable}')
        D_Daughters_vars.append(f'pi1_{variable}')
        D_Daughters_vars.append(f'pi2_{variable}')


    # vertex fitting D, save vtx variables before the 2nd treefit
    vx.treeFit('D+:K2pi', conf_level=0.00, updateAllDaughters=True, massConstraint=[], ipConstraint=False, path=main_path)
    vm.addAlias('vtxChi2','extraInfo(chiSquared)')
    vm.addAlias('vtxNDF','extraInfo(ndf)')
    vm.addAlias('vtxReChi2','formula(vtxChi2/vtxNDF)')
    vm.addAlias('flightDistanceSig','formula(flightDistance/flightDistanceErr)')
    vm.addAlias('flightTimeSig','formula(flightTime/flightTimeErr)')
    ma.variablesToExtraInfo('D+:K2pi', variables={'flightDistanceSig':'D_A1FflightDistanceSig'},option=1, path=main_path)
    vm.addAlias('A1FflightDistanceSig_IP','extraInfo(D_A1FflightDistanceSig)')

    vertex_vars = ['vtxReChi2',]#'vtxNDF','flightDistanceSig','flightTimeSig',]

    # the mass cut is needed again because the treefit updates the daughters
    ma.applyCuts('D+:K2pi', f'vtxReChi2<13 and {DMcut2}', path=main_path)


    # ----------------
    # D* Veto
    # ----------------
    from stdPi0s import stdPi0s

    # create a pi0
    pi0_list ='eff50_May2020Fit'
    stdPi0s(listtype=pi0_list, beamBackgroundMVAWeight='MC15ri',
            fakePhotonMVAWeight='MC15ri', path=main_path)
    pi0_daughter_cut = '''daughter(0, beamBackgroundSuppression)>0.5 and daughter(1, beamBackgroundSuppression)>0.5
     and daughter(0, fakePhotonSuppression)>0.1 and daughter(1, fakePhotonSuppression)>0.1'''
    ma.cutAndCopyList('pi0:slow',f'pi0:{pi0_list}', cut=pi0_daughter_cut, path=main_path)

    # use good D± from the correct mass window
    ma.cutAndCopyList('D+:good', 'D+:K2pi', cut='1.855 <M< 1.885',path=main_path)
    ma.reconstructDecay('D*+:veto -> D+:good pi0:slow', cut='', path=main_path)
    
    # BCS and save
    vm.addAlias('massDiff_0', 'massDifference(0)')
    vm.addAlias('dis_massDiff_0','abs( massDiff_0 - 0.14065 )')
    ma.rankByLowest('D*+:veto', 'dis_massDiff_0', numBest=1, path=main_path)   
    
    veto_dict = {'massDiff_0': 'DstVeto_massDiff_0'}
    ma.variablesToEventExtraInfo('D*+:veto', veto_dict, path=main_path)
    veto_vars = []
    for key, value in veto_dict.items():
        vm.addAlias(value, f'ifNANgiveX(eventExtraInfo({value}), -1.0)')
        veto_vars.append(value)
    
    
    # --------------------------
    # Reconstruct B, vertex fit
    # --------------------------
    ma.reconstructDecay('anti-B0:Dl =norad=> D+:K2pi e-:corrected ?addbrems', cut='', path=main_path)
    vx.treeFit('anti-B0:Dl', conf_level=0, updateAllDaughters=False, massConstraint=[], ipConstraint=True, path=main_path)

    # Get the distance between vertices De/IP and D+
    # vm.addAlias('vtxDDSig', 'vertexDistanceOfDaughterSignificance(0,0)')
    # vm.addAlias('vtxIPDSig', 'vertexDistanceOfDaughterSignificance(0)') this output the same distribution as above

    # Angle bewteen 1st and 2nd daughter
    vm.addAlias('cos_angle_0_1', 'cos( daughterAngle(0,1) )')
    vm.addAlias('CMS_cos_angle_0_1', 'useCMSFrame( cos_angle_0_1 )')

    # Calculate DOCA(D,l)
    ma.calculateDistance('anti-B0:Dl', 'anti-B0:Dl -> ^D+:K2pi ^e-:corrected', "vertextrack", path=main_path)
    vm.addAlias('D_l_DisSig', 'formula( extraInfo(CalculatedDistance) / extraInfo(CalculatedDistanceError) )')

    # define the total reChi2 and p_D_l
    vm.addAlias('D_ReChi2', 'formula( vtxReChi2 + daughter(0, vtxReChi2) )')
    vm.addAlias('p_0_1', 'formula( daughter(0, CMS_p) + daughter(1, CMS_p) )')

    ma.applyCuts('anti-B0:Dl', 'vtxReChi2<14 and CMS_E<5.4', path=main_path)

    # MC Truth Matching
    ma.matchMCTruth('anti-B0:Dl', path=main_path)

    vm.addAlias('genGMPDG','genMotherPDG(1)')
    vm.addAlias('mcDaughter_0_PDG', 'mcDaughter(0,PDG)')
    vm.addAlias('mcDaughter_1_PDG', 'mcDaughter(1,PDG)')
    B_mcDaughters_vars = ['mcDaughter_0_PDG','mcDaughter_1_PDG']

    Ancestor_info = []
    B_types = ['511','521']

    for t in B_types:
        for i in range(7):  # check 7 daughters
            name = f'{t}_{i}_daughterPDG'
            var = f'varForFirstMCAncestorOfType({t}, mcDaughter({i}, PDG))'
            vm.addAlias(name, f'ifNANgiveX({var}, -1.0)')
            Ancestor_info.append(name)

    general_mc_vars = ['genGMPDG','genMotherPDG','mcErrors','mcPDG']

    # generate the DecayHash
    # main_path.add_module('ParticleMCDecayString', listName='anti-B0:Dl', fileName=output_hash)
    # vm.addAlias('DecayHash','extraInfo(DecayHash)')
    # vm.addAlias('DecayHashEx','extraInfo(DecayHashExtended)')


    # ----------
    # Build ROE
    # ----------
    ma.fillParticleList('pi+:all', '', path=main_path)
    ma.tagCurlTracks('pi+:all', mcTruth=True, selectorType='mva', path=main_path)
    vm.addAlias('isCurl', 'extraInfo(isCurl)')
    vm.addAlias('isTruthCurl', 'extraInfo(isTruthCurl)')
    vm.addAlias('truthBundleSize', 'extraInfo(truthBundleSize)')

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
                    beamBackgroundSuppression>0.05 and fakePhotonSuppression>0.1 and minC2TDist>25'
    roe_mask1 = ('my_mask',  loose_track, loose_gamma)
    ma.appendROEMasks('anti-B0:Dl', [roe_mask1], path=main_path)


    # creates V0 particle lists and uses V0 candidates to update/optimize the Rest Of Event
    ma.updateROEUsingV0Lists('anti-B0:Dl', mask_names='my_mask', default_cleanup=True, selection_cuts=None,
                             apply_mass_fit=True, fitter='treefit', path=main_path)

    ma.updateROEMask("B0:Dl","my_mask",tight_track, tight_gamma, path=main_path)


    # ----------
    # ROE vars
    # ----------
    roe_kinematics = ["roeP(my_mask)"]#, "roeE(my_mask)", "roePx(my_mask)",
                      #"roePy(my_mask)","roePz(my_mask)","roePt(my_mask)",]
    # roe_MC_kinematics = ['roeMC_E','roeMC_M','roeMC_P',
    #                      'roeMC_PTheta','roeMC_Pt',
    #                      'roeMC_Px','roeMC_Py','roeMC_Pz',]

    # Kinematic variables in CMS
    roe_cms_kinematics = vu.create_aliases(roe_kinematics, "useCMSFrame({variable})", "CMS")
    # roe_cms_MC_kinematics = vu.create_aliases(roe_MC_kinematics, "useCMSFrame({variable})", "CMS")

    roe_Mbc_Deltae = ["roeMbc(my_mask)", "roeM(my_mask)","roeDeltae(my_mask)",]

    roe_E_Q = ['roeCharge(my_mask)', 'roeNeextra(my_mask)','roeEextra(my_mask)',]

    # roe_multiplicities = ["nROE_Charged(my_mask)",'nROE_ECLClusters(my_mask)',
    #                       'nROE_NeutralECLClusters(my_mask)','nROE_KLMClusters',
    #                       'nROE_NeutralHadrons(my_mask)',"nROE_Photons(my_mask)",
    #                       'nROE_Tracks(my_mask)','nROE_Composites(my_mask)',]
    #                       #'roeMC_MissFlags(my_mask)',]

    # vm.addAlias('nROE_e','nROE_Charged(my_mask, 11)')
    # vm.addAlias('nROE_mu','nROE_Charged(my_mask, 13)')
    # vm.addAlias('nROE_K','nROE_Charged(my_mask, 321)')
    # vm.addAlias('nROE_pi','nROE_Charged(my_mask, 211)')
    # roe_nCharged = ['nROE_e','nROE_mu','nROE_K','nROE_pi']

    vm.addAlias('CMS0_weDeltae','weDeltae(my_mask,0)')
    # vm.addAlias('Lab1_weDeltae','weDeltae(my_mask,1)')
    #Option for correctedB_deltae variable should only be 0/1 (CMS/LAB)
    #Option for correctedB_mbc variable should only be 0/1/2 (CMS/LAB/CMS with factor)
    vm.addAlias('CMS0_weMbc','weMbc(my_mask,0)')
    # vm.addAlias('Lab1_weMbc','weMbc(my_mask,1)')
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


    # ------------------
    # Tag side B vertex
    # ------------------
    # vx.TagV("anti-B0:Dl", fitAlgorithm="Rave", maskName='my_mask', path=main_path)
    vx.TagV('B0:Dl',confidenceLevel=0.0,trackFindingType='standard_PXD',MCassociation='breco',constraintType='tube', 
            reqPXDHits=0, maskName='my_mask', fitAlgorithm='KFit', kFitReqReducedChi2=5.0, path=main_path)
    vm.addAlias('TagVReChi2','formula(TagVChi2/TagVNDF)')
    vm.addAlias('TagVReChi2IP','formula(TagVChi2IP/TagVNDF)')

    TVVariables = ['TagVReChi2','TagVReChi2IP',] #'DeltaZ',        'DeltaZErr',    
    #                'TagVx',     'TagVxErr',     'TagVy',         'TagVyErr',
    #                'TagVz',     'TagVzErr',     'TagVNTracks']

    ma.applyCuts('anti-B0:Dl', '4<roeMbc(my_mask) and -5<roeDeltae(my_mask)<5 and \
                  4<CMS2_weMbc and -5<CMS0_weDeltae<5 and abs(roeCharge(my_mask))<3 and \
                  0.2967<Lab5_weMissPTheta<2.7925 and 0.2967<Lab6_weMissPTheta<2.7925 and \
                  0<TagVReChi2<100 and 0<TagVReChi2IP<100', path=main_path)


    # ---------------------
    # Continuum Suppression
    # ---------------------
    ma.buildContinuumSuppression(list_name="B0:Dl", roe_mask="my_mask", path=main_path)

    vm.addAlias('KSFWV1','KSFWVariables(et)') #correlates with p_D + p_l
    vm.addAlias('KSFWV2','KSFWVariables(mm2)') #correlates with mm2
    vm.addAlias('KSFWV3','KSFWVariables(hso00)')
    vm.addAlias('KSFWV4','KSFWVariables(hso01)')
    vm.addAlias('KSFWV5','KSFWVariables(hso02)')
    vm.addAlias('KSFWV6','KSFWVariables(hso03)')
    vm.addAlias('KSFWV7','KSFWVariables(hso04)')
    vm.addAlias('KSFWV8','KSFWVariables(hso10)')
    vm.addAlias('KSFWV9','KSFWVariables(hso12)')
    vm.addAlias('KSFWV10','KSFWVariables(hso14)')
    vm.addAlias('KSFWV11','KSFWVariables(hso20)') #correlates with mm2
    vm.addAlias('KSFWV12','KSFWVariables(hso22)')
    vm.addAlias('KSFWV13','KSFWVariables(hso24)')
    vm.addAlias('KSFWV14','KSFWVariables(hoo0)')
    vm.addAlias('KSFWV15','KSFWVariables(hoo1)')
    vm.addAlias('KSFWV16','KSFWVariables(hoo2)')
    vm.addAlias('KSFWV17','KSFWVariables(hoo3)')
    vm.addAlias('KSFWV18','KSFWVariables(hoo4)')
    # vm.addAlias('CC1','CleoConeCS(1)')
    # vm.addAlias('CC2','CleoConeCS(2)')
    # vm.addAlias('CC3','CleoConeCS(3)')
    # vm.addAlias('CC4','CleoConeCS(4)')
    # vm.addAlias('CC5','CleoConeCS(5)')
    # vm.addAlias('CC6','CleoConeCS(6)')
    # vm.addAlias('CC7','CleoConeCS(7)')
    # vm.addAlias('CC8','CleoConeCS(8)')
    # vm.addAlias('CC9','CleoConeCS(9)')

    CSVariables = [
        'isContinuumEvent',  "R2",  "thrustBm",  "thrustOm",  "cosTBTO",  "cosTBz",
        "KSFWV1",        "KSFWV2",  "KSFWV3",    "KSFWV4",    "KSFWV5",   "KSFWV6",
        "KSFWV7",        "KSFWV8",  "KSFWV9",    "KSFWV10",   "KSFWV11",  "KSFWV12",
        "KSFWV13",      "KSFWV14",  "KSFWV15",   "KSFWV16",   "KSFWV17",  "KSFWV18",
    #     "CC1",              "CC2",  "CC3",       "CC4",       "CC5",      "CC6",
    #     "CC7",              "CC8",  "CC9",
    ]


    # Write variables to Ntuples
    # vm.addAlias('cos_pV','cosAngleBetweenMomentumAndVertexVector')
    # vm.addAlias('cos_pB','cosThetaBetweenParticleAndNominalB')


    # ---------------------
    # Save to ntuples
    # ---------------------
    if args.printMCDecayTree:
        ma.printMCParticles(onlyPrimaries=False, maxLevel=-1, path=main_path,
                            showProperties=False, showMomenta=False, showVertices=False, showStatus=False, 
                            suppressPrint=True)

    b_vars = vu.create_aliases_for_selected(
        list_of_variables= vc.deltae_mbc + roe_Mbc_Deltae + roe_E_Q + roe_cms_kinematics
        + CSVariables + we + vertex_vars + TVVariables + B_mcDaughters_vars
        + ['mcErrors','mcPDG','dr','D_l_DisSig','CMS_cos_angle_0_1'],
        decay_string='^anti-B0:Dl =norad=> D+:K2pi e-:corrected',
        prefix=['B0'])

    D_vars = vu.create_aliases_for_selected(
        list_of_variables= cms_kinematics + vc.kinematics + vc.dalitz_3body + vc.inv_mass 
        + D_Daughters_vars + vertex_vars + general_mc_vars + Ancestor_info
        + ['pErr','dM','BFM','A1FflightDistanceSig_IP'],
        decay_string='anti-B0:Dl =norad=> ^D+:K2pi e-:corrected',
        prefix=['D'])

    for i in range(2):
        vm.addAlias(f'd{i}_mcPDG', f'daughter({i}, mcPDG)')
    l_vars = vu.create_aliases_for_selected(
        list_of_variables= cms_kinematics + vc.kinematics + general_mc_vars + pid_vars
        + ['isBremsCorrected','dM','pValue','isCloneTrack','d0_mcPDG','d1_mcPDG','eID'],
        decay_string='anti-B0:Dl =norad=> D+:K2pi ^e-:corrected',
        prefix=['ell'])

    #nu_vars = vu.create_aliases_for_selected(
    #    list_of_variables= cms_kinematics + vc.inv_mass,
    #    decay_string='^nu_e:missing',
    #    prefix=['nu'])

    #tag_vars = vu.create_aliases_for_selected(
    #    list_of_variables= cms_kinematics + vc.kinematics + vc.inv_mass + tag_nParticle,
    #    decay_string='^B0:tagFromROE',
    #    prefix=['tag'])

    candidate_vars = ['Ecms'] + veto_vars + b_vars + D_vars + l_vars

    ma.variablesToNtuple('anti-B0:Dl', candidate_vars, useFloat=True,
                         filename=output_file, treename='B0', path=main_path, basketsize=1_000_000)

    b2.process(path=main_path)
