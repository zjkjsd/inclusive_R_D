#!/usr/bin/env python3
#####################################################################
# The Belle II code is designed to load the proper geometry of the detector depending on which experiment and run combination is used. Experiment 1003 is reserved as a place holder for run-independent (run=0) MC for early Phase 3 (partial PXD installed). Experiment 0 and run 0 are reserved for run-independent MC for Phase 3 (full PXD installed). (https://confluence.desy.de/display/BI/Experiment+numbering)
#
# We will also start by producing 10000 events. You can change this later.
#
# Note that the global tag used in the script will be the default one for the release, which is appropriate for run-independent MC. If you produce run-dependent MC, you will need to set the appropriate global tag (https://confluence.desy.de/display/BI/Global+Tag+%28GT%29+page).
#
# ####################################################################

import basf2 as b2
mypath=b2.Path()

# Load the EventInfoSetter module and set the exp/run/evt details
# expList=1003 for early phase 3, 0 for full Belle2 geometry
mypath.add_module("EventInfoSetter", expList=0, runList=0, evtNumList=10)

# the command line code is: 
# bsub -q l 'basf2 /current/directory/0_Gen_MC.py  D_tau/Dst_l  e/mu  -n 10000'
# Add the generator
import sys
decaymode = sys.argv[1]
light_lepton = sys.argv[2]
decfile=f'/home/belle/zhangboy/R_D/Signal_MC_ROEx1/B2{decaymode}_nu/decfiles/B2{decaymode}_nu_{light_lepton}.dec'
output =f'/home/belle/zhangboy/R_D/Signal_MC_ROEx1/B2{decaymode}_nu/MC/MC_{light_lepton}.root'
import generators as ge
ge.add_evtgen_generator(path=mypath, finalstate='signal', signaldecfile=decfile)

# Simulate the detector response and the L1 trigger
import simulation as si
import background
si.add_simulation(path=mypath, 
                  bkgfiles=background.get_background_files())

# Simulate the L1 trigger
#import L1trigger as l1
#l1.add_tsim(path=mypath)

# Reconstruct the objects
import reconstruction as re
re.add_reconstruction(path=mypath)

# Create the mDST output file
import mdst
mdst.add_mdst_output(path=mypath, filename=output)

# Process the steering path
b2.process(path=mypath)

# Finally, print out some statistics about the modules execution
print(b2.statistics)

""
import basf2 as b2
import generators as ge
import simulation as si
import L1trigger as l1
import reconstruction as re
import mdst as mdst
import glob as glob

# set database conditions (in addition to default)
b2.conditions.append_globaltag('mc_production_MC15rd_a_exp20_bucket26')
b2.conditions.append_globaltag('data_reprocessing_prompt')
b2.conditions.append_globaltag('AIRFLOW_online_snapshot_20211110-092214')


# background (collision) files
bg = glob.glob('./*.root')
#if running locally
bg_local = glob.glob("./sub00/*.root")


# create path
main = b2.create_path()

# specify number of events to be generated
main.add_module("EventInfoSetter", expList=0, runList=0, evtNumList=10000)

# events generator

# decay file
decfile = b2.find_file('decfiles/dec/1163340000.dec')

# generate events from decfile
ge.add_evtgen_generator(path=main, finalstate='signal', signaldecfile=decfile)
            

# detector simulation
si.add_simulation(main, bkgfiles=bg)

# reconstruction
re.add_reconstruction(main)

# Finally add mdst output
mdst.add_mdst_output(main, filename="mdst.root")

# process events and print call statistics
b2.process(main)
print(b2.statistics)
