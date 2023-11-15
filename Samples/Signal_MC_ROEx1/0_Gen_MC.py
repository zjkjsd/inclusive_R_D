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
import b2test_utils as b2tu
import generators as ge
import simulation as si
import reconstruction as re
import mdst
import glob

# set database conditions (in addition to default) release-05-02-00
b2.conditions.prepend_globaltag("mc_production_MC15ri_a")

# background (collision) files
bg = glob.glob('/group/belle2/dataprod/BGOverlay/early_phase3/release-06-00-05/overlay/BGx1/set0/*.root')

# create path
main = b2.create_path()

# specify number of events to be generated
main.add_module("EventInfoSetter", expList=1003, runList=0, evtNumList=50000)

# the command line code is: 
# bsub -q l 'basf2 /current/directory/0_Gen_MC.py  D_tau/Dst_l -n 10000'
# Add the generator
import sys
decaymode = sys.argv[1]
decfile=f'./B2{decaymode}_nu/decfiles/{sys.argv[2]}.dec'
output =f'./B2{decaymode}_nu/MC/MC_{sys.argv[2]}_test.root'

# generate events from decfile
ge.add_evtgen_generator(path=main, finalstate='signal', signaldecfile=decfile)
            

# detector simulation
si.add_simulation(main, bkgfiles=bg)

# reconstruction
re.add_reconstruction(main)

# Finally add mdst output
mdst.add_mdst_output(main, filename=output)

# process events and print call statistics
# b2.process(main)
b2tu.safe_process(main)
print(b2.statistics)
