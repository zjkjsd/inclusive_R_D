#!/bin/bash
# script to test locally the steering script

# +
# The script requires 2 input arguments
if [ "$#" -ne 2 ]; then
    echo -e "usage\n\t$0 <SteeringFile> <EventType>\nAborting"
    exit 1
fi

steering=$1
EventType=$2

# Save the output Ntuples into different folders
if [ "$steering" == "2_Reconstruction_e.py" ]; then
    destination_folder='../Samples/Generic_MC15ri/e_channel/MC15ri_local_200fb'
elif [ "$steering" == "3_Reconstruction_e_wrongCharge.py" ]; then
    destination_folder='../Samples/Generic_MC15ri/e_channel/MC15ri_local_wrongCharge_200fb'
elif [ "$steering" == "4_Reconstruction_mu.py" ]; then
    destination_folder='../Samples/Generic_MC15ri/mu_channel/MC15ri_local_200fb'
fi

echo "Selected folder: $destination_folder"


# Submit jobs to the s queue
i=0
for file in /group/belle2/dataprod/MC/MC15ri/${EventType}/sub0?/*root
do
    echo -e "Submitting local job $i"
    bsub -q s "basf2 ${steering} -i ${file} -o ${destination_folder}/${EventType}/${EventType}_${i}.root"
    ((i++))
done

# -eo ${EventType}_${i}.err
