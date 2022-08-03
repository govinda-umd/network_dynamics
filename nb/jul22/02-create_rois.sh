#!/bin/bash

proj_dir=/home/govindas/network_dynamics
masks_dir=${proj_dir}/data/rois/mashid

anat_file=/home/govindas/parcellations/templates/MNI152_T1_2mm_brain.nii.gz

echo "${masks_dir}"

r=10 # radius == 6mm

# cat ${masks_dir}/center_coords.txt
n=1
while IFS= read -r cx cy cz
do
    # echo ${cx} ${cy} ${cz}
    
    # 3dcalc \
    #     -a ${anat_file} \
    #     -expr 'n*step(${r}*${r} - (x-${cx})*(x-${cx}) - (y-${cy})*(y-${cy}) - (z-${cz})*(z-${cz}))' \
    #     -prefix ${masks_dir}/individual_nifti_files/roi_"${n}".nii.gz
    
    n=$((n+1))
done < ${masks_dir}/center_coords.txt