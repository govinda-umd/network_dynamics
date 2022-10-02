#!/bin/csh

# Script to run trial-level individual GLM analysis using AFNI's 3dDeconvovle and 3dREMLfit
#  - note: takes a few minutes per subject
# Modified on July 12, 2021 by MD from ROI-level first-level analysis code

set subj = $1
set set_name = mashid #aba

# Set default paths
set proj_path = /home/govindas/vscode-BSWIFT-mnt/MAX	#/data/bswift-1/Pessoa_Lab/MAX
set masksAndCensorsPath = $proj_path/dataset/preproc/masksAndCensors/MAX"$subj"
set mask = $masksAndCensorsPath/mask_"$set_name"_goodVoxels.nii.gz	#MAX_ROIs_final_gm_85_goodVoxels.nii.gz

# Set output path
# set out = $proj_path/dataset/first_level/ROI/neutral_runs_conditionLevel_Triallevel_FNSandFNT/MAX_ROIs_final_gm_85/MAX"$subj"
set out = /home/govindas/network_dynamics/data/max/neutral_runs_trial_level_FNSandFNT/${set_name}/MAX"$subj"
mkdir -p "$out"

# Get details of bad runs
set bad_runs = (`cat "$proj_path"/scripts/runs_to_exclude_neutral.txt | grep MAX"$subj" | awk '{ print $2 }'`)
set num_of_bad_runs = $#bad_runs

# Set input and stim_times paths
set input_path = "$proj_path"/dataset/preproc/MAX"$subj"/func_neutral
if ($num_of_bad_runs > 0) then
	set inputFile_suffix = '_reducedRuns'
	set stim_path = "$proj_path"/stim_times_neutral/MAX"$subj"
else if ($num_of_bad_runs == 0) then
	set inputFile_suffix = ''
	set stim_path = "$proj_path"/stim_times_neutral
endif
set input_path_final = "$input_path""$inputFile_suffix"

# Define input files
set input = "$input_path_final"/MAX"$subj"_EP_Main_TR_MNI_2mm_I_denoised"$inputFile_suffix".nii.gz
set fileCensMotion = "$input_path_final"/MAX"$subj"_MotionCensor_1.0_Main"$inputFile_suffix".txt #if shock-contaminated volumes are to be censored
set fileRawMotion = "$input_path_final"/MAX"$subj"_MotionPar_Main"$inputFile_suffix".txt
set fileDerMotion = "$input_path_final"/MAX"$subj"_MotionParDeriv_Main"$inputFile_suffix".txt

set fileCensMotionAndShock = "$masksAndCensorsPath"/MAX"$subj"_ShockTouchMotionCensor"$inputFile_suffix".txt
set fileConcatInfo = "$masksAndCensorsPath"/runConcatInfo"$inputFile_suffix".1D

# Get details of good runs
set good_runs = (`echo "tmp = '"$bad_runs"'.split(); bad_runs = [int(r) for r in tmp]; print(' '.join([str(i) for i in range(2,8,2) if i not in bad_runs]))" | python`)
set nruns = $#good_runs
@ nvolume = 336

echo "Processing MAX"$subj",  $nruns runs ($num_of_bad_runs bad runs excluded)"
echo "Stim path: $stim_path"
echo "Input: $input"
echo "fileCensMotion: $fileCensMotion"
echo "fileRawMotion: $fileRawMotion"
echo "fileDerMotion: $fileDerMotion"
echo "fileCensMotionAndShock: $fileCensMotionAndShock"
echo "fileConcatInfo: $fileConcatInfo"

echo "The output path is "$out""

# Extract timeseries
# rm -rf $out/MAX"$subj"_meanTS.1D
# 3dROIstats -quiet -mask $mask $input > $out/MAX"$subj"_meanTS.1D

	setenv AFNI_1D_TIME YES
	setenv AFNI_1D_TIME_TR 1.25

echo 'going out of shell'

python ./zscoring_meanTS.py MAX"$subj"

echo 'back to shell!'