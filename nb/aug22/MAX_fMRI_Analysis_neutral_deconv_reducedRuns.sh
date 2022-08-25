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
rm -rf $out/MAX"$subj"_meanTS.1D
3dROIstats -quiet -mask $mask $input > $out/MAX"$subj"_meanTS.1D

	setenv AFNI_1D_TIME YES
	setenv AFNI_1D_TIME_TR 1.25
	
# Run 3dDeconvolve and 3dREMLfit
echo '\n' "Subject MAX"$subj" ... running multiple regression using 3dDeconvolve:"
3dDeconvolve -overwrite \
	-input $out/MAX"$subj"_meanTS.1D\' \
	-polort A \
	-local_times \
	-concat "$fileConcatInfo" \
	-noFDR \
	-num_stimts 8 \
	-censor "$fileCensMotionAndShock" \
	-stim_times_IM 1 "$stim_path"/FNS.txt 'CSPLIN(0,16.25,14)' -stim_label 1 FNS \
	-stim_times_IM 2 "$stim_path"/FNT.txt 'CSPLIN(0,16.25,14)' -stim_label 2 FNT \
	-stim_times 3 "$stim_path"/RNS.txt 'GAM(8.6,0.547,16.25)' -stim_label 3 RNS \
	-stim_times 4 "$stim_path"/RNT.txt 'GAM(8.6,0.547,16.25)' -stim_label 4 RNT \
	-stim_times 5 "$stim_path"/rate_RNS.txt 'GAM(8.6,0.547,2.0)' -stim_label 5 r_RNS \
	-stim_times 6 "$stim_path"/rate_RNT.txt 'GAM(8.6,0.547,2.0)' -stim_label 6 r_RNT \
	-stim_times_IM 7 "$stim_path"/rate_FNS.txt 'GAM(8.6,0.547,2.0)' -stim_label 7 r_FNS \
	-stim_times_IM 8 "$stim_path"/rate_FNT.txt 'GAM(8.6,0.547,2.0)' -stim_label 8 r_FNT \
	-ortvec "$fileRawMotion"'[1..6]' 'MotionParam' \
        -ortvec "$fileDerMotion"'[0..5]' 'MotionParamDerv' \
	-x1D "$out"/"MAX"$subj"_Main_block_Deconv.x1D" \
	-x1D_uncensored "$out"/"MAX"$subj"_Main_block_Deconv_uncensored.x1D" \
	-x1D_stop
	
	1dcat "$out"/"MAX"$subj"_Main_block_Deconv.x1D" > "$out"/"MAX"$subj"_Main_block_Deconv_clean.x1D"

echo "***** Running 3dREMLfit *****"
3dREMLfit -overwrite -matrix "$out"/"MAX"$subj"_Main_block_Deconv.x1D" \
	-input $out/MAX"$subj"_meanTS.1D\' \
	-noFDR \
	-GOFORIT \
	-tout \
	-Rbeta $out/"MAX"$subj"_Main_block_Deconv_betas.1D" \
	-Rbuck "$out"/"MAX"$subj"_Main_block_Deconv_bucket.1D" \
	-Rvar "$out"/"MAX"$subj"_Main_block_Deconv_REMLvar.1D" \
	-Rerrts "$out"/"MAX"$subj"_Main_block_Deconv_REML_errs.1D" \
	-Rwherr "$out"/"MAX"$subj"_Main_block_Deconv_REML_wherrs.1D" \

rm -rf "$out"/"MAX"$subj"_Main_block_Deconv_bucket_clean.1D"
1dcat "$out"/"MAX"$subj"_Main_block_Deconv_bucket.1D" > "$out"/"MAX"$subj"_Main_block_Deconv_bucket_clean.1D"