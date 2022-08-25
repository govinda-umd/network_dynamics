#!/bin/csh

set subj = $1
set smoothedDataFlag = $2 # 0, or 1


# Set default paths
set proj_path = /home/govindas/vscode-BSWIFT-mnt/MAX #/data/bswift-1/Pessoa_Lab/MAX
#set local_proj_path = /home/murty/MAX
set t_mean_SD_path = "$proj_path"/dataset/preproc/temporal_mean_SD/MAX"$subj"
set t_mean_SD_SmoothedData_path = "$proj_path"/dataset/preproc/temporal_mean_SD_smoothedData/MAX"$subj"

if ($smoothedDataFlag == 0) then
	set out = /home/govindas/network_dynamics/data/max/masksAndCensors/MAX"$subj"	
	# set out = $proj_path/dataset/preproc/masksAndCensors/MAX"$subj"
else if ($smoothedDataFlag == 1) then
	set out = /home/govindas/network_dynamics/data/max/masksAndCensors_smoothedData/MAX"$subj"	
	# set out = $proj_path/dataset/preproc/masksAndCensors_smoothedData/MAX"$subj"
endif

mkdir -p "$out"

# Set mask names and paths
set mask_name = $3	#MAX_ROIs_final_gm_85, or mashid, or brenton, etc
set mask_85 = /home/govindas/network_dynamics/data/rois/$mask_name/final_mask.nii.gz	#$proj_path/ROI_masks/"$mask_name".nii.gz
set mask = "$out"/"$mask_name"_goodVoxels.nii.gz

# set multiMask = $MNI_gm
# set multiMask_name = (`basename $multiMask .nii.gz`)
# set multiMask_goodVoxels = "$out"/"$multiMask_name"_goodVoxels.nii.gz

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
#set input = "$input_path_final"/MAX"$subj"_EP_Main_TR_MNI_2mm_I_denoised"$inputFile_suffix".nii.gz
set fileCensMotion = "$input_path_final"/MAX"$subj"_MotionCensor_1.0_Main"$inputFile_suffix".txt #if shock-contaminated volumes are to be censored
set fileRawMotion = "$input_path_final"/MAX"$subj"_MotionPar_Main"$inputFile_suffix".txt
set fileDerMotion = "$input_path_final"/MAX"$subj"_MotionParDeriv_Main"$inputFile_suffix".txt

# The following are stored in local path due to permission issues
set fileCensMotionAndShock = "$out"/MAX"$subj"_ShockTouchMotionCensor"$inputFile_suffix".txt
set fileConcatInfo = "$out"/runConcatInfo"$inputFile_suffix".1D

# Get details of good runs
set good_runs = (`echo "tmp = '"$bad_runs"'.split(); bad_runs = [int(r) for r in tmp]; print(' '.join([str(i) for i in range(2,8,2) if i not in bad_runs]))" | python`)
set nruns = $#good_runs
@ nvolume = 336

echo "Processing MAX"$subj",  $nruns runs ($num_of_bad_runs bad runs excluded)"
echo "Stim path: $stim_path"
#echo "Input: $input"
echo "fileCensMotion: $fileCensMotion"
echo "fileRawMotion: $fileRawMotion"
echo "fileDerMotion: $fileDerMotion"
echo "fileCensMotionAndShock: $fileCensMotionAndShock"
echo "fileConcatInfo: $fileConcatInfo"

#######################
# Combining Shock, Touch and motion censor files
rm -rf $fileCensMotionAndShock
1deval -overwrite -a $stim_path/ShockTouchCensor15.txt -b $fileCensMotion -expr 'a*b' > $fileCensMotionAndShock

echo '\n' "Creating "$fileConcatInfo" "
set concatList = ""
set run_len = ""
set i = 0
set idx = 0

while ( $i < $nruns )
  @ idx = $i * $nvolume
  set concatList = "$concatList"" ""$idx"
  set run_len = "$run_len"" ""420"
  @ i += 1
end
rm -f "$fileConcatInfo"
echo $concatList > "$fileConcatInfo"
echo "concat list: "$concatList"; Run lengths: "$run_len" "

echo "The output path is "$out""
echo ""

#######################
# Extract good voxels
foreach runNum ($good_runs)
	
	if ($smoothedDataFlag == 0) then
		set fileName = "$t_mean_SD_path"/MAX"$subj"_tSNR_run"$runNum"_Main_TR_MNI_2mm_I_denoised.nii.gz
	else if ($smoothedDataFlag == 1) then
		set fileName = "$t_mean_SD_SmoothedData_path"/MAX"$subj"_tSNR_run"$runNum"_Main_TR_MNI_2mm_SI_denoised.nii.gz
	endif 
	echo "tSNR filename: "$fileName" "
	
	3dcalc -overwrite \
		-a $fileName'[0]' \
		-b $fileName'[1]' \
		-expr 'not(or(astep(a-100,5),step(b-25)))' \
		-prefix "$out"/goodVoxelsMask_run"$runNum".nii.gz
end

if ($nruns == 1) then
	cp "$out"/goodVoxelsMask_run"$runNum".nii.gz "$out"/goodVoxelsMask.nii.gz
else if ($nruns == 2) then
	3dcalc -overwrite \
		-a "$out"/goodVoxelsMask_run"$good_runs[1]".nii.gz \
		-b "$out"/goodVoxelsMask_run"$good_runs[2]".nii.gz \
		-expr 'a*b' \
		-prefix "$out"/goodVoxelsMask.nii.gz
else if ($nruns == 3) then
	3dcalc -overwrite \
		-a "$out"/goodVoxelsMask_run"$good_runs[1]".nii.gz \
		-b "$out"/goodVoxelsMask_run"$good_runs[2]".nii.gz \
		-c "$out"/goodVoxelsMask_run"$good_runs[3]".nii.gz \
		-expr 'a*b*c' \
		-prefix "$out"/goodVoxelsMask.nii.gz
endif

#######################
# Extract common voxels for SI and I data
foreach runNum ($good_runs)
	3dcalc -overwrite \
		-a "$t_mean_SD_path"/MAX"$subj"_tSNR_run"$runNum"_Main_TR_MNI_2mm_I_denoised.nii.gz'[0]' \
		-b "$t_mean_SD_path"/MAX"$subj"_tSNR_run"$runNum"_Main_TR_MNI_2mm_I_denoised.nii.gz'[1]' \
		-c "$t_mean_SD_SmoothedData_path"/MAX"$subj"_tSNR_run"$runNum"_Main_TR_MNI_2mm_SI_denoised.nii.gz'[0]' \
		-d "$t_mean_SD_SmoothedData_path"/MAX"$subj"_tSNR_run"$runNum"_Main_TR_MNI_2mm_SI_denoised.nii.gz'[1]' \
		-expr 'and(notzero(a),notzero(b),notzero(c),notzero(d))' \
		-prefix "$out"/commonVoxelsMask_run"$runNum".nii.gz
end

if ($nruns == 1) then
	cp "$out"/commonVoxelsMask_run"$runNum".nii.gz "$out"/commonVoxelsMask.nii.gz
else if ($nruns == 2) then
	3dcalc -overwrite \
		-a "$out"/commonVoxelsMask_run"$good_runs[1]".nii.gz \
		-b "$out"/commonVoxelsMask_run"$good_runs[2]".nii.gz \
		-expr 'a*b' \
		-prefix "$out"/commonVoxelsMask.nii.gz
else if ($nruns == 3) then
	3dcalc -overwrite \
		-a "$out"/commonVoxelsMask_run"$good_runs[1]".nii.gz \
		-b "$out"/commonVoxelsMask_run"$good_runs[2]".nii.gz \
		-c "$out"/commonVoxelsMask_run"$good_runs[3]".nii.gz \
		-expr 'a*b*c' \
		-prefix "$out"/commonVoxelsMask.nii.gz
endif

foreach runNum ($good_runs)
	 rm -rf "$out"/commonVoxelsMask_run"$runNum".nii.gz
end

#######################
# Get good voxels for specified ROIs only
3dcalc -overwrite -a "$out"/goodVoxelsMask.nii.gz -b "$out"/commonVoxelsMask.nii.gz -c "$mask_85" -expr 'a*b*c' -prefix "$mask"

# Get good voxels for grey matter mask
# 3dcalc -overwrite -a "$out"/goodVoxelsMask.nii.gz -b "$out"/commonVoxelsMask.nii.gz -c "$multiMask" -expr 'a*b*c' -prefix "$multiMask_goodVoxels"
echo "Created good voxel masks"

