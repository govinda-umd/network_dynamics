#!/bin/bash

#subs=( 202 203 209 214 222 )
for subj in {101..224} #"${subs[@]}" #
do
	if [[ $subj -eq 121  || $subj -eq 126 || $subj -eq 136 || $subj -eq 137 || $subj -eq 140 || $subj -eq 148 || $subj -eq 153 || $subj -eq 159 || $subj -eq 168 || $subj -eq 174 || $subj -eq 180 || $subj -eq 193 || $subj -eq 197 || $subj -eq 206 || $subj -eq 208 ]]; then
		continue
	fi

	./MAX_fMRI_Analysis_neutral_deconv_reducedRuns.sh $subj |& tee ./errorLogs/MAX"$subj".txt
done
