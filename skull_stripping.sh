#!/bin/bash

inpath=data/ABIDE
FSLDIR=/usr/local/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH

for ID in sub-0051456_T1w

do 

	echo $ID 
	/usr/local/fsl/bin/bet2 $inpath/${ID}.nii.gz $inpath/${ID}_ss.nii.gz -f 0.5 -g 0 -m


done



