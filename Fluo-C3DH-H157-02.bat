@echo off
REM Run the tracking routine my_track.exe with five input parameters:
REM parent_folder subfolder SEGMENTATION_OBJECT_SIZE SEGMENTATION_ITERATIONS SPLIT DOWNSAMPLE Z_ADJUST MODELNAME

REM Prerequisities: ---

.\run_ctc_cytocensus\run_ctc_cytocensus.exe ..\Fluo-C3DH-H157_Test\ "02" 40 6 0 4 8 "pv20_1574722630_C3DHH157_"