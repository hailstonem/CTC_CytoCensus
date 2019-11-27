@echo off
REM Run the tracking routine my_track.exe with five input parameters:
REM parent_folder subfolder SEGMENTATION_OBJECT_SIZE SEGMENTATION_ITERATIONS SPLIT DOWNSAMPLE Z_ADJUST MODELNAME

REM Prerequisities: ---

.\run_ctc_cytocensus\run_ctc_cytocensus.exe ..\Fluo-N3DH-CE\ "02" 25 6 1 2 10 "pv20_1573748234_CelegansCTC4_"