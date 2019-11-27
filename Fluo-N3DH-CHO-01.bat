@echo off
REM Run the tracking routine my_track.exe with five input parameters:
REM parent_folder subfolder SEGMENTATION_OBJECT_SIZE SEGMENTATION_ITERATIONS SPLIT DOWNSAMPLE Z_ADJUST MODELNAME

REM Prerequisities: ---

.\run_ctc_cytocensus\run_ctc_cytocensus.exe ..\Fluo-N3DH-CHO\ "01" 38 3 1 2 5 "pv20_1573825326_CTCCHO10_"