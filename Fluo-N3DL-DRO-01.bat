@echo off
REM Run the tracking routine my_track.exe with five input parameters:
REM parent_folder subfolder SEGMENTATION_OBJECT_SIZE SEGMENTATION_ITERATIONS SPLIT DOWNSAMPLE Z_ADJUST MODELNAME

REM Prerequisities: ---

.\run_ctc_cytocensus\run_ctc_cytocensus.exe  ..\Fluo-N3DL-DRO\ "01" 6 2 1 2 3 "pv20_1574809217_CTCDROp5_"