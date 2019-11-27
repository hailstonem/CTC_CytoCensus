@echo off
REM Run the tracking routine my_track.exe with five input parameters:
REM parent_folder subfolder SEGMENTATION_OBJECT_SIZE SEGMENTATION_ITERATIONS SPLIT DOWNSAMPLE Z_ADJUST MODELNAME

REM Prerequisities: ---

.\run_ctc_cytocensus\run_ctc_cytocensus.exe ..\Fluo-N3DL-TRIC_Test\ "02" 8 2 1 2 3 "pv20_1573838178_TRICCTCModel2_"