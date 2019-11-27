@echo off
REM Run the tracking routine my_track.exe with five input parameters:
REM parent_folder subfolder SEGMENTATION_OBJECT_SIZE SEGMENTATION_ITERATIONS SPLIT DOWNSAMPLE Z_ADJUST MODELNAME

REM Prerequisities: ---

python run_ctc.py "E:\\Fluo-N3DH-SIM+\\" "02" 28 2 0 2 6 "pv20_1573672026_SmallSIMPlus_"