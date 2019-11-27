# CTC_CytoCensus
 Benchmark CytoCensus+MorphACME on Cell Segmentation Challenge

#To run on existing datasets

Use the supplied run_ctc.spec file with pyinstaller to create executable (directory mode), then run the corresponding batch file.

#To run on a new dataset

First train a model using CytoCensus. Copy the trained model (found at ~/.densitycount) to the models folder. Use the supplied run_ctc.spec file with pyinstaller to create executable (directory mode). Create a new batch file, setting appropriate object size parameters and Z-spacing then run the corresponding batch file.