# DiabetesRiskPrediction

## src
Contains everything related to deep learning training using the old dataset.
Needs to be updated.

## preproc
- fixpath.jl fixes the paths that were generated due to downloading the directories
 one-by-one through dropbox.

- patientToPathDict.jl creates a dictionary where ptID are the keys and the
 values are the paths where the glucose timeseries are stored.

- cleanData.jl Linear imputation on glucose time series for values outside the bounds
 38-400.

- dataParsing.jl parses data. Fixes date issues. Removes duplicates.


## statistics
- browseData.jl Bunch of plots.
- patientDayStruc Creates a structure that holds CGM per day among other things.
- sampleAv.jl averages the CGM per day samples.

## ML
- parseTimeSeries.jl
- model.jl
- plotsML.jl
