# DiabetesRiskPrediction
This respository is part of research work aimed at diabetes risk prediction using CGM time series.

## preproc
- fixPaths.jl fixes the paths that were generated due to downloading the directories
 one-by-one through Dropbox.

- patientToPathDict.jl creates a dictionary where ptID are the keys and the
 values are the paths where the glucose timeseries are stored.

- cleanData.jl Linear imputation on glucose time series for values outside the bounds
 39-400.

- dataParsing.jl parses data. Fixes date issues. Removes duplicates.


## statistics
- browseData.jl This can be thought of as a kind of notebook to explore the data set (e.g., compute the amount of imputed data).
- patientDayStruc Creates a structure that holds CGM per day among other things.
- sampleAv.jl averages the CGM per day samples.
- histograms.jl Creates the CGM histograms per individual.

## ML
- models.jl Contains the LSTM, GRU and RNN architectures. Contains the training functions.
- CNN.jl Contains the CNN architectures. Contains the training functions.
- parseTimeSeries.jl Prepares the data for training, i.e., generates pairs of (x,y), such that x is a vector of consecutive measurements, and y is the next measurement.
- plots.jl Plots to visualize the training process.
