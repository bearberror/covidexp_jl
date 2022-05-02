using CSV: read, write
using DataFrames
import MLJ.MLJBase.partition
using DataFramesMeta
import StatsBase: sample, shuffle

data = read("./data/raw/covid.csv", DataFrame)
select!(data, Not([:Severity_Mild, :Severity_Moderate, :Severity_None, :Country]))

data, eval_set = partition(data, 0.9)

data_0 = @chain data begin
  @rsubset :Severity_Severe == 0
end

data_1 = @chain data begin
  @rsubset :Severity_Severe == 1
end

data_0 = data_0[sample(1:nrow(data_0), 46257, replace=false), :]
main_data = vcat(data_1, data_0)
data_0 = Nothing
data_1 = Nothing
data = Nothing
GC.gc()
main_data = main_data[shuffle(axes(main_data, 1)), :]

write("./data/processed/training_set.csv", main_data)
write("./data/processed/eval_set.csv", eval_set)
