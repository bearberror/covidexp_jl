import CSV.read
using DataFrames
using MLJ
using Chain
using ComputationalResources
using DataFramesMeta
MLJ.default_resource(CPUThreads())
include("./metrics.jl")
## Training and eval with CV
train = read("./data/processed/training_set.csv", DataFrame)

y, X = unpack(train, ==(:Severity_Severe))

y = coerce(y, OrderedFactor)
X = coerce(X, Count => Continuous)
## setting param-grid
XGB = @load XGBoostClassifier pkg = "XGBoost"
xgb_tuned = XGB(nthread=8, max_depth=5, eta=0.1, num_round=500)

model = @chain xgb_tuned begin
  machine(X, y)
  fit!
end

fe_im_df = @chain begin
  DataFrame(report(model)[:feature_importances])
  filter(:gain => gain -> gain >= mean(_.gain), _)
end

##
X_cut = select(X, fe_im_df.fname)

#evaluate!(model_2, resampling = CV(nfolds = 5, shuffle= true), acceleration = CPUThreads(), measure = [log_loss, accuracy])

##
max_depth = range(xgb, :max_depth, values=[5, 8, 10])
num_round = range(xgb, :num_round, values=[100, 500, 1000])
eta = range(xgb, :eta, values=[0.01, 0.1, 0.5])
xgb = XGB(nthread=8, gamma=0.8)
tune_model = TunedModel(model=xgb, resampling=CV(nfolds=3, shuffle=true),
  measure=log_loss, tuning=Grid(),
  range=[max_depth, num_round, eta], acceleration=CPUThreads(),repeats=1)

mach = machine(tune_model, X_cut, y)
result = @time fit!(mach)

rep_df = @chain begin
  DataFrame(report(result).plotting[:parameter_values], :auto) 
  rename(Dict(["x1", "x2", "x3"] .=> report(result).plotting[:parameter_names])) 
  insertcols!(4, :measurement => report(result).plotting[:measurements])
  sort!(order(:measurement))
end
