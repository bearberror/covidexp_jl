import CSV.read
using DataFrames
using MLJ
using Chain
using ComputationalResources
using MLJParticleSwarmOptimization
MLJ.default_resource(CPUThreads())
## Training and eval with CV
train = read("./data/processed/training_set.csv", DataFrame)

y, X = unpack(train, ==(:Severity_Severe))

y = coerce(y, OrderedFactor)
X = coerce(X, Count => Continuous)
## setting param-grid
XGB = @load XGBoostClassifier pkg = "XGBoost"
xgb = XGB(nthread=8)

max_depth = range(xgb, :max_depth, values=[5, 8, 10])
num_round = range(xgb, :num_round, values=[100, 500, 1000])
eta = range(xgb, :eta, values=[0.01, 0.1, 0.5])
##
xgb_tune = TunedModel(model=xgb,
  resampling=CV(nfolds=5, shuffle=true),
  measure=log_loss, tuning=ParticleSwarm(n_particles = 20),
  range=[eta, max_depth, num_round],
  acceleration=CPUThreads(), repeats=1, n = 20)
mach = machine(xgb_tune, X, y)
result = @time fit!(mach)
##
report(result)
report(result).best_model

rep_df = @chain begin
  DataFrame(report(result).plotting[:parameter_values], :auto) 
  rename(Dict(["x1", "x2", "x3"] .=> report(result).plotting[:parameter_names])) 
  insertcols!(4, :measurement => report(result).plotting[:measurements])
  sort!(order(:measurement))
end


