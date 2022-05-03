import CSV.read
using DataFrames
using MLJ
using Chain
using EvoTrees
include("./metrics.jl")
MLJ.default_resource(CPUThreads())
## Training and eval with CV
train = read("./data/processed/training_set.csv", DataFrame)

y, X = unpack(train, ==(:Severity_Severe))

y = coerce(y, OrderedFactor)
X = coerce(X, Count => Continuous)

XGB = @load XGBoostClassifier pkg = "XGBoost"
xgb = XGB()
@load EvoTreeClassifier
evo = EvoTreeClassifier(γ = 5, λ = 10, nrounds = 1000, max_depth = 1, rowsample = 1, colsample = 0)

model = @chain xgb begin
  machine(X, y)
  fit!
end

model_evo = @chain evo begin
  machine(X, y)
  fit!
end
##
@time evaluate!(model, resampling = CV(shuffle = true, nfolds = 3), measure = [log_loss, recall, precision], operation = [predict, predict_mode, predict_mode])
@time evaluate!(model_evo, resampling = CV(shuffle = true, nfolds = 3), measure =log_loss)

## Eval with eval Set
eval_set = read("./data/processed/eval_set.csv", DataFrame)
eval_y, eval_X = unpack(eval_set, ==(:Severity_Severe))
eval_y = coerce(eval_y, OrderedFactor)
eval_X = coerce(eval_X, Count => Continuous)

pred_y = predict_mode(model, eval_X)
yhat = predict(model, eval_X)

SelMetrics.metrics(pred_y, eval_y, yhat)
