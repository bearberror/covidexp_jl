import CSV.read
using DataFrames
using MLJ
using Chain
include("./metrics.jl")
MLJ.default_resource(CPUThreads())
## Training and eval with CV
train = read("./data/processed/training_set.csv", DataFrame)

y, X = unpack(train, ==(:Severity_Severe))

y = coerce(y, OrderedFactor)
X = coerce(X, Count => Continuous)

XGB = @load XGBoostClassifier pkg = "XGBoost"
xgb = XGB()

model = @chain xgb begin
  machine(X, y)
  fit!
end
##
@time evaluate!(model, resampling = CV(shuffle = true, nfolds = 3), measure = [log_loss])

## Eval with eval Set
eval_set = read("./data/processed/eval_set.csv", DataFrame)
eval_y, eval_X = unpack(eval_set, ==(:Severity_Severe))
eval_y = coerce(eval_y, OrderedFactor)
eval_X = coerce(eval_X, Count => Continuous)

pred_y = predict_mode(model, eval_X)
yhat = predict(model, eval_X)

SelMetrics.metrics(pred_y, eval_y, yhat)
