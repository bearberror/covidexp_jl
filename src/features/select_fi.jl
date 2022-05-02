using CSV: read, write
using DataFrames
using MLJ
using Chain
MLJ.default_resource(CPUThreads())
## Fit model
train = read("./data/processed/training_set.csv", DataFrame)
eval_set = read("./data/processed/eval_set.csv", DataFrame)
y, X = unpack(train, ==(:Severity_Severe))

y = coerce(y, OrderedFactor)
X = coerce(X, Count => Continuous)

XGB = @load XGBoostClassifier pkg = "XGBoost"
xgb_tuned = XGB(nthread=8, max_depth=5, eta=0.1, num_round=500)

model = @chain xgb_tuned begin
  machine(X, y)
  fit!
end
# Demonstrated feature importance
fe_im_df = @chain begin
  DataFrame(report(model)[:feature_importances])
  filter(:gain => gain -> gain >= mean(_.gain), _)
end

@chain train begin
  select(_, fe_im_df.fname)
  write("./data/processed/train_cutted.csv",_ )
end

@chain eval_set begin
  select(_, fe_im_df.fname)
  write("./data/processed/eval_cutted.csv",_ )
end
