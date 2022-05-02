module SelMetrics
export metrics

import MLJ: MLJBase

function metrics(pred_y, test_y, yhat)
  metrics_dict = Dict("precision" => MLJBase.Precision()(pred_y, test_y),
    "recall" => MLJBase.Recall()(pred_y, test_y),
    "accuracy" => MLJBase.Accuracy()(pred_y, test_y),
    "auc" => MLJBase.AUC()(yhat, test_y),
    "f1" => MLJBase.FScore()(pred_y, test_y))
  for (i, j) in metrics_dict
    println(i, ": ", j)
  end
end

end