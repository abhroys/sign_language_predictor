from dataclasses import dataclass, field
from models.Xgboost import Xgboost





@dataclass
class XgbConfig:
    booster: str = "gbtree"
    silent: int = 0
    max_depth: int = 2
    subsample: float = 1
    colsample_bytree: float = 0.7
    reg_lambda: int = 3
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    tree_method: str = "gpu_hist"  ## change it to `hist` if gpu not available



config = XgbConfig()
model = Xgboost(config=config)
print(model)