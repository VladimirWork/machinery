from sklearn import metrics
import xgboost
import catboost
import lightgbm


class XGBInstance:
    def __init__(self, initial_params_deviation: dict):
        self.model = xgboost.XGBClassifier()
        self.initial_params_deviation = initial_params_deviation


class CBInstance:
    def __init__(self, initial_params_deviation: dict):
        self.model = catboost.CatBoostClassifier()
        self.initial_params_deviation = initial_params_deviation


class LGBMInstance:
    def __init__(self, initial_params_deviation: dict):
        self.model = lightgbm.LGBMClassifier()
        self.initial_params_deviation = initial_params_deviation


if __name__ == '__main__':
    pass
