from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost
import catboost
import lightgbm
import numpy as np
import pandas as pd


class GeneralClassifier:
    def __init__(self, model, initial_params_deviation: dict):
        self.model = model
        self.initial_params_deviation = initial_params_deviation
        self.grid_search = GridSearchCV(self.model,
                                        param_grid=self.initial_params_deviation,
                                        cv=3,
                                        verbose=10,
                                        scoring='roc_auc',
                                        n_jobs=-1)

    def __call__(self, x_train, y_train):
        self.grid_search.fit(x_train, y_train)
        self.best_params = self.grid_search.best_params_
        print('Best params: {}'.format(self.best_params))
        self.model = xgboost.XGBClassifier(**self.best_params)
        self.model.fit(x_train, y_train)
        return self.model


class GeneralRegressor:
    pass


def auc(model, x_train, x_test, y_train, y_test):
    if isinstance(model, lightgbm.LGBMClassifier):
        return (metrics.roc_auc_score(y_train, model.predict(x_train)),
                metrics.roc_auc_score(y_test, model.predict(x_test)))
    else:
        return (metrics.roc_auc_score(y_train, model.predict_proba(x_train)[:, 1]),
                metrics.roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))


if __name__ == '__main__':
    data = pd.read_csv('C:\\Users\\admin\\Downloads\\flights.csv')
    data = data.sample(frac=0.1, random_state=10)
    data = data[["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT", "AIR_TIME", "DEPARTURE_TIME", "DISTANCE", "ARRIVAL_DELAY"]]
    data.dropna(inplace=True)
    data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"] > 10) * 1
    cols = ["AIRLINE", "FLIGHT_NUMBER", "DESTINATION_AIRPORT", "ORIGIN_AIRPORT"]

    for item in cols:
        data[item] = data[item].astype("category").cat.codes + 1

    x_train, x_test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1),
                                                        data["ARRIVAL_DELAY"],
                                                        random_state=10,
                                                        test_size=0.25)

    # xgb_instance = GeneralClassifier(xgboost.XGBClassifier(),
    #                                  {'max_depth': [5, 10, 15],
    #                                   'min_child_weight': [3, 6, 9],
    #                                   'n_estimators': [200],
    #                                   'learning_rate': [0.05, 0.1, 0.2]})(x_train, y_train)
    # cb_instance = GeneralClassifier(catboost.CatBoostClassifier(),
    #                                 {'depth': [5, 10, 15],
    #                                  'learning_rate': [0.05, 0.1, 0.2],
    #                                  'l2_leaf_reg': [3, 6, 9],
    #                                  'iterations': [200]})(x_train, y_train)
    lgbm_instance = GeneralClassifier(lightgbm.LGBMClassifier(),
                                      {'max_depth': [25, 50, 75],
                                       'learning_rate': [0.01, 0.05, 0.1],
                                       'num_leaves': [300, 900, 1200],
                                       'n_estimators': [200]})(x_train, y_train)

    # print('XGB AUC: {}'.format(auc(xgb_instance, x_train, x_test, y_train, y_test)))
    # print('CB AUC: {}'.format(auc(cb_instance, x_train, x_test, y_train, y_test)))
    print('LGBM AUC: {}'.format(auc(lgbm_instance, x_train, x_test, y_train, y_test)))
