# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedKFold
import optuna.integration.lightgbm as oplgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score


def tuning(x_train, y_train, categorical_features):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    lgb_train = oplgb.Dataset(data=x_train, label=y_train, categorical_feature=categorical_features,
                              free_raw_data=False)
    print("lgb_train: ", lgb_train)

    tuner = oplgb.LightGBMTunerCV(
        params,
        lgb_train,
        verbose_eval=100,
        early_stopping_rounds=100,
        folds=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    )

    tuner.run()

    print("Best score:", tuner.best_score)
    best_params = tuner.best_params
    print("Best params:", best_params)
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

    return tuner.best_params


def predict(best_params, x_train, y_train, x_test, categorical_features, sub):
    y_preds = []
    models = []
    oof_train = np.zeros((len(x_train),))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):
        X_tr = x_train.loc[train_index, :]
        X_val = x_train.loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)

        model = lgb.train(
            best_params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            verbose_eval=10,
            num_boost_round=1000,
            early_stopping_rounds=10
        )

        oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = model.predict(x_test, num_iteration=model.best_iteration)

        y_preds.append(y_pred)
        models.append(model)

    scores = [
        m.best_score["valid_1"]["binary_logloss"] for m in models
    ]
    score = sum(scores) / len(scores)
    print("===CV scores===")
    print("scores: ", scores)
    print("score: ", score)

    y_pred_oof = (oof_train > 0.5).astype(int)
    print("accuracy_score: ", accuracy_score(y_train, y_pred_oof))

    y_sub = sum(y_preds) / len(y_preds)
    y_sub = (y_sub > 0.5).astype(int)

    sub["Survived"] = y_sub
    sub.to_csv("../output/submission_lightgbm_skfold.csv", index=False)
