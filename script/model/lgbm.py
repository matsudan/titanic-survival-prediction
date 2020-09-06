# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import optuna.integration.lightgbm as lgb


class TunerCVCheckpointCallback(object):
    """Optuna の LightGBMTunerCV から学習済みモデルを取り出すためのコールバック"""

    def __init__(self):
        # オンメモリでモデルを記録しておく辞書
        self.cv_boosters = {}

    @staticmethod
    def params_to_hash(params):
        """パラメータを元に辞書のキーとなるハッシュを計算する"""
        params_hash = hash(frozenset(params.items()))
        return params_hash

    def get_trained_model(self, params):
        """パラメータをキーとして学習済みモデルを取り出す"""
        params_hash = self.params_to_hash(params)
        return self.cv_boosters[params_hash]

    def __call__(self, env):
        """LightGBM の各ラウンドで呼ばれるコールバック"""
        # 学習に使うパラメータをハッシュ値に変換する
        params_hash = self.params_to_hash(env.params)
        # 初登場のパラメータならモデルへの参照を保持する
        if params_hash not in self.cv_boosters:
            self.cv_boosters[params_hash] = env.model


def objectives(x_train, y_train, x_test, categorical_features):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    lgb_train = lgb.Dataset(data=x_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
    print("lgb_train: ", lgb_train)
    # lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train, categorical_feature=categorical_features)

    checkpoint_cb = TunerCVCheckpointCallback()
    callbacks = [
        checkpoint_cb,
    ]

    tuner = lgb.LightGBMTunerCV(
        params,
        lgb_train,
        verbose_eval=100,
        early_stopping_rounds=100,
        folds=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        callbacks=callbacks,
    )

    tuner.run()

    print("Best score:", tuner.best_score)
    best_params = tuner.best_params
    print("Best params:", best_params)
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

    # NOTE: 念のためハッシュの衝突に備えて Trial の数と学習済みモデルの数を比較しておく
    print("checkpoint_cb.cv_boosters: ", checkpoint_cb.cv_boosters)
    print("tuner.study.trials: ", tuner.study.trials)
    assert len(checkpoint_cb.cv_boosters) == len(tuner.study.trials) - 1

    # 最も良かったパラメータをキーにして学習済みモデルを取り出す
    cv_booster = checkpoint_cb.get_trained_model(tuner.best_params)
    print("cv_booster: ", cv_booster)
    print("cv_booster.best_iteration: ",cv_booster.best_iteration)

    y_pred_proba_list = cv_booster.predict(
        x_test,
        num_iteration=cv_booster.best_iteration
    )

    print("y_pred_proba_list: ", y_pred_proba_list)
