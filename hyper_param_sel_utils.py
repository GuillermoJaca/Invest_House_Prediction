#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 07:56:12 2022

@author: guillermogarcia
"""
import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score


def optuna_lightbm(data_train,data_train_transformed,target,features_optuna):
  sel = ~data_train_transformed[target].isnull()
  target = target
  feat_cat = data_train[features_optuna].select_dtypes("object").columns.tolist()
  X = data_train_transformed[sel][features_optuna]
  y = data_train_transformed[sel][target]

  def objective(trial,X = X,y = y, feat_cat = feat_cat):

    param = {
            "verbosity": 1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

    model = LGBMRegressor(**param)

    score = cross_val_score(model,
              X = X,
              y = y,
              cv=5,
              n_jobs = -1,
              scoring = 'neg_mean_absolute_error',
              fit_params = {'categorical_feature':feat_cat})

    return np.mean(score)

  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=50)
  print('Number of finished trials:', len(study.trials))
  print('Best trial:', study.best_trial.params)

  return study.best_trial.params



def optuna_xgboost(data_train,data_train_transformed,target,features_optuna):
  sel = ~data_train_transformed[target].isnull()
  target = target
  feat_cat = data_train[features_optuna].select_dtypes("object").columns.tolist()
  X = data_train_transformed[sel][features_optuna]
  y = data_train_transformed[sel][target]

  def objective(trial,X = X,y = y, feat_cat = feat_cat):

    param = {
        "verbosity": 1,
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    model = XGBRegressor(**param)  

    score = cross_val_score(model,
              X = X,
              y = y,
              cv=5,
              n_jobs = -1,
              scoring = 'neg_mean_absolute_error')

    return np.mean(score)

  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=50)
  print('Number of finished trials:', len(study.trials))
  print('Best trial:', study.best_trial.params)

  return study.best_trial.params


def optuna_catboost(data_train,data_train_transformed,target,features_optuna):
  sel = ~data_train_transformed[target].isnull()
  target = target
  feat_cat = data_train[features_optuna].select_dtypes("object").columns.tolist()
  X = data_train_transformed[sel][features_optuna]
  y = data_train_transformed[sel][target]

  def objective(trial,X = X,y = y, feat_cat = feat_cat):
    '''
    param = {
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        'subsample': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.006, 0.018),
        'n_estimators':  25000,
        'max_depth': trial.suggest_categorical('max_depth', [7,10,14,16]),
        'random_state': trial.suggest_categorical('random_state', [24, 48,2020]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
    }
    '''

    param = {
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)


    model = CatBoostRegressor(**param) 

    score = cross_val_score(model,
              X = X,
              y = y,
              cv=5,
              n_jobs = -1,
              scoring = 'neg_mean_absolute_error',
              fit_params = {'cat_features':feat_cat})

    return np.mean(score)

  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=50)
  print('Number of finished trials:', len(study.trials))
  print('Best trial:', study.best_trial.params)

  return study.best_trial.params













