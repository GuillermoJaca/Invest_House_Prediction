#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 07:39:31 2022

@author: guillermogarcia
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection._split import KFold
from sklearn.model_selection import cross_val_score

##### FEATURE PERMUTATION

def featImpMDA_reg(clf, X, y,categorical_feature ,n_splits=10):
    #feat importance based on OOS score reduction
    #cvGen = KFold(n_splits=n_splits)
    tscv = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(dtype='float64'), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(tscv.split(X=X)):
        x0, y0 = X.iloc[train, :], y.iloc[train]
        x1, y1 = X.iloc[test,:], y.iloc[test]
        fit = clf.fit(X=x0, y=y0,categorical_feature = categorical_feature) # the fit occures
        pred = fit.predict(x1) #prediction before shuffles
        y_pred = pd.Series(pred, index=y1.index)
        scr0.loc[i] = mean_absolute_error(y1, y_pred)
        for j in X.columns:
            X1_ = x1.copy(deep=True)
            np.random.shuffle(X1_[j].values) #shuffle one columns
            prob = fit.predict(X1_) #prediction after shuffle
            y_prob = pd.Series(prob, index=y1.index)
            scr1.loc[i,j] = mean_absolute_error(y1, y_prob)
    imp=(-1*scr1).add(scr0, axis=0)
    imp = imp/(-1*scr1)
    imp=pd.concat({'mean':imp.mean(), 'std':imp.std()*imp.shape[0]**-.5}, axis=1) #CLT
    return imp

def feature_permutation(data_train_transformed,data_train, features_evaluate, target):

  sel_no_nan = ~data_train_transformed[target].isnull()
  cat_feature_list = data_train[features_evaluate].select_dtypes("object").columns.tolist()

  clf = LGBMRegressor()
  imp = featImpMDA_reg(clf = clf,
                      X = data_train_transformed[sel_no_nan][features_evaluate],
                      y = data_train[sel_no_nan][target],
                      categorical_feature = cat_feature_list)
  
  imp.sort_values('mean', inplace=True)
  plt.figure(figsize=(10, imp.shape[0] / 5))
  imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
  plt.title('MDA results')
  plt.show()

  return imp


######### FORWARD FEATURE SELECTION

def forward_feat_selection(data_train_transformed,features_imp_target,target,model,sel,cat_feature_list):
  score_past = -999
  features_forward = []
  for enum, feat_imp_target in enumerate(features_imp_target):

    if enum == 0:
      features_forward.append(feat_imp_target)

    feat_cat = set(features_forward).intersection(set(cat_feature_list))

    score = cross_val_score(model,
                    X = data_train_transformed[sel][features_forward],
                    y = data_train_transformed[sel][target],
                    cv=5,
                    n_jobs = -1,
                    scoring = 'neg_mean_absolute_error',
                    fit_params = {'categorical_feature':feat_cat})
    
    score = score.mean()
    print('---------------')
    print('feature:',feat_imp_target)
    print('features:',features_forward)
    print('SCORE:', score)

    if (score > score_past) and (enum != 0):
      features_forward.append(feat_imp_target)
      score_past = score
    else:
      print('feature', feat_imp_target,'does not improve score')

  return features_forward


def backward_feat_selection(data_train_transformed,features_imp_target,target,model,sel,cat_feature_list):
  score_past = -999
  features_backwards = features_imp_target
  for enum, feat_imp_target in enumerate(features_imp_target):

    if enum == 0:
      features_backwards.remove(feat_imp_target)

    feat_cat = set(features_backwards).intersection(set(cat_feature_list))

    score = cross_val_score(model,
                    X = data_train_transformed[sel][features_backwards],
                    y = data_train_transformed[sel][target],
                    cv=5,
                    n_jobs = -1,
                    scoring = 'neg_mean_absolute_error',
                    fit_params = {'categorical_feature':feat_cat})
    
    score = score.mean()
    print('---------------')
    print('feature:',feat_imp_target)
    print('features:',features_backwards)
    print('SCORE:', score)

    if (score > score_past) and (enum != 0):
      features_backwards.remove(feat_imp_target)
      score_past = score
    else:
      print('feature', feat_imp_target,'does not improve score')

  return features_backwards


