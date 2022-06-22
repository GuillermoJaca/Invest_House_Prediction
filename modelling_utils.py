#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 07:27:30 2022

@author: guillermogarcia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMRegressor
from pathlib import Path
import json

MODEL_FOLDER = "models"
MODEL_CONFIGS_FOLDER = "model_configs"

target_1 = 'baseRentSquareMeter' 
target_2 = 'price_costs_square_meter'
target_error = 'targetError'

######## Modell pre-processing

def transform_categorical_features_to_int(data_train,features_0):
  cat_feature_list = data_train[features_0].select_dtypes("object").columns.tolist()

  data_train_transformed = data_train.copy()

  back_mapping = {} #for test_set
  for cat_feat in cat_feature_list:
    data_train_transformed[cat_feat], mapping = data_train_transformed[cat_feat].factorize()
    back_mapping[cat_feat] = {i:enum for enum,i in enumerate(list(mapping))} 

  return data_train_transformed, back_mapping


def modelling(data_train,data_train_transformed,model_1,model_2,features_1,features_2,cat_features):
  if cat_features:
    cat_feature_list_1 = data_train[features_1].select_dtypes("object").columns.tolist()
    cat_feature_list_2 = data_train[features_2].select_dtypes("object").columns.tolist()
  else:
    cat_feature_list_1 = []
    cat_feature_list_2 = []
  
  params_1 = {'categorical_feature':cat_feature_list_1}
  params_2 = {'categorical_feature':cat_feature_list_2}

  # MODEL FOR BASERENT
  data_train.loc[:,'prediction_model_1'] = cross_val_predict(model_1,
                                                            X = data_train_transformed[features_1],
                                                            y = data_train_transformed[target_1],
                                                            cv=5,
                                                            n_jobs = -1,
                                                            fit_params = params_1)
  
  data_train['error_1'] = data_train[target_1] - data_train['prediction_model_1']


  #MODEL FOR PRICECOSTS
  sel_no_nan = ~data_train_transformed[target_2].isnull()
  data_train.loc[sel_no_nan,'prediction_model_2'] = cross_val_predict(model_2,
                                                            X = data_train_transformed[sel_no_nan][features_2],
                                                            y = data_train_transformed[sel_no_nan][target_2],
                                                            cv=5,
                                                            n_jobs = -1,
                                                            fit_params = params_2)

  data_train['error_2'] = data_train[target_2] - data_train['prediction_model_2'] 
  data_train['total_error'] = data_train['error_1'].abs() + data_train['error_2'].abs()

  return data_train


def visualize_samples_error(data,error_column, absolute_error_threshold):

  sel = data[error_column] > absolute_error_threshold
  data[sel][error_column].hist(bins = 30)
  plt.title('Positive Error')
  plt.show()

  sel = data[error_column] < - absolute_error_threshold
  data[sel][error_column].hist(bins = 30)
  plt.title('Negative Error')
  plt.show()

  sel = data[error_column].abs() > absolute_error_threshold
  data[sel][error_column].hist(bins = 30)
  plt.show()

  return data[sel]


def visualize_errors_features(data,error_column,features_visualize):

  for feat in features_visualize:
    if feat == 'geo_plz': continue
    print('--------------',feat,'-------------')
    if ((data[feat].dtypes == np.float64) or ('size' in feat )):

      categories, edges = pd.qcut(data[feat], 10, retbins = True, labels = False, duplicates = 'drop')
      feat = f'{feat}_bin'
      print('edges',edges)
      print('categories',categories)
      data[feat] = edges[categories]

      df_aux = pd.DataFrame(data.groupby(feat)[error_column].mean())
      df_aux['size'] = data.groupby(feat)[error_column].size()
      #df_aux = df_aux.sort_values(error_column,ascending = False)

      fig, ax = plt.subplots()
      df_aux.plot(y = error_column,ax = ax,figsize = (10,5), ds = 'steps-post', grid = True)
      df_aux.plot(y = 'size',ax = ax,secondary_y = True,figsize = (10,5), ds = 'steps-post', grid = True)
      plt.show()

      print(df_aux.iloc[:10])
      print(df_aux.iloc[-10:])

    else:

      df_aux = pd.DataFrame(data.groupby(feat)[error_column].mean())
      df_aux['size'] = data.groupby(feat)[error_column].size()
      df_aux = df_aux.sort_values(error_column,ascending = False)

      fig, ax = plt.subplots()
      df_aux.plot(y = error_column,ax = ax,figsize = (10,5), ds = 'steps-post', grid = True)
      df_aux.plot(y = 'size',ax = ax,secondary_y = True,figsize = (10,5), ds = 'steps-post', grid = True)
      plt.show()

      print(df_aux.iloc[:10])
      print(df_aux.iloc[-10:])

      
  return 


def features_mod_pipeline(data_train,data_train_transformed,features_1,features_2,threshold,plot_error_features = True,cat_features = True):

  model_1 = LGBMRegressor()
  model_2 = LGBMRegressor()

  data_modelled = modelling(data_train = data_train,
                          data_train_transformed = data_train_transformed,
                          model_1 = model_1,
                          model_2 = model_2,
                          features_1 = features_1,
                          features_2 = features_2,
                          cat_features = cat_features)

  print('TOTAL ERROR:',data_modelled['total_error'].abs().mean())
  print('ERROR_1:',data_modelled['error_1'].abs().mean())
  print('ERROR_2:',data_modelled['error_2'].abs().mean())
    

  if plot_error_features:
      features_plot = list(set(features_1).union(set(features_2)))
      samples_error = visualize_samples_error(data = data_modelled,
                        error_column = 'total_error',
                        absolute_error_threshold = threshold)
    
      visualize_errors_features(data = data_modelled,
                          error_column = 'total_error',
                          features_visualize = features_plot)

      return data_modelled, samples_error

  return data_modelled


def modelling_error(data_train,data_train_transformed,model,features):

  cat_feature_list = data_train[features].select_dtypes("object").columns.tolist()
  
  params = {'categorical_feature':cat_feature_list}

  # MODEL FOR BASERENT
  sel_no_nan = ~data_train_transformed[target_error].isnull()
  data_train.loc[sel_no_nan,'prediction_error'] = cross_val_predict(model,
                                            X = data_train_transformed[sel_no_nan][features],
                                            y = data_train_transformed[sel_no_nan][target_error],
                                            cv=5,
                                            n_jobs = -1,
                                            fit_params = params)
  
  data_train['Error_of_errors'] = data_train[target_error] - data_train['prediction_error']

  return data_train

def features_error_mod_pipeline(data_train,data_train_transformed,features,threshold,plot_error_features = True,params = False):

  if params == False:
    model = LGBMRegressor()
  else:
    model = LGBMRegressor(**params)

  data_modelled = modelling_error(data_train = data_train,
                          data_train_transformed = data_train_transformed,
                          model = model,
                          features = features)

  print('TOTAL ERROR:',data_modelled['Error_of_errors'].abs().mean())

  if plot_error_features:
      features_plot = features
      samples_error = visualize_samples_error(data = data_modelled,
                        error_column = 'Error_of_errors',
                        absolute_error_threshold = threshold)
    
      visualize_errors_features(data = data_modelled,
                          error_column = 'Error_of_errors',
                          features_visualize = features_plot)

      return data_modelled, samples_error

  return data_modelled


def save_model(model, name):
    try:
        Path(MODEL_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    pd.to_pickle(model, f"{MODEL_FOLDER}/{name}.pkl")

def save_model_config(model_config, model_name):
    try:
        Path(MODEL_CONFIGS_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    with open(f"{MODEL_CONFIGS_FOLDER}/{model_name}.json", 'w') as fp:
        json.dump(model_config, fp)


def load_model(name):
    path = Path(f"{MODEL_FOLDER}/{name}.pkl")
    if path.is_file():
        model = pd.read_pickle(f"{MODEL_FOLDER}/{name}.pkl")
    else:
        model = False
    return model


def load_model_config(model_name):
    path_str = f"{MODEL_CONFIGS_FOLDER}/{model_name}.json"
    path = Path(path_str)
    if path.is_file():
        with open(path_str, 'r') as fp:
            model_config = json.load(fp)
    else:
        model_config = False
    return model_config



