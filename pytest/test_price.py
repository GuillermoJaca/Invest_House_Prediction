#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:51:49 2022

@author: guillermogarcia
"""
import pytest
import pandas as pd
import numpy as np
import json

def trans_bool_features(data,bool_features):
    for col_bool in bool_features: 
        data[col_bool] = data[col_bool].astype(int)
    return data

###### DATA IMMO

@pytest.fixture
def data_immo():
  data_immo = pd.read_csv('data/immo_data.csv')
  return data_immo

def test_given_data_immo_when_convert_categorical_features_then_integer_column(data_immo):
    
    bool_features = ['newlyConst','balcony','hasKitchen','cellar','lift','garden']
    data_immo = trans_bool_features(data_immo,bool_features)
    assert ((data_immo[bool_features].dtypes == np.int64).sum() == len(bool_features))


####### TEST SET

@pytest.fixture
def data_test():
  data_test = pd.read_parquet('data/data_test.parquet.gzip')
  return data_test

@pytest.fixture
def back_mapping():
    with open('data/back_mapping.json', 'r') as fp:
        back_mapping = json.load(fp)
    return back_mapping

def test_given_data_when_back_mapping_then_numeric_column(data_test,back_mapping):
  data_test = data_test.replace(back_mapping)
  data_test['streetPlain'] = pd.to_numeric(data_test['streetPlain'], errors='coerce').fillna(99999)
  assert data_test['streetPlain'].dtypes == np.float64 
  
  
  
  
  