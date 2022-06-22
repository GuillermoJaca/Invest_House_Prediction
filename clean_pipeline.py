#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:14:46 2022

@author: guillermogarcia
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from featexp import get_univariate_plots

# TARGET TRANSFORMATIONS

def create_new_features_target(data):
    data['baseRentSquareMeter'] = data['baseRent'] / data['livingSpace']
    data['price_costs'] = data['totalRent'] - data['baseRent']
    data['price_costs_square_meter'] = data['price_costs'] / data['livingSpace']
    return data

def clean_possible_errors(data):
    
    #if price is 0 it's an error
    data['totalRent'].replace(0, np.nan, inplace=True) 
    data['baseRent'].replace(0, np.nan, inplace=True)
    
    #Hauses with no livingSpace are wrong
    sel0 = (data['livingSpace'] == 0)
    
    #Unusual high values of baseRent
    sel1 = (data['livingSpace'] <= 10) & (data['baseRentSquareMeter'] > 50)
    sel1prima = (data['baseRent'] > 50000)
    sel1primaprima = (data['baseRentSquareMeter'] > 200)
    
    #Houses with more than one room and with less than 10 square meter are probably wrong
    sel2 = (data['livingSpace'] < 10) & (data['noRooms'] > 1)
    
    #Unusual high values of priceCosts
    sel3 = (data['price_costs_square_meter'] > 50)
    
    #Unusual low values of baseRent
    sel4 = (data['baseRentSquareMeter'] < 1) & (data['baseRent'] < 100)
    
    #Unusual low values of priceCosts
    sel5 = (data['price_costs'] < 0)
    
    print(len(data[(sel0 | sel1 | sel1prima | sel1primaprima | sel2 | sel3 | sel4 | sel5)]),'data points removed!')
    
    return data[~(sel0 | sel1 | sel1prima | sel1primaprima | sel2 | sel3 | sel4 | sel5)] ,data[(sel0 | sel1 | sel1prima | sel1primaprima | sel2 | sel3 | sel4 | sel5)] 

# CATEGORICAL FUNCTIONS

def create_nan_class(df,columns):
    '''
    Create a new class with name 'nan' in the columns given
    '''
    
    for col in columns:
        df[col] = df[col].apply(lambda x: x if not pd.isnull(x) else 'nan')
        
    return df

def trans_bool_features(data,bool_features):
    for col_bool in bool_features: 
        data[col_bool] = data[col_bool].astype(int)
    return data

def firing_types_clean(data):
    target_class = 'totalRent'
    features_firing_to_cap = list((data.groupby('firingTypes')[target_class].count()[(data.groupby('firingTypes')[target_class].count() < 1500)]).index)
    sel = (data['firingTypes'].isin(features_firing_to_cap))
    data.loc[sel,'firingTypes'] = 'too_little_count'
    return data

def energyEfficiencyClass_transform(data):
    di = {'A_PLUS':0, 'A':1 ,"B":2, 'C':3,'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'NO_INFORMATION':9}
    data = data.replace({"energyEfficiencyClass": di})
    return data

# CONTINUOUS FUNCTIONS

def cap_outliers_features(df_train,dict_features):
    '''
    Cap outliers from one value ownwards.
    '''
    for key, val in dict_features.items():
        df_train[f'{key}_capped_outliers'] = df_train[key].apply(lambda x: val if x >= val else x)
    
    return df_train

def room_features(data):
    #Get noRooms that make no sense
    data['SpacePerRoom'] = data['livingSpace'] / data['noRooms']
    sel = (data['SpacePerRoom'] < 5) & (data['livingSpace'] > 5)

    #Substitute for nan
    data.loc[sel,'noRooms'] = np.nan
    data['SpacePerRoom'] = data['livingSpace'] / data['noRooms']

    #Substitute nan for -999 for having 2 options
    data['noRooms_-999'] = data['noRooms'].fillna(-999, inplace=True)
    data['SpacePerRoom_-999'] = data['SpacePerRoom'].fillna(-999, inplace=True)

    return data

def floor_features(data):
    sel = data['numberOfFloors'] - data['floor'] < 0 #cases wrong (floor > numberOfFloors)
    data.loc[sel,'numberOfFloors'] = data.loc[sel,'floor']
    data['relative_floor'] = data['floor'] / data['numberOfFloors']
    data['lift1-1'] = data['lift'].apply(lambda x: x if x > 0 else -1)
    data['floor_lift'] = data['floor'] * data['lift1-1']
    
    return data

# YEAR FEATURES

def year_features(data):
    data['date_number'] = data['date'].apply(lambda x: 2000 + int(x[-2:]))
    data['years_old'] = data['date_number'] - data['yearConstructed']
    data['years_last_refurbish'] = data['date_number'] - data['lastRefurbish']
    
    data['years_old'] = data['years_old'].apply(lambda x:x if x >= 0 else 0)
    data['years_last_refurbish'] = data['years_last_refurbish'].apply(lambda x:x if x >= 0 else 0)
    
    return data

# LOCATION FEATURES

def group_statistics_mean(data_,feature,target):
    name_new_column = f'{feature}_{target}_mean'
    temp = data_.groupby(feature)[target].agg(['mean']).rename({'mean':name_new_column},axis=1)
    data_ = pd.merge(data_,temp,on=feature,how='left')
    return data_[name_new_column], name_new_column

def location_features_mean(data):
    for target_aux in ['baseRent','price_costs']:
        for loc_feat in ['geo_bln','geo_krs','geo_plz','streetPlain']:

            data_new_feat, name_new_column = group_statistics_mean(data_ = data,
                            feature = loc_feat,
                            target = target_aux)

            data.loc[:,name_new_column] = np.array(data_new_feat)
            
    return data

def group_statistics_size(data_,feature,target):
    name_new_column = f'{feature}_{target}_size'
    temp = data_.groupby(feature)[target].agg(['size']).rename({'size':name_new_column},axis=1)
    data_ = pd.merge(data_,temp,on=feature,how='left')
    return data_[name_new_column], name_new_column

def features_size(data,list_features):
    for loc_feat in list_features:
        data_new_feat, name_new_column = group_statistics_size(data_ = data,
                            feature = loc_feat,
                            target = 'baseRent')

        data.loc[:,name_new_column] = np.array(data_new_feat)
        
    return data


def pipeline(data,test = False):
    
    ###### FEATURES #####
    target_class = 'totalRent'
    ignore_features = ["scoutId", "pricetrend", "yearConstructedRange", "noRoomsRange",
                       "livingSpaceRange", "thermalChar", "street", "picturecount", "date"]
    
    #Boolean features
    bool_features = ['newlyConst','balcony','hasKitchen','cellar','lift','garden']
    
    #Categorical
    transform2categorical = ['heatingType','telekomTvOffer','firingTypes','condition','interiorQual',
                             'typeOfFlat','energyEfficiencyClass','electricityBasePrice','petsAllowed'] 
    
    features_add_nan_class = ['heatingType','telekomTvOffer','condition','interiorQual','typeOfFlat',
                              'electricityBasePrice','petsAllowed']
    
    #Continuous
    continuous_features = ['serviceCharge','noParkSpaces',
                         'baseRent','livingSpace','baseRentRange',
                         'noRooms','floor','numberOfFloors',
                         'heatingCosts','electricityKwhPrice'] 
    
    telekom_features = ['telekomUploadSpeed','telekomHybridUploadSpeed']
    
    transform2continuous = ['date','yearConstructed','lastRefurbish']
    
    #Location
    location_features = ['geo_bln','geo_krs','geo_plz','streetPlain']
    
    
    #Create new target features
    data = create_new_features_target(data)
    #Clean possible errors
    print('targets...')
    if test == True:
        data_clean, removed_data = clean_possible_errors(data)
        removed_data.to_csv('removed_test_data.csv',index = False)
    else:
        data_clean,_ = clean_possible_errors(data)
    
    #Bool_features
    print('bool...')
    data_clean = trans_bool_features(data = data_clean,
                                    bool_features = bool_features)
    
    # Categorical
    print('categorical...')
    #data_clean = create_nan_class(df = data_clean, columns = features_add_nan_class)
    data_clean = firing_types_clean(data_clean)
    data_clean = energyEfficiencyClass_transform(data_clean)
    
    #Continuous
    print('continuous...')
    dict_features_cap = {'serviceCharge':250,'noParkSpaces':2}
    data_clean = cap_outliers_features(df_train = data_clean, dict_features = dict_features_cap)
    data_clean = floor_features(data = data_clean)
    
    #Year features
    print('year features...')
    data_clean = year_features(data_clean)
    
    #Location features
    print('location mean...')
    data_clean = location_features_mean(data_clean)
    print('location size...')
    data_clean = features_size(data_clean,
                               list_features = ['geo_bln','geo_krs','geo_plz','streetPlain'])
    
    #More size features 
    data_clean = features_size(data_clean,
                               list_features = ['garden', 'heatingType', 'telekomTvOffer',
                                                'condition', 'interiorQual', 'typeOfFlat'])
    
    return data_clean


def main():
    # Load data
    data_immo = pd.read_csv('data/immo_data.csv')
    data,test_set = train_test_split(data_immo,train_size = 0.8,random_state = 42)

    #Apply cleaining pipeline
    data_train = pipeline(data)
    data_train.to_parquet('data/data_train.parquet.gzip',compression='gzip')  
    data_test = pipeline(test_set,test = True)
    data_test.to_parquet('data/data_test.parquet.gzip',compression='gzip')  

if __name__ == "__main__":
    main()
    

