#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 17Feb2022
Author: Lynn Menchaca

Resources:
Author: Abhini Shetye
Title: Feature Selection with sklearn and Pandas
https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
"""
"""
From the data cleaning and initial analysis file the features that look
like they would play an impact:

Age_Infant, HomePlanet_Earth, HomePlanet_Europa, CryoSleep,
Cabin_Deck_B,C,E,F, Cabin_Side_S
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data_file_path = '/Users/lynnpowell/Documents/DS_Projects/Spaceship_Titanic/'
df_train = pd.read_csv(data_file_path+'Analysis_Train_SpaceTitanic.csv')
df_test = pd.read_csv(data_file_path+'Analysis_Train_SpaceTitanic.csv')

#Using Pearson Correlation to see feature relations to Transported
#Also refered to as the Filter Method in Feature Selection with sklearn and Pandas
#Comparing correlation feature values to target column

df_copy = df_train.copy()


df_cor = df_train.corr()
cor_target = abs(df_cor['Transported'])

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
relevant_features_list = relevant_features.index.values.tolist()
df_relevant = df_train[relevant_features_list]
#Compare features to each other to avoid overfitting model
#This time dropping the features with high correlation values
relevant_cor = df_relevant[relevant_features_list].corr()
cor_relevant_high = []

for feat_cor in relevant_features_list:
    print(feat_cor)
    cor_compare = abs(relevant_cor[feat_cor])
    relev_high = cor_compare[cor_compare>0.3]
    relev_high_list = relev_high.index.values.tolist()
    cor_relevant_high.append(relev_high_list)
    print('Features with high correlation to {}: '.format(feat_cor))
    print(relev_high_list)

#Looking at the relevant_cor data
#Cryosleep has high correlation with with money being spent.
#This makes sense becuase if the passenger choose to be put in 
#suspended animation the spend money is $0.
#However to consider the people awake I will keep CryoSleep feature
#Using the Person Correlation method the featrues that correlate the most
#is cryo sleep and money spent
#Final list of pearson features with low correlation to the other features
pearson_features = ['CryoSleep', 'Spend_Sum_$0']
#This method is hard becuase so many of the features relate to each other
#with spending and cryo sleep.
#This method would work better if the features were more independent of each other

#Removing missing rows from: 
#Age, RoomService, Spa, VRDeck, Cabin, Homeplanet, CryoSleep
#Continue to consider some of their features
drop_miss_cat = ['HomePlanet', 'Cabin']
drop_miss_num = ['CryoSleep', 'Age','RoomService', 'Spa', 'VRDeck']

print(df_copy.shape)
#rows=8693, col=50
#df_copy['HomePlanet'].value_counts().Missing
#HomePlanet = 201 Missing
#df_copy['Cabin'].value_counts().Missing
#Cabin = 199 Missing
df_low = df_copy[(df_copy.HomePlanet != 'Missing') & (df_copy.Cabin != 'Missing')]
print(df_low.shape)
#rows=8299, col=50, deleted rows = 394

df_low = df_low[df_low['CryoSleep'] != -1]
for feat in drop_miss_num:
    df_low = df_low[df_low[feat] != -0.5]
print(df_low.shape)
#rows7594, col=50, deleted rows = 705

df_low['RoomService'].min()

##Dropping features low correlation to transported features
#Dropping features with high correlation to other features
    #To avoid overfitting
drop_cor_feat = ['PassengerId', 'HomePlanet', 'Cabin', 'Destination', 'Age',
       'VIP', 'FoodCourt', 'ShoppingMall','Age_Categories', 'Spend_Sum',
       'RoomService_0', 'FoodCourt_0', 'ShoppingMall_0', 'Spa_0', 'VRDeck_0',
       'Cabin_Deck', 'Cabin_Side', 'HomePlanet_Missing', 'Destination_55 Cancri e',
       'Destination_Missing', 'Destination_PSO J318.5-22',
       'Destination_TRAPPIST-1e', 'Age_Missing', 'Age_Child',
       'Age_Teenager', 'Age_Young Adult', 'Age_Adult', 'Age_Senior',
       'Cabin_Deck_A', 'Cabin_Deck_D','Cabin_Deck_G', 'Cabin_Deck_Missing',
       'Cabin_Deck_T', 'Cabin_Side_Missing', 'Cabin_Side_S']
df_low = df_low.drop(drop_cor_feat,axis=1)
print(df_low.shape)
#rows=7594, col=15, deleted cols = 35


#Wrapper Method
df_low.columns



