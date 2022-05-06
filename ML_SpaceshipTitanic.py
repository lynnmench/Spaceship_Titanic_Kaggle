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
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt

#Reading in data files created from Data_Clean_Analyze_SpaceshipTitanic
data_file_path = '/Users/lynnpowell/Documents/DS_Projects/Spaceship_Titanic/'
df_train = pd.read_csv(data_file_path+'Analysis_Train_SpaceTitanic.csv')
df_test = pd.read_csv(data_file_path+'Analysis_Train_SpaceTitanic.csv')

#df_copy = df_train.copy()

#Functions Logistic Regression accuracy with cross validation
def model_accuracy(model, df_train, target):
    train_X = df_train
    target_y = target
    #Using Logist Regression to evaluate by feature selections
    scores = cross_val_score(model, train_X, target_y, cv=10)
    accuracy = np.median(scores)
    return accuracy

#To keep track of Logistic Regression with each feature
model_accuracy = []
model_index = []


#Looking just at the spending features
spending_list = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                 'Spend_Sum', 'Spend_Sum_0', 'RoomService_0','FoodCourt_0',
                 'ShoppingMall_0', 'Spa_0', 'VRDeck_0']
rescale_col = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                 'Spend_Sum']
df_spend = df_train[spending_list]
df_spend_scale = df_spend.copy()
#rescale train_X values
for col in rescale_col:
    df_spend_scale[col] = minmax_scale(df_spend_scale[col])

#spend accuracy
lr = LogisticRegression()
model_accuracy.append(model_accuracy(lr,df_spend, df_train['Transported']))
model_index.append('Spending Feat')

#spend accuracy with rescale
model_accuracy.append(model_accuracy(lr,df_spend_scale))
model_index.append('Spending Rescale Feat')

#checking coefficent of spending not scaled
lr = LogisticRegression()
lr.fit(df_spend, df_train['Transported'])
coefficients = lr.coef_
spend_feat_importance = pd.Series(coefficients[0],
                               index=df_spend.columns).sort_values(ascending=False)
high_spend_feat = spend_feat_importance[spend_feat_importance > 0.2]
#RoomService_0, ShoppingMall_0, VRDeck_0 are to close to Spend_Sum_0
spend_feat = ['Spend_Sum_0','RoomService','ShoppingMall','Spa','VRDeck']
['RoomService', 'Spa','VRDeck','Spend_Sum_$0']

#Looking at age and families
family_list = ['ID_Group','Age','Age_Missing', 'Age_Infant', 'Age_Child','Age_Teenager',
               'Age_Young Adult', 'Age_Adult', 'Age_Senior', 'family_sum',
               'family_kid_num','family_small_kids', 'family_teen',
               'kids_without_adults']

df_family = df_train[family_list]
model_accuracy.append(model_accuracy(lr,df_family, df_train['Transported']))
model_index.append('Family Feat')

#checking coefficent of families and ages
lr = LogisticRegression()
lr.fit(df_family, df_train['Transported'])
coefficients = lr.coef_
family_feat_importance = pd.Series(coefficients[0],
                               index=df_family.columns).abs().sort_values(ascending=False)
high_family_feat = family_feat_importance[family_feat_importance > 0.2]
#family_feat = high_family_feat.index.values.tolist()
#Spend_Sum_0, RoomService_0, ShoppingMall_0, VRDeck_0
family_list.append('Transported')
#Does the correlation of families look like the Logistic Regression coefficent
df_family_cor = df_train[family_list].corr()
cor_family_target = df_family_cor['Transported'].abs().sort_values(ascending=False)
#Features that share high values in both correlation and lr coefficient
#ID Group is low but I don't want to part with it yet
family_feat = ['Age_infant', 'Age_Child', 'family_sum','ID_Group']












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
#suspended animation the spend money is $0. Spend_Sum_$0 also includes passengers
#That are awake that spend $0.
#Final list of pearson features with removing high correlation to other features
pearson_features = ['RoomService', 'Spa','VRDeck','Spend_Sum_$0']


#Removing missing rows from: 
#Age, RoomService, Spa, VRDeck, Cabin, Homeplanet, CryoSleep
#Continue to consider some of their features
drop_miss_cat = ['HomePlanet', 'Cabin']
drop_miss_num = ['Age','RoomService', 'Spa', 'VRDeck']

row_num = df_copy.shape[0]
col_num = df_copy.shape[1]
#rows=8693, col=50
#df_copy['HomePlanet'].value_counts().Missing
#HomePlanet = 201 Missing
#df_copy['Cabin'].value_counts().Missing
#Cabin = 199 Missing
df_low = df_copy[(df_copy.HomePlanet != 'Missing') & (df_copy.Cabin != 'Missing')]
print(df_low.shape)
#rows=8299, col=50, deleted rows = 394

#df_low = df_low[df_low['CryoSleep'] != -1]
for feat in drop_miss_num:
    df_low = df_low[df_low[feat] != -0.5]
df_low['RoomService'].min()
print(df_low.shape)
#rows=8299, col=50, deleted rows = 394

##Dropping features low correlation to transported features
#Dropping features with high correlation to other features
    #To avoid overfitting
drop_cor_feat = ['PassengerId', 'HomePlanet', 'Cabin', 'Destination','ID_Group',
       'VIP', 'FoodCourt', 'ShoppingMall','Age_Categories', 'Spend_Sum',
       'RoomService_0', 'FoodCourt_0', 'ShoppingMall_0', 'Spa_0', 'VRDeck_0',
       'Cabin_Deck', 'Cabin_Side', 'HomePlanet_Missing', 'Destination_55 Cancri e',
       'Destination_Missing', 'Destination_PSO J318.5-22','CryoSleep',
       'Destination_TRAPPIST-1e', 'Age_Missing', 'Age_Child','Age_Infant',
       'Age_Teenager', 'Age_Young Adult', 'Age_Adult', 'Age_Senior',
       'Cabin_Deck_A', 'Cabin_Deck_D','Cabin_Deck_G', 'Cabin_Deck_Missing',
       'Cabin_Deck_T', 'Cabin_Side_Missing', 'Cabin_Side_S']
df_low = df_low.drop(drop_cor_feat,axis=1)
row_del_total = row_num - df_low.shape[0]
col_del_total = col_num - df_low.shape[1]
print(df_low.shape)
#rows=7594, col=14, deleted cols = 36


#Wrapper Method -> Recursive Feature Elemination (RFE) with Cross-Validation (CV)
numerical_train = df_train.select_dtypes(include=['float64','int64'])
#columns = df_low.columns.drop('Transported')
columns = numerical_train.columns.drop('Transported')
#df_low.describe(include='all',percentiles=[])

train_X = df_train[columns]
target = df_train['Transported']
#train_X = df_low[columns]
#target = df_low['Transported']

lr = LogisticRegression(solver='lbfgs', max_iter=3000)
selector = RFECV(lr, cv=10)
selector.fit(train_X, target)
optimized_columns = train_X.columns[selector.support_]
gs = selector.grid_scores_
print(selector.support_)



