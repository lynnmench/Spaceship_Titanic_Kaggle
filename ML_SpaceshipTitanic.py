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

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_file_path = '/Users/lynnpowell/Documents/DS_Projects/Spaceship_Titanic/'
df_train = pd.read_csv(data_file_path+'Analysis_Train_SpaceTitanic.csv')
df_test = pd.read_csv(data_file_path+'Analysis_Train_SpaceTitanic.csv')

#Using Pearson Correlation to see feature relations to Transported
#plt.figure(figsize=(12,10))
df_cor = df_train.corr()
#sns.heatmap(df_cor, annot=True, cmap=plt.cm.Reds)
#plt.show()
#To many columns to acuratly see results

#Comparing features to target column
cor_target = abs(df_cor['Transported'])

#Selecting highly correlated features
pearson_relevant_features = cor_target[cor_target>0.2]
df_pearson_relevant = df_train[pearson_relevant_features.index.values.tolist()]
#Compare features to each other to avoid overfitting model
#This time dropping the features with high correlation values
pearson_cor = df_pearson_relevant[pearson_relevant_features.index.values.tolist()].corr()

