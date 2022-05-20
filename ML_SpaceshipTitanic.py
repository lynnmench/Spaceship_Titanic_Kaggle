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

New to data science and I'm intersted in seeing if numeric values like
age are better in bins or left alone in column. Playing with compareing
overfited data on models (Linear Regressoin, Logistic Regression, Random Forest
and K-Nearest Neighbor) with a set of data that will be both cros validated
and train/test split.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm


#Reading in data files created from Data_Clean_Analyze_SpaceshipTitanic
data_file_path = '/Users/lynnpowell/Documents/DS_Projects/Spaceship_Titanic/'
df_train = pd.read_csv(data_file_path+'Initial_Analysis_Train_SpaceTitanic.csv')
df_test = pd.read_csv(data_file_path+'Initial_Analysis_Test_SpaceTitanic.csv')

#df_copy = df_train.copy()

#Function: Logistic Regression accuracy with cross validation
#Used to compate Feature selection methods
def model_accuracy(model, df_train, target):
    train_X = df_train
    target_y = target
    #Using Logist Regression to evaluate by feature selections
    scores = cross_val_score(model, train_X, target_y,
                             scoring = 'neg_mean_absolute_error',cv=10)
    accuracy = np.mean(scores)
    return accuracy


#Function: Test data set on multiple models
#Linear Regression, Lasso Regression (Alpha Search),
#Random Forest Classifier (Grid Search), K-Nearest Neighbor (Grid Search)
def ml_model_full(df_train, target, set_name):
    model_acc_list = [] 
    model_acc_index = []
    model_time = []
    i=0
    model_list = []
    model_name = []
    
    #Linear Regression
    linr = LinearRegression()
    model_list.append(linr)
    model_name.append(set_name+' Linear Regression')
    
    #Logistic Regression
    logr = LogisticRegression()
    model_list.append(logr)
    model_name.append(set_name+' Logistic Regression')
    
    #Random Forest Classifier (grid search)
    rfc_hyperpar = {
        'criterion' : ['entropy', 'gini'],
        'max_depth' : [5, 10],
        'max_features' : ['log2', 'sqrt'],
        'min_samples_leaf' : [1,5],
        'min_samples_split' : [3,5],
        'n_estimators' : [6,9]
    }
    rfc = RandomForestClassifier(random_state=7)
    rfc_grid = GridSearchCV(rfc, param_grid=rfc_hyperpar, cv=10)
    rfc_grid.fit(df_train, target)
    #rfc_best_params = grid.best_params_
    #rfc_best_score = grid.best_score_
    best_rfc = rfc_grid.best_estimator_
    model_list.append(best_rfc)
    model_name.append(set_name+' Random Forest Classifier')
    
    #K-Nearest Neighbors
    knn_hyperpar = {
        "n_neighbors": range(1,20,2),
        "weights": ["distance", "uniform"],
        "algorithm": ['brute'],
        "p": [1,2]
    }
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn, param_grid=knn_hyperpar, cv=10)
    knn_grid.fit(df_train, target)
    #knn_best_params = grid.best_params_
    #knn_best_score = grid.best_score_
    best_knn = knn_grid.best_estimator_
    model_list.append(best_knn)
    model_name.append(set_name+' K Nearest Neighbors')

    
    #Calculate the model accuracy
    for ml_model in model_list:
        start_time = time.time()
        model_acc_list.append(model_accuracy(ml_model,df_train, target))
        model_acc_index.append(model_name[i])
        model_time.append(time.time() - start_time)
        i+=1
    
    return model_acc_list, model_acc_index, model_time

def ml_model_split(train, target, set_name):
    split_accuracy = [] 
    split_acc_index = []
    model_time = []
    
    train_X, test_X, train_y, test_y = train_test_split(
        train, target, test_size=0.2,random_state=7)

    #Logistic Regression
    logr = LogisticRegression()
    logr.fit(train_X,train_y)
    #logr_predictions = logr.predict(test_X)
    #logr_mean_predic = mean_absolute_error(test_y, logr_predictions)
    split_accuracy.append(logr.score(train_X, train_y))
    split_acc_index(set_name+' LogR Split Train')
    split_accuracy.append(logr.score(test_X, test_y))
    split_acc_index(set_name+' LogR Split Test')


    #Random Forest Classifier (grid search)
    rfc_hyper = {
        'criterion' : ['entropy', 'gini'],
        'max_depth' : [5, 10],
        'max_features' : ['log2', 'sqrt'],
        'min_samples_leaf' : [1,5],
        'min_samples_split' : [3,5],
        'n_estimators' : [6,9]
    }
    rfc = RandomForestClassifier(random_state=7)
    rfc_grid = GridSearchCV(rfc, param_grid=rfc_hyper, cv=10)
    rfc_grid.fit(x_train, target)
    #best_params = grid.best_params_
    #best_score = grid.best_score_
    #print(best_params)
    #print(best_score)
    best_rfc = rfc_grid.best_estimator_
    best_rfc.fit(train_X,train_y)
    rfc_predictions = best_rfc.predict(test_X)
    rfc_mean_predic = mean_absolute_error(test_y, rfc_predictions)
    rfc_score_train = best_rfc.score(train_X, train_y)
    rfc_score_test = best_rfc.score(test_X, test_y)
    #model_acc.append(mean_absolute_error(test_y, predictions))
    #model_index.append('Init Unseen RFC')

    #K-Nearest Neighbors
    knn_hyper = {
        "n_neighbors": range(1,20,2),
        "weights": ["distance", "uniform"],
        "algorithm": ['brute'],
        "p": [1,2]
    }
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn, param_grid=knn_hyper, cv=10)
    knn_grid.fit(x_train, target)
    #print(grid.best_params_)
    #print(grid.best_score_)
    best_knn = knn_grid.best_estimator_
    best_knn.fit(train_X,train_y)
    knn_predictions = best_knn.predict(test_X)
    knn_mean_predic = mean_absolute_error(test_y, knn_predictions)
    knn_score_train = best_knn.score(train_X, train_y)
    knn_score_test = best_knn.score(test_X, test_y)
    #model_acc.append(mean_absolute_error(test_y, predictions))
    #model_index.append('Init Unseen KNN')

    
    return split_accuracy, split_acc_index
    


#To keep track of the prediction accuracy with each feature and ML model
fs_accuracy = []
fs_index = []
fs_time = []


#Looking just at the spending features
#Bins, Rescale and unaltered columns
#Comparing Correlatin and Coefficient
#Leaving Spend_Sum off for now becuase it was created from the other columns
#and I don't want it to play a factor now
spend_original = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
              'Spend_Sum']
spend_bins = ['RoomService_Missing', 'RoomService_$0',
       'RoomService_$0.1_2000', 'RoomService_$2001_30000', 'FoodCourt_Missing',
       'FoodCourt_$0', 'FoodCourt_$0.1_2000', 'FoodCourt_$2001_30000',
       'ShoppingMall_Missing', 'ShoppingMall_$0', 'ShoppingMall_$0.1_2000',
       'ShoppingMall_$2001_30000', 'Spa_Missing', 'Spa_$0', 'Spa_$0.1_2000',
       'Spa_$2001_30000', 'Spend_Sum_Missing', 'Spend_Sum_$0',
       'Spend_Sum_$0.1_500', 'Spend_Sum_$501_1000', 'Spend_Sum_$1001_1500',
       'Spend_Sum_$1501_2500', 'Spend_Sum_$2501_4000', 'Spend_Sum_$4001_36000',
       'VRDeck_Missing', 'VRDeck_$0', 'VRDeck_$0.1_2000', 'VRDeck_$2001_30000',]
rescale_col = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                 'Spend_Sum']

df_spend_original = df_train[spend_original]
df_spend_bins = df_train[spend_bins]
df_spend_scale = df_train[rescale_col]
#rescale train_X values
for col in rescale_col:
    df_spend_scale[col] = minmax_scale(df_spend_scale[col])

#spend accuracy location
logr = LogisticRegression()
start_time = time.time()
fs_accuracy.append(model_accuracy(logr,df_spend_original, df_train['Transported']))
fs_index.append('Feat Select Spend Orignial')
fs_time.append(time.time() - start_time)
#spend accuracy with bins
start_time = time.time()
fs_accuracy.append(model_accuracy(logr,df_spend_bins, df_train['Transported']))
fs_index.append('Feat Select Spend Bins')
fs_time.append(time.time() - start_time)
#spend accuracy with rescale
start_time = time.time()
fs_accuracy.append(model_accuracy(logr,df_spend_scale, df_train['Transported']))
fs_index.append('Feat Select Spend Rescale')
fs_time.append(time.time() - start_time)

#Looking at coefficients for the normalized data
#Rescaled values spent at each location
logr.fit(df_spend_scale, df_train['Transported'])
coefficients = logr.coef_
spend_coeff_rescale = abs(pd.Series(coefficients[0], 
                                    index=df_spend_scale.columns)).sort_values(ascending=False)

#Coefficient of Spend Bins
logr.fit(df_spend_bins, df_train['Transported'])
coefficients = logr.coef_
spend_coeff_bins = abs(pd.Series(coefficients[0],
                               index=df_spend_bins.columns)).sort_values(ascending=False)
spend_coeff_bins = spend_coeff_bins[spend_coeff_bins > 0.9].index.values.tolist()

#Chekcing correlation of spending
#spend_feat = ['Transported']+spend_loca+spend_bins
spend_feat = spend_bins + ['Transported'] + rescale_col
df_spend = df_train[spend_feat]
#rescale train_X values
for col in rescale_col:
    df_spend[col] = minmax_scale(df_spend[col])
spend_cor = df_spend.corr()
target_spend_cor = spend_cor['Transported'].abs().sort_values(ascending=False)
target_spend_bin_cor = target_spend_cor.drop(labels=(rescale_col+['Transported']))
spend_bins_feat = target_spend_bin_cor[target_spend_bin_cor > 0.27].index.values.tolist()

#Spending feature list for ML models
spend_feat_rescale = spend_coeff_rescale[spend_coeff_rescale > 8].index.values.tolist()
spend_feat = np.unique(spend_coeff_bins + spend_bins_feat).tolist()

#Looking at age and families
family_list = ['ID_Group','Age','Age_Missing', 'Age_Infant', 'Age_Child','Age_Teenager',
               'Age_Young Adult', 'Age_Adult', 'Age_Senior', 'family_sum',
               'family_kid_num','family_small_kids', 'family_teen',
               'kids_without_adults', 'Cryo_Fam']

df_family = df_train[family_list]
start_time = time.time()
fs_accuracy.append(model_accuracy(logr,df_family, df_train['Transported']))
fs_index.append('Family Feat')
fs_time.append(time.time() - start_time)

#checking coefficent of families and ages
logr.fit(df_family, df_train['Transported'])
coefficients = logr.coef_
family_coeff = pd.Series(coefficients[0],
                               index=df_family.columns).abs().sort_values(ascending=False)
family_coeff_feat = family_coeff[family_coeff > 0.35].index.values.tolist()
#family_feat = high_family_feat.index.values.tolist()
#Spend_Sum_0, RoomService_0, ShoppingMall_0, VRDeck_0
family_list.append('Transported')
#Does the correlation of families look like the Logistic Regression coefficent
df_family_cor = df_train[family_list].corr()
cor_family_target = df_family_cor['Transported'].abs().sort_values(ascending=False)
cor_family_target = cor_family_target.drop(labels=(['Age','Transported']))
family_cor_feat = cor_family_target[cor_family_target > 0.06].index.values.tolist()

#Cryo_Fam has high correlation becuase CryoSleep has high correlation
#Features that share high values in both correlation and lr coefficient
family_feat_rescale = ['Age','family_sum']
family_feat = np.unique(family_coeff_feat + family_cor_feat).tolist()
family_drop = ['Cryo_Fam','Age_Young Adult']
for x in family_drop:
    family_feat.remove(x)


#Analyzing remaining features: Destination, Cabin, Home planet and VIP
#Not analyzing Cryo Sleep becuase it is to close to Spend Sum $0
status_list = ['CryoSleep', 'VIP', 'HomePlanet_Earth',
       'HomePlanet_Europa', 'HomePlanet_Mars', 'HomePlanet_Missing',
       'Destination_55 Cancri e', 'Destination_Missing',
       'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e',
       'Cabin_Deck_A', 'Cabin_Deck_B', 'Cabin_Deck_C', 'Cabin_Deck_D',
       'Cabin_Deck_E','Cabin_Deck_F', 'Cabin_Deck_G', 'Cabin_Deck_Missing',
       'Cabin_Deck_T', 'Cabin_Side_Missing', 'Cabin_Side_P', 'Cabin_Side_S']

df_status = df_train[status_list]
start_time = time.time()
fs_accuracy.append(model_accuracy(logr,df_status, df_train['Transported']))
fs_index.append('Status Feat')
fs_time.append(time.time() - start_time)

#checking coefficent of status features
lr = LogisticRegression()
lr.fit(df_status, df_train['Transported'])
coefficients = lr.coef_
status_feat_importance = pd.Series(coefficients[0],
                               index=df_status.columns).abs().sort_values(ascending=False)
high_status_feat = status_feat_importance[status_feat_importance > 0.28].index.values.tolist()

#Checking correlation of status features
status_list.append('Transported')
df_status_cor = df_train[status_list].corr()
cor_status_target = df_status_cor['Transported'].abs().sort_values(ascending=False)
cor_status_target = cor_status_target.drop(labels=('Transported'))
status_cor_feat = cor_status_target[cor_status_target > 0.09].index.values.tolist()

#Features that share high values in both correlation and lr coefficient
status_feat = np.unique(high_status_feat + status_cor_feat).tolist()

#Rescale features and status features
#compareing correlation to make sure none of them are to close to each other
rescale_feat = spend_feat_rescale + family_feat_rescale
rescale_all = rescale_feat + status_feat
df_rescale = df_train[rescale_all]
for col in rescale_feat:
    df_rescale[col] = minmax_scale(df_rescale[col])

df_rescale_cor = df_rescale.corr().abs()
#Cabin Side P and Cabin Side S have high correlation
rescale_drop = ['Cabin_Side_P','Destination_TRAPPIST-1e',
                'HomePlanet_Earth','Cabin_Deck_T']
df_rescale = df_rescale.drop(rescale_drop,axis=1)

#compareing correlation to make sure none of them are to close to each other
feat_cor_coeff = spend_feat + family_feat + status_feat
df_cor_coeff = df_train[feat_cor_coeff]
full_feat_corr = df_cor_coeff.corr().abs()
cryo_corr = full_feat_corr['CryoSleep'].abs().sort_values(ascending=False)
cryo_corr = cryo_corr.drop(labels=('CryoSleep'))
high_cryo_corr = cryo_corr[cryo_corr > 0.53].index.values.tolist()


high_match_corr = high_cryo_corr+['Cabin_Side_P','Destination_TRAPPIST-1e',
                                  'HomePlanet_Earth','Cabin_Deck_T']
df_cor_coeff = df_cor_coeff.drop(high_match_corr, axis=1)


#Final Data Frames to run ML models on
#df_rescale
#df_cor_coeff
acc_full_rescale = ml_model_full(df_rescale,df_train['Transported'],'Rescale')
fs_accuracy = fs_accuracy + acc_full_rescale[0]
fs_index = fs_index + acc_full_rescale[1]
fs_time = fs_time + acc_full_rescale[2]

acc_corr_coeff = ml_model_full(df_cor_coeff,df_train['Transported'],'Corr Coeff')
fs_accuracy = fs_accuracy + acc_corr_coeff[0]
fs_index = fs_index + acc_corr_coeff[1]
fs_time = fs_time + acc_corr_coeff[2]


#Using Split to evaluate on parts of the data frame the model has not seen
ml_model_split(train, target, set_name)



#Initial list after anazlying the spending, families and status of passengers
#Using Pearson Correlation to see feature relations to Transported
#Also refered to as the Filter Method in Feature Selection with sklearn and Pandas
#Comparing correlation feature values to target column
init_feat = ['Transported'] + spend_feat + family_feat + status_features
df_init = df_train[init_feat].copy()

df_cor = df_init.corr()
cor_target = abs(df_cor['Transported']).sort_values(ascending=False)

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features_list = relevant_features.index.values.tolist()
df_relevant = df_init[relevant_features_list]
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

#From correlation with Transported was higher with Money spent, 
#home planet and cabin. Destination and family had a
#lower correlation with transported. 
#Home planet Europa has high correlation with Cablin Deck B and C.
#Spend $0 had higher correlation to the locations money was spent.
#No other features appear to have high correlation

#Baseline of Model Methods
x_train=df_init.drop('Transported', axis=1)
target=df_init['Transported']

acc = ML_Model_FullDF(x_train, target, 'Base')
print(acc[0])
print(acc[1])

#Linear Regression
linr = LinearRegression()
model_acc.append(model_accuracy(linr,x_train, target))
model_index.append('Init Linear Regression')

#Lasso Regression with alpha search
#default alpha=1, best alpha=
lml = Lasso(alpha=0.1)
print(model_accuracy(lml,x_train, target))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/1000)
    lml = Lasso(alpha=(i/1000))
    error.append(model_accuracy(lml,x_train, target))

plt.plot(alpha, error)
err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
print(df_err[df_err.error == max(df_err.error)])
model_acc.append(model_accuracy(lml,x_train, target))
model_index.append('Init Lasso Regression')

#Random Forest Classifier (grid search)
hyperparameters = {
    'criterion' : ['entropy', 'gini'],
    'max_depth' : [5, 10],
    'max_features' : ['log2', 'sqrt'],
    'min_samples_leaf' : [1,5],
    'min_samples_split' : [3,5],
    'n_estimators' : [6,9]
}
rfc = RandomForestClassifier(random_state=7)
grid = GridSearchCV(rfc, param_grid=hyperparameters, cv=10)
grid.fit(x_train, target)
best_params = grid.best_params_
best_score = grid.best_score_
print(best_params)
print(best_score)
best_rfc = grid.best_estimator_
model_acc.append(model_accuracy(best_rfc,x_train, target))
model_index.append('Init Random Forest Classifier')

#K-Nearest Neighbors
hyperparameters = {
    "n_neighbors": range(1,20,2),
    "weights": ["distance", "uniform"],
    "algorithm": ['brute'],
    "p": [1,2]
}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid=hyperparameters, cv=10)
grid.fit(x_train, target)
print(grid.best_params_)
print(grid.best_score_)
best_knn = grid.best_estimator_
model_acc.append(model_accuracy(best_knn,x_train, target))
model_index.append('Init K Nearest Neighbors')


#From initial dataframe best ML models:
#Logistic Regression (spending featrues), Random Forest Classifier, K-Nearest Neighbors
#checking methods on data it would not have seen

train=df_init.drop('Transported', axis=1)
target=df_init['Transported']
train_X, test_X, train_y, test_y = train_test_split(
    train, target, test_size=0.2,random_state=7)

#Logistic Regression
lr.fit(train_X,train_y)
predictions = lr.predict(test_X)
model_acc.append(mean_absolute_error(test_y, predictions))
model_index.append('Init Unseen LR')

#Random Forest Classifier (grid search)
rfc_hyper = {
    'criterion' : ['entropy', 'gini'],
    'max_depth' : [5, 10],
    'max_features' : ['log2', 'sqrt'],
    'min_samples_leaf' : [1,5],
    'min_samples_split' : [3,5],
    'n_estimators' : [6,9]
}
rfc = RandomForestClassifier(random_state=7)
rfc_grid = GridSearchCV(rfc, param_grid=rfc_hyper, cv=10)
rfc_grid.fit(x_train, target)
#best_params = grid.best_params_
#best_score = grid.best_score_
#print(best_params)
#print(best_score)
best_rfc = rfc_grid.best_estimator_
best_rfc.fit(train_X,train_y)
rfc_predictions = best_rfc.predict(test_X)
rfc_mean_predic = mean_absolute_error(test_y, rfc_predictions)
rfc_score_train = best_rfc.score(train_X, train_y)
rfc_score_test = best_rfc.score(test_X, test_y)
#model_acc.append(mean_absolute_error(test_y, predictions))
#model_index.append('Init Unseen RFC')

#K-Nearest Neighbors
knn_hyper = {
    "n_neighbors": range(1,20,2),
    "weights": ["distance", "uniform"],
    "algorithm": ['brute'],
    "p": [1,2]
}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, param_grid=knn_hyper, cv=10)
knn_grid.fit(x_train, target)
#print(grid.best_params_)
#print(grid.best_score_)
best_knn = knn_grid.best_estimator_
best_knn.fit(train_X,train_y)
knn_predictions = best_knn.predict(test_X)
knn_mean_predic = mean_absolute_error(test_y, knn_predictions)
knn_score_train = best_knn.score(train_X, train_y)
knn_score_test = best_knn.score(test_X, test_y)
#model_acc.append(mean_absolute_error(test_y, predictions))
#model_index.append('Init Unseen KNN')


#Wrapper Method -> Recursive Feature Elemination (RFE) with Cross-Validation (CV)



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



