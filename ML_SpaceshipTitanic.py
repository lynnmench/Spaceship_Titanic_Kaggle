#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 17Feb2022
Author: Lynn Menchaca

Resources:
Kaggle Competition -> Spaceship Titanic
Data Quest -> Data Science Lectures
"""
"""
From the data cleaning and initial analysis file initial theory of
the features that have a high impact on transporation results:

Age_Infant, HomePlanet_Earth, HomePlanet_Europa, CryoSleep,
Cabin_Deck_B,C,E,F, Cabin_Side_S

New to data science and I'm intersted if select dummy variables cause less 
overrfitting or if it is better to rescale integer columns. Playing with 
compareing overfited data on models (Linear Regressoin, Logistic Regression,
Random Forest and K-Nearest Neighbor) with a set of 
data that will be both cross validated and train/test split.
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


#Reading in the data files created from Data_Clean_Analyze_SpaceshipTitanic
data_file_path = '/Users/lynnpowell/Documents/DS_Projects/Spaceship_Titanic/'
df_train = pd.read_csv(data_file_path+'Initial_Analysis_Train_SpaceTitanic.csv')
df_test = pd.read_csv(data_file_path+'Initial_Analysis_Test_SpaceTitanic.csv')


#Functions

#Function to calculate the cross validation score using negative mean
#absolute error. Adding 1 to the mean score to convert the number to positive.
#Returns a single accuracy value for that model.
def model_accuracy(model, df_train, target):
    train_X = df_train
    target_y = target
    scores = cross_val_score(model, train_X, target_y,
                             scoring = 'neg_mean_absolute_error',cv=10)
    accuracy = np.mean(scores) + 1
    return accuracy


#Function to test a data set on multiple models
#Linear Regression, Logistic Regression,
#Random Forest Classifier (Grid Search), K-Nearest Neighbor (Grid Search)
#Evaluating models with cross validation function (model_accuracy)
def ml_model_full(df_train, target, set_name):
    model_acc_list = [] 
    model_acc_index = []
    model_time = []
    
    #Linear Regression
    linr = LinearRegression()
    start_time = time.time()
    model_acc_list.append(model_accuracy(linr,df_train, target))
    model_acc_index.append(set_name+' Linear Regression')
    model_time.append(time.time() - start_time)
    
    #Logistic Regression
    logr = LogisticRegression()
    start_time = time.time()
    model_acc_list.append(model_accuracy(logr,df_train, target))
    model_acc_index.append(set_name+' Logistic Regression')
    model_time.append(time.time() - start_time)

    #Random Forest Classifier (grid search)
    #Hyperparameters found in Data Quest Machine Learning Models Lecture
    rfc_hyperpar = {
        'criterion' : ['entropy', 'gini'],
        'max_depth' : [5, 10],
        'max_features' : ['log2', 'sqrt'],
        'min_samples_leaf' : [1,5],
        'min_samples_split' : [3,5],
        'n_estimators' : [6,9]
    }
    rfc = RandomForestClassifier(random_state=7)
    start_time = time.time()
    rfc_grid = GridSearchCV(rfc, param_grid=rfc_hyperpar, cv=10)
    rfc_grid.fit(df_train, target)
    #rfc_best_params = grid.best_params_
    #rfc_best_score = grid.best_score_
    best_rfc = rfc_grid.best_estimator_
    model_acc_list.append(model_accuracy(best_rfc,df_train, target))
    model_acc_index.append(set_name+' Random Forest Classifier')
    model_time.append(time.time() - start_time)
    
    #K-Nearest Neighbors
    #Hyperparameters found in Data Quest Machine Learning Models Lecture
    knn_hyperpar = {
        "n_neighbors": range(1,20,2),
        "weights": ["distance", "uniform"],
        "algorithm": ['brute'],
        "p": [1,2]
    }
    knn = KNeighborsClassifier()
    start_time = time.time()
    knn_grid = GridSearchCV(knn, param_grid=knn_hyperpar, cv=10)
    knn_grid.fit(df_train, target)
    #knn_best_params = grid.best_params_
    #knn_best_score = grid.best_score_
    best_knn = knn_grid.best_estimator_
    model_acc_list.append(model_accuracy(best_knn,df_train, target))
    model_acc_index.append(set_name+' K Nearest Neighbors')
    model_time.append(time.time() - start_time)
    
    return model_acc_list, model_acc_index, model_time


#Function to test a data set on multiple models
#Linear Regression, Logistic Regression,
#Random Forest Classifier (Grid Search), K-Nearest Neighbor (Grid Search)
#Evaluating models with train test split, 70% to 20%
def ml_model_split(train, target, set_name):
    split_accuracy = [] 
    split_acc_index = []
    model_time = []
    
    train_X, test_X, train_y, test_y = train_test_split(
        train, target, test_size=0.2,random_state=7)
    
    #Linear Regression
    linr = LinearRegression()
    start_time = time.time()
    linr.fit(train_X,train_y)
    split_accuracy.append(linr.score(train_X, train_y))
    split_acc_index.append(set_name+' LinR Split Train')
    model_time.append(time.time() - start_time)
    split_accuracy.append(linr.score(test_X, test_y))
    split_acc_index.append(set_name+' LinR Split Test')
    model_time.append(time.time() - start_time)

    #Logistic Regression
    logr = LogisticRegression()
    start_time = time.time()
    logr.fit(train_X,train_y)
    #logr_predictions = logr.predict(test_X)
    #logr_mean_predic = mean_absolute_error(test_y, logr_predictions)
    split_accuracy.append(logr.score(train_X, train_y))
    split_acc_index.append(set_name+' LogR Split Train')
    model_time.append(time.time() - start_time)
    split_accuracy.append(logr.score(test_X, test_y))
    split_acc_index.append(set_name+' LogR Split Test')
    model_time.append(time.time() - start_time)


    #Random Forest Classifier (grid search)
    #Hyperparameters found in Data Quest Machine Learning Models Lecture
    rfc_hyper = {
        'criterion' : ['entropy', 'gini'],
        'max_depth' : [5, 10],
        'max_features' : ['log2', 'sqrt'],
        'min_samples_leaf' : [1,5],
        'min_samples_split' : [3,5],
        'n_estimators' : [6,9]
    }
    rfc = RandomForestClassifier(random_state=7)
    start_time = time.time()
    rfc_grid = GridSearchCV(rfc, param_grid=rfc_hyper, cv=10)
    rfc_grid.fit(train_X, train_y)
    #best_params = grid.best_params_
    #best_score = grid.best_score_
    #print(best_params)
    #print(best_score)
    best_rfc = rfc_grid.best_estimator_
    best_rfc.fit(train_X,train_y)
    #rfc_predictions = best_rfc.predict(test_X)
    #rfc_mean_predic = mean_absolute_error(test_y, rfc_predictions)
    rfc_score_train = best_rfc.score(train_X, train_y)
    rfc_score_test = best_rfc.score(test_X, test_y)
    split_accuracy.append(rfc_score_train)
    split_acc_index.append(set_name+' RFC Split Train')
    model_time.append(time.time() - start_time)
    split_accuracy.append(rfc_score_test)
    split_acc_index.append(set_name+' RFC Split Test')
    model_time.append(time.time() - start_time)

    #K-Nearest Neighbors
    #Hyperparameters found in Data Quest Machine Learning Models Lecture
    knn_hyper = {
        "n_neighbors": range(1,20,2),
        "weights": ["distance", "uniform"],
        "algorithm": ['brute'],
        "p": [1,2]
    }
    knn = KNeighborsClassifier()
    start_time = time.time()
    knn_grid = GridSearchCV(knn, param_grid=knn_hyper, cv=10)
    knn_grid.fit(train_X,train_y)
    #print(grid.best_params_)
    #print(grid.best_score_)
    best_knn = knn_grid.best_estimator_
    best_knn.fit(train_X,train_y)
    #knn_predictions = best_knn.predict(test_X)
    #knn_mean_predic = mean_absolute_error(test_y, knn_predictions)
    knn_score_train = best_knn.score(train_X, train_y)
    knn_score_test = best_knn.score(test_X, test_y)
    split_accuracy.append(knn_score_train)
    split_acc_index.append(set_name+' KNN Split Train')
    model_time.append(time.time() - start_time)
    split_accuracy.append(knn_score_test)
    split_acc_index.append(set_name+' KNN Split Test')
    model_time.append(time.time() - start_time)

    return split_accuracy, split_acc_index, model_time
    


#To keep track of the prediction accuracy with each feature and ML model
fs_accuracy = []
fs_index = []
fs_time = []


#Looking just at the spending features
#Bins, Rescale and unaltered columns
#Comparing Correlatin and Coefficient
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

#spend accuracy using logistic regression
logr = LogisticRegression()
start_time = time.time()
fs_accuracy.append(model_accuracy(logr,df_spend_original, df_train['Transported']))
fs_index.append('Feat Select Spend Orignial')
fs_time.append(time.time() - start_time)
#spend accuracy of bins using logistic regression
start_time = time.time()
fs_accuracy.append(model_accuracy(logr,df_spend_bins, df_train['Transported']))
fs_index.append('Feat Select Spend Bins')
fs_time.append(time.time() - start_time)
#spend accuracy of rescale using logistic regression
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

#Checking correlation of spending
spend_feat = spend_bins + ['Transported'] + rescale_col
df_spend = df_train[spend_feat]
#rescale train_X values
for col in rescale_col:
    df_spend[col] = minmax_scale(df_spend[col])
spend_cor = df_spend.corr()
target_spend_cor = spend_cor['Transported'].abs().sort_values(ascending=False)
target_spend_bin_cor = target_spend_cor.drop(labels=(rescale_col+['Transported']))
spend_bins_feat = target_spend_bin_cor[target_spend_bin_cor > 0.27].index.values.tolist()

#Spending feature list for ML models with high correlation and coefficient
spend_feat_rescale = spend_coeff_rescale[spend_coeff_rescale > 8].index.values.tolist()
spend_feat = np.unique(spend_coeff_bins + spend_bins_feat).tolist()

#Looking at age and families
family_list = ['ID_Group','Age','Age_Missing', 'Age_Infant', 'Age_Child','Age_Teenager',
               'Age_Young Adult', 'Age_Adult', 'Age_Senior', 'family_sum',
               'family_kid_num','family_small_kids', 'family_teen',
               'kids_without_adults', 'Cryo_Fam']

#age accuracy using logistic regression
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
family_list.append('Transported')

#Checking correlation of ages and families
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


#Analyzing remaining features (status): Destination, Cabin, Home planet and VIP
status_list = ['CryoSleep', 'VIP', 'HomePlanet_Earth',
       'HomePlanet_Europa', 'HomePlanet_Mars', 'HomePlanet_Missing',
       'Destination_55 Cancri e', 'Destination_Missing',
       'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e',
       'Cabin_Deck_A', 'Cabin_Deck_B', 'Cabin_Deck_C', 'Cabin_Deck_D',
       'Cabin_Deck_E','Cabin_Deck_F', 'Cabin_Deck_G', 'Cabin_Deck_Missing',
       'Cabin_Deck_T', 'Cabin_Side_Missing', 'Cabin_Side_P', 'Cabin_Side_S']

#status accuracy using logistic regression
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
rescale_feat = spend_feat_rescale + family_feat_rescale + ['Transported']
rescale_all = rescale_feat + status_feat
df_rescale = df_train[rescale_all]
for col in rescale_feat:
    df_rescale[col] = minmax_scale(df_rescale[col])

#compareing correlation of all features to 
#make sure none of them are to close to each other
df_rescale_cor = df_rescale.corr().abs()
#Cabin Side P and Cabin Side S have high correlation
rescale_drop = ['Cabin_Side_P', 'Cabin_Deck_T','VIP','Cabin_Deck_A']
df_rescale = df_rescale.drop(rescale_drop,axis=1)

#compareing correlation of all features to 
#make sure none of them are to close to each other
#Also checking to see if any other feature has high correlation to 'CryoSleep'
feat_cor_coeff = spend_feat + family_feat + status_feat + ['Transported']
df_cor_coeff = df_train[feat_cor_coeff]
full_feat_corr = df_cor_coeff.corr().abs()
cryo_corr = full_feat_corr['CryoSleep'].abs().sort_values(ascending=False)
cryo_corr = cryo_corr.drop(labels=('CryoSleep'))
high_cryo_corr = cryo_corr[cryo_corr > 0.53].index.values.tolist()


high_match_corr = high_cryo_corr+['Cabin_Side_P','Cabin_Deck_T','VIP','Cabin_Deck_A']
df_cor_coeff = df_cor_coeff.drop(high_match_corr, axis=1)


#Final Data Frames to run ML models on
df_rescale_train = df_rescale.drop('Transported', axis=1)
df_cor_coeff_train = df_cor_coeff.drop('Transported', axis=1)
#df_cor_coeff
#ML models on rescale data frame
acc_full_rescale = ml_model_full(df_rescale_train,df_rescale['Transported'],'Rescale')
fs_accuracy = fs_accuracy + acc_full_rescale[0]
fs_index = fs_index + acc_full_rescale[1]
fs_time = fs_time + acc_full_rescale[2]
#ML models on correlation and coefficeitn feature selection data frame
acc_corr_coeff = ml_model_full(df_cor_coeff_train,df_cor_coeff['Transported'],'Corr Coeff')
fs_accuracy = fs_accuracy + acc_corr_coeff[0]
fs_index = fs_index + acc_corr_coeff[1]
fs_time = fs_time + acc_corr_coeff[2]


#Using Split to evaluate on parts of the data frame the model has not seen
#Splits for ML models on rescale data frame
split_rescale = ml_model_split(df_rescale_train,df_rescale['Transported'],'Rescale')
fs_accuracy = fs_accuracy + split_rescale[0]
fs_index = fs_index + split_rescale[1]
fs_time = fs_time + split_rescale[2]

#Splits for ML models on correlation and coefficeitn feature selection data frame
split_corr_coeff = ml_model_split(df_cor_coeff_train,df_cor_coeff['Transported'],'Corr Coeff')
fs_accuracy = fs_accuracy + split_corr_coeff[0]
fs_index = fs_index + split_corr_coeff[1]
fs_time = fs_time + split_corr_coeff[2]



#Create data frame with all results
df_results = pd.DataFrame({'Accuracy Score':fs_accuracy,
                           'ML Model':fs_index,
                           'Time_Sec':fs_time}).sort_values(by=['Accuracy Score'], ascending=False)

#ML model with the higest accuracy was Random Forest with rescale features
test_feat = df_rescale_train.columns
df_ml_test = df_test[test_feat]


#Running Test Data Frame using ML model Random Forest Classifier
#Hyperparameters found in Data Quest Machine Learning Models Lecture
rfc_hyper = {
    'criterion' : ['entropy', 'gini'],
    'max_depth' : [5, 10],
    'max_features' : ['log2', 'sqrt'],
    'min_samples_leaf' : [1,5],
    'min_samples_split' : [3,5],
    'n_estimators' : [6,9]
}
rfc = RandomForestClassifier(random_state=7)
start_time = time.time()
rfc_grid = GridSearchCV(rfc, param_grid=rfc_hyper, cv=10)
rfc_grid.fit(df_rescale_train, df_rescale['Transported'])
best_rfc = rfc_grid.best_estimator_
best_rfc.fit(df_rescale_train, df_rescale['Transported'])
rfc_predictions = best_rfc.predict(df_ml_test)
rfc_grid.fit(df_ml_test, rfc_predictions)
best_rfc = rfc_grid.best_estimator_
best_rfc.fit(df_rescale_train, df_rescale['Transported'])
rfc_score = best_rfc.score(df_ml_test, rfc_predictions)
end_time = time.time() - start_time

#Accuracy was 0.833 with the test data
#I don't expect that high of accuracy when submitted to Kaggle

#Adding the test data frame accuracy result to the df_results
df_test_acc = {'Accuracy Score': rfc_score, 'ML Model': 'RFC Rescale Test', 'Time_Sec': end_time}
df_results = df_results.append(df_test_acc, ignore_index = True)
df_results = df_results.sort_values(by=['Accuracy Score'], ascending=False)

#Saving Data Frame with ML Model Accuracy Results
df_results.to_csv('Space_ML_Accuracy_Results.csv',index=False)

#Formating results for Kaggle Submission
test_passenger_ids = df_test['PassengerId']
submission = {'PassengerId': test_passenger_ids,
                 'Transported': rfc_predictions}
submission_df = pd.DataFrame(submission)
submission_df['Transported'] = submission_df['Transported'].astype('bool')
submission_df.to_csv('Space_Submission.csv',index=False)

#First Submission Transported was in 1/0 not True/False

#First real submitted file on 20May2022
#Accuracy Score From Kaggle: 0.68085
#Top Accuracy Score: 
#Rank From Kaggle: 1960 out of 2158 Submitted Teams
#2103 first non 0 accuracy score
#Top Ranked Accuracy Score: 0.83680
#This means my model was very overfitted

#Second Submittion on 01June2022
#Removed Deck A, T and VIP features from training data
#Also noticed my test file was not cleaned the same as my training file
#   during the clean and analize process.
#Accuracy Score From Kaggle: 0.74187
#Top Accuracy Score: 0.84428
#Rank From Kaggle: 1788 out of 2175 Submitted Teams
#2122 first non 0 accuracy score
#Top Ranked Accuracy Score: 0.83680

