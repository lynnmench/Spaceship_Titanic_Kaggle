#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lynnpowell

Date: 14Nov2022

"""

"""

Feature Selection Methods:
    - Uivariate Selection
    - SelectKBest Algorithm
    - Feature Importance
    - Pearson Correlation Coefficient
    - Information Gain



"""




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import time

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



#Reading in the data files created from Data_Clean_Analyze_SpaceshipTitanic
data_file_path = '/Users/lynnpowell/Documents/DS_Projects/Spaceship_Titanic/'
df_train = pd.read_csv(data_file_path+'EDA_FeatEngr_train.csv')
df_test = pd.read_csv(data_file_path+'EDA_FeatEngr_test.csv')


# --------------------- Feature Selection Methods ---------------------

#### Pearson Correlation ####
df_corr = df_train.corr()
target_corr = df_corr['Transported'].abs().sort_values(ascending=False)
target_bin_corr = target_corr.drop(labels=(['Transported']))
corr_feat = target_bin_corr[target_bin_corr > 0.1].index.values.tolist()
corr_feat_df = pd.DataFrame(data=corr_feat, columns=['Features'])
corr_order_feat = target_bin_corr.index.values.tolist()
corr_feat_full_df = pd.DataFrame(data=corr_order_feat, columns=['Features']).reset_index(drop=True)

#### Univariate Selection: ####

X = df_train.drop('Transported', axis=1)
y = df_train['Transported']


X_col = X.shape[1]

### Apply SelectKBest Algorithm
### Also refered to as information gain?
ordered_rank_features = SelectKBest(score_func=chi2, k=X_col)
ordered_feature = ordered_rank_features.fit(X,y)

univar_score = pd.DataFrame(ordered_feature.scores_, columns=['Score'])
univar_col = pd.DataFrame(X.columns)

univar_df = pd.concat([univar_col, univar_score], axis=1)
univar_df.columns=['Features','Score']

# For SelectKBest Algorithm the higher the score the higher the feature importance
univar_df['Score'].sort_values(ascending=False)
univar_df = univar_df.nlargest(50, 'Score').reset_index(drop=True)

##### Feature Importance: #####
#This method provides a score for each feature of your data frame
#The higher the score the more relevant the data is

model = ExtraTreesClassifier()
model.fit(X,y)

#print(model.feature_importances_)
feat_impotant = pd.Series(model.feature_importances_, index=X.columns)
feat_impotant.nlargest(20).plot(kind='barh')
feat_impot_df = feat_impotant.sort_values(ascending=False).to_frame().reset_index()
feat_impot_df.columns=['Features','Feat Import']


#### Information Gain ####
#Looking to see what highly correlated features are important to the final answer

mutual_info_values = mutual_info_classif(X,y)
mutual_info = pd.Series(mutual_info_values, index=X.columns)
mutual_info.sort_values(ascending=False)
mutual_info_df = mutual_info.sort_values(ascending=False).to_frame().reset_index()
mutual_info_df.columns=['Features','Mutual Info']


'''
# Comparing feature selection methods to see what columns come up the most often

feat_select = pd.concat([univar_df, feat_impot_df, corr_feat_full_df, mutual_info_df], axis=1, ignore_index=True)
feat1 = feat_select.iloc[:7,0].tolist()
#feat2 = feat_select.iloc[:7,2].tolist()
feat3 = feat_select.iloc[:7,4].dropna().tolist()
feat4 = feat_select.iloc[:7,5].tolist()
#top_features = feat1+feat2+feat3+feat4
top_features = feat1+feat3+feat4

top_feat_unique = []
for feat in top_features:
    if feat not in top_feat_unique:
        top_feat_unique.append(feat)
'''



#----------- Machine Learning Functions --------------------

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
        train, target, test_size=0.3,random_state=7)
    
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



# ------- Machine Learning Evaluations --------------


# Comparing feature selection methods to see what columns come up the most often

feat_select = pd.concat([univar_df, feat_impot_df, corr_feat_full_df, mutual_info_df], axis=1, ignore_index=True)
feat1 = feat_select.iloc[:8,0].tolist()
feat2 = feat_select.iloc[:6,2].tolist()
#feat3 = feat_select.iloc[:5,4].dropna().tolist()
feat3 = feat_select.iloc[:7,4].tolist()
feat4 = feat_select.iloc[:8,5].tolist()
#top_features = feat1+feat2+feat3+feat4
top_features = feat1+feat3+feat4

top_feat_unique = []
for feat in top_features:
    if feat not in top_feat_unique:
        top_feat_unique.append(feat)

my_guess_feat = ['CryoSleep','Age','Total_Spent','Cabin_Num','Ship_Deck','Ship_Side_S','HomePlanet_Earth']

#To keep track of the prediction accuracy with each feature and ML model
fs_accuracy = []
fs_index = []
fs_time = []

#Running models on top features selected using feature selection methods above
feat_1 = ml_model_split(df_train[my_guess_feat],df_train['Transported'],'Feat_Select_my')
fs_accuracy = fs_accuracy + feat_1[0]
fs_index = fs_index + feat_1[1]
fs_time = fs_time + feat_1[2]

#Running models on top features selected using feature selection methods above
feat_1 = ml_model_split(df_train[feat1],df_train['Transported'],'Feat_Select_1')
fs_accuracy = fs_accuracy + feat_1[0]
fs_index = fs_index + feat_1[1]
fs_time = fs_time + feat_1[2]

#Running models on top features selected using feature selection methods above
feat_2 = ml_model_split(df_train[feat2],df_train['Transported'],'Feat_Select_2')
fs_accuracy = fs_accuracy + feat_2[0]
fs_index = fs_index + feat_2[1]
fs_time = fs_time + feat_2[2]

#Running models on top features selected using feature selection methods above
feat_3 = ml_model_split(df_train[feat3],df_train['Transported'],'Feat_Select_3')
fs_accuracy = fs_accuracy + feat_3[0]
fs_index = fs_index + feat_3[1]
fs_time = fs_time + feat_3[2]

#Running models on top features selected using feature selection methods above
feat_4 = ml_model_split(df_train[feat4],df_train['Transported'],'Feat_Select_4')
fs_accuracy = fs_accuracy + feat_4[0]
fs_index = fs_index + feat_4[1]
fs_time = fs_time + feat_4[2]


#Running models on top features selected using feature selection methods above
full_lst = ml_model_split(df_train[top_feat_unique],df_train['Transported'],'Feat_Select_full')
fs_accuracy = fs_accuracy + full_lst[0]
fs_index = fs_index + full_lst[1]
fs_time = fs_time + full_lst[2]


#Create data frame with all results
df_results = pd.DataFrame({'Accuracy Score':fs_accuracy,
                           'ML Model':fs_index,
                           'Time_Sec':fs_time}).sort_values(by=['Accuracy Score'], ascending=False)

df_results = df_results.sort_values(by=['Accuracy Score'], ascending=False)

#df_results.to_csv(data_file_path+'Results_2_From_2nd_Attempt.csv',index=False)


