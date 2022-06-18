Author: Lynn Menchaca
Start Date: 15Apr2022

# Kaggle Spaceship Titanic
The purpose of this project is to predict which passengers were transported off the Spaceship Titanic after it collided with a spacetime anomaly hidden within a dust cloud. Kaggle provides the traning data file with personal information for about 2/3rd the passengers on board (about 8700 passengers), along with if they were transported or not. Kaggle aslo provides the test data file, to be used with your machine learning model to predict which of the remaining 1/3rd passengers were transported. The passenger ID and transported status from the test data file is submitted in the Kaggle competition.

#### -- Project Status: Active

## Project Overview
### Resources
Kaggle Competition: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview)

Data Quest: Data Science Lectures on Data Analysis, Visualization and Machine Learning Models

ReadMe Templet:
Github, RocioSNg, [Project-README-template.md](https://github.com/sfbrigade/data-science-wg/blob/master/dswg_project_resources/Project-README-template.md)

### Language/Platform/Libraries
* Python
* Jupyter
* Spyder
* Pandas
* numpy
* matplotlib
* sklearn

### Project Outline
* Inferential Statistics
* data exploration/descriptive statistics
* data processing/cleaning
* Data Visualization
* Feature Selection
* Machine Learning/Predictive Modeling
* writeup/reporting

(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

## Project Description

This project was greate for beginners becuase it required minimal data cleaning. To start with there were no columns with time stamps or string data that required excessive cleaning. The total number of rows with missing data was 2087 which was about 1/3rd the entire data. The total number of rows with missing data was 2087 which was about 1/3rd the entire data. Based off the data provided in the train file, I broke out the data in three sub categories family and age, money spent on the spaceship and passenger records. Below are my assumptions and hypothesis explored in this project.

1. What happend to passengers in suspended animation?
2. The families in suspended animation that had at least one group member awake, were they more likely to be transported?
3. Were children more likely to be transported?
4. Were families with children more likely to be transported?
5. Were passengers with VIP status more likely to be transported?
6. Is there an imbalance for passengers transported?
7. A assumption to fill in data for passengers, those that spent money were awake and passengers that were in cryo sleep spent no money.
8. A big assumption, passengers in groups had the same information for cabin, destination, home planet and same VIP status.

### Data Cleaning, Analysis and Visualization
I began with the passengers in suspended animation (CryoSleep) to see if each group had at least one awake group member. About 2/3rds of the CryoSleep groups had all members in CryoSleep. Analyzing the difference between the groups with all passengers in CryoSleep and those with at least one member awake, the difference did not significantly impact transported probablility.

To verify passenger that were in suspended animation spent no money, I summed the money spent at each location for CryoSleep passengers. Once the infomation was confirmed I filled in any null values at each spending location with $0 for passengers that were listed true for cryo sleep. Knowing this information I also set null cryo sleep values for passengers as false if any amount of money was spent at any location on the ship. There were passengers that were awake and spent $0, so I could not set passengers as true for CryoSleep just because they spent $0 at each location.

A big assumption I made with the data is passengers traveling together came from the same hometown, were staying on the same cabin floor and ship side, were traveling to the same destination and had the same VIP status. Given more time to work on the project I would perform an analysis on each of those three features to prove how accurate that assumption is. Using this assumption, I filled any null information for home planet, cabin, destination and VIP status for passengers in each group with what was listed for the other members. Based off the previous information I could not make the assumption if someone was in a group of cryo sleepers they were a cryo sleeper too.

Next, I analyzed the ‘Name’ column. Each row in the ‘Name’ column only had 2 words, I’m assuming a first name and a last name. I was looking to see if they had titles, rank, prefixes or suffixes listed with the names to provide a little more information about each passenger. Since this was not the case, I didn’t want to try to make any assumptions based of the name for gender or society status. With this information and since it had missing values I dropped the ‘Name’ column all together.

I wanted to see if there was any significant imbalance I had to take in to consideration of the passengers transported. To do this I looked at the percentage of passengers that were transported vs stayed. The result was 50.4% transported and 49.6% stayed. Since the values are so close I do not consider there to be any significant imbalance.

To make analyzing easier I broke the training data columns down in to three categories, families and age, money spent and passenger information.

#### Families and Age
For analyzing age of passengers, I wanted to see if children and families with children were more likely to survive. To do this I sorted the age column in to 7 different categorical bins. I then plotted passenger transported for each category. Looking at the age categorical data it looks like infants, children and teenagers were more likely to be transported(with infants and children having the greatest odds). To see if families with children were more likely to be transported I used the ID group information to group the passengers and count the different age categories in each group. This was also reviewed with multiple bar plots.


#### Money Spent at each Spaceship Location
The next group analyzed was money spent. Similar to analyzing age I split the amount spent at each location in to 3 different bins. When analyzing spending I only looked at passengers that were awake. This is because one of my bins was $0 and I didn’t want the Cryo Sleep passengers (who also had a spend of $0) to affect the analysis. I was surprised to see the passengers with high spending at the food court and the shopping mall where the only bins with a higher percentage of transportation. All other bins at each location had a higher stay on spaceship percentage.

#### Passenger Information
The final group analyzed was passenger information and status on board the Spaceship Titanic. The first step was to split the cabin information to two different columns for the deck and side of the ship the passenger was staying on. Next I filled in any remaining categorical data null values with “Missing”. Finally, I bar plotted the data for home planet, destination, cryo sleep, VIP and cabin deck and side. I was surprised that the VIP status for passengers had a higher percentage of passengers that stayed vs being transported.


To wrap up data cleaning and analysis I converted true/false/missing to 1/0/-1 for all boolean or integer columns. Any remaining categorical data I converted in to dummy variables of 1 and 0. I made sure there were no null values remaining in the data frame. I exported the cleaned test and training data frame to new csv files. Both of these files will be used with the machine learning models.

### Machine Learning
This was my first full data science project and I was interested in seeing if numeric values like age and spending are better for machine learning models in bins or rescaled. I was also interested in seeing how long each model ran for in relation to how accurate the results were.

To make the process easier and cleaner I created 3 functions: model_accuracy, ml_model_full and ml_model_split. The model_accuracy function is used to calculate the model accuracy using cross validation and negative mean absolute error. I took a mean of the accuracy scores and add 1 to make the value positive and easier to read. The ml_model_full function used 4 different ML models linear regression, logistic regression, random forest and k-nearest neighbors. The accuracy score for each ML model came from the function model_accuracy. The hyper parameters used for the random forest model and the k-nearest neighbors model I found in different Data Quest lectures. This function helped me run multiple ml models on multiple data frames without having to type everything out multiple times. The final function ml_model_split, was using the same ml models from the ml_model_full fuction, but with train test split analysis. My goal with using both cross validation and train test split was to try to see if I could catch my over fitted models before submitting my answers to Kaggle.

Similar to my data analysis I split my features out in to three categories: spending, age and passenger information. This made feature selection easier to analyze using Pearson correlation and coefficient. From each category I kept features that had high correlation to transported and low correlation to each other (to avoid over fitting). I also kept features with the higher coefficient values. I used the logistic regression model with the model_accuray function on each category. I was curious to see what had the largest impact on model accuracy before combining each feature in to one data frame for the final model accuracy. I split the feature selection for the spend and age category in to two different data frames dummy variables and rescale. I was curious to see if rescale was too much information for the model and if the bins and dummy variables were better to use.

After doing feature selection for each of the three categories I ended up with two full data frames, one with rescaled values and one with select dummy variables. Both data frames ran though the ml_model_full and ml_model_split functions. The random forest classifier was the most accurate model and ran just slightly more efficient then the k-nearest neighbors. 

I ran the test data file on the random forest model to predict which passengers were transported. The predictions come as true/false answers that needed to be converted to 1/0 for Kaggle submission. I exported the test results along with the results of the accuracy and execution time of each machine learning model to cvs files. For exporting the test file to submit to Kaggle I had to make sure the heading was correct as well, this meant setting the index as false.

### Results
The first submission had true/false as the results and received a score of 0.

The 2nd submission I received a score of 0.68085. This was so far off from my perditions, I did a little dive and noticed two mistakes. The first mistake was in analyzing the data, I did not perform the same data clean for the test file as I did for the train file. The second mistake was in the feature selection process, some features were not deleted in my final data frame used with the ML models. This caused my model to be overfitted and a lower accuracy score with the test data submitted to Kaggle. 

With everything corrected after my 2nd submission, for the 3rd submission I received a score of 0.74187 and a rank of 1788 out of 2175 submitted teams. This score was much closer to my predicted scores from models used with the training data. I still think the model is slightly overfitted. 

### Kaggle Score Improvement
Given more time I would analyze some of my assumptions made in the data cleaning stage for higher accuracy. I would use more complex methods for feature selection (wrapper methods or embedded methods) to prevent an over fitted model. Finally, I would continue to learn about and use more complex machine learning models to increase my overall accuracy.


## Featured Notebooks/Analysis/Deliverables
* [Data Cleaning and Analysis](https://github.com/lynnmench/Spaceship_Titanic_Kaggle/blob/291638a53e0128d5282fccb50447f6f22a28123c/Data_Clean_Analyze_SpaceshipTitanic.ipynb)
* [Machine Learning Feature Selection and Models](https://github.com/lynnmench/Spaceship_Titanic_Kaggle/blob/291638a53e0128d5282fccb50447f6f22a28123c/ML_SpaceshipTitanic.py)
