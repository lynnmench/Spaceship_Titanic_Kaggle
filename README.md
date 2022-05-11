Author: Lynn Menchaca
Start Date: 15Apr2022

# Kaggle Spaceship Titanic
The purpose of this project is to predict which passengers were transported off the Spaceship Titanic after it collided with a spacetime anomaly hidden within a dust cloud. Kaggle provides the traning data file with personal information for about 2/3rd the passengers on board (about 8700 people), along with if they were transported or not. Kaggle aslo provides the test data file, to be used with your machine learning model to predict which of the remaining 1/3rd passengers were transported. The passenger ID and transported status from the test data file is submitted in the Kaggle competition.

#### -- Project Status: Active

## Project Overview
### Resources
Kaggle Competition: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview)

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


## Project Description

Based off the data provided in the train file, I broke out the data in three categories family and age, money spent on the spaceship and passenger records. With age and family I wanted to see if children and families with children were more likly to be transported. The other two big factors I was interested in, from the passenger records, was what happend to the passengers that had the VIP status and choose to travel on the spaceship in suspended animationi (CryoSleep).

(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

### Data Cleaning, Analysis and Visualization
This project was greate for beginners becuase it required minimal data cleaning. To start with there were no columns with time stamps or string data that required excessive cleaning. The total number of rows with missing data was 2087 which was about 1/3rd the entire data. If a passengers was in suspended animation this means they did not spend any money during their time on board the ship. Using that information, if a passenger was true for CryoSleep I filled missing data from the various spending locaitons with $0. This information filled in 301 missing rows. Since the null data was still a high percentage of rows, for the purpose of analyzing, the rest of the null data was later filled in as "Missing". In the data cleaning process I droped the "Name" column becuase it contained missing data and did not have any rank, prefix or sufix tied to the names. The "PassengerId" column was also very similar except it contained more data about the families on board the ship. 

1786

### 
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    *If using offline data mention that and how they may obtain the data from the froup)*
    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)
