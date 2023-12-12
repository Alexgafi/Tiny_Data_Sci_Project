# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:22:54 2023

@author: adfw980
"""

#We will import the Python Libraries we are about to use

import csv

import pandas as pd

import matplotlib as plt

import matplotlib.pyplot as plt

import statsmodels. api as sm

import sys

import seaborn as sns

import numpy as np

from scipy.stats import probplot

from scipy import stats

# We will open the csv train file and look at the Data. 
# We will check and handle missing data, address outliers and scrutinize features

# We will open the train, test and samples_submission in different DataFrames

train = pd.read_csv ( 'C:/Users/adfw980/OneDrive - City, University of London/Desktop/DataSci_Project/house-prices-advanced-regression-techniques/train.csv' )

test = pd.read_csv( 'C:/Users/adfw980/OneDrive - City, University of London/Desktop/DataSci_Project/house-prices-advanced-regression-techniques/test.csv' )

submission = pd.read_csv ( 'C:/Users/adfw980/OneDrive - City, University of London/Desktop/DataSci_Project/house-prices-advanced-regression-techniques/sample_submission.csv')

#We will eliminate the ID column in both DataFrames so that it won't interfere with our analysis

train = train.drop("Id", axis = 1)

test = test.drop("Id", axis = 1)

#We will look at information about the train dataset

train.info()

train.head()

#We will now Describe the Target Variable

train['SalePrice'].describe()

#We will check if there are any missing values in the target data:
    
    train['SalePrice'].isnull().sum()
#The result is 0

#We will check the quantiles of SalesPrice

train['SalePrice'].quantile([0, 0.25, 0.50, 0.75, 0.99])

#Now we will explore the Data Distribution of the Target Value as well as the Distributions of the Features 

sns.boxplot (data = train, x='SalePrice')

# Observations: We can observe a considerable number of Outliers

plt.hist(train['SalePrice'])

# The data is also negatively Skewed

#Now we will try to apply the log function on the target data and see how it influences its distribution
#We will compare the distribution before and after transformation
#We will compare both histograms and Q-Q plots or probability plots

target = train['SalePrice']

target_log = np.log(target)

#We will make the histogram and Q-Q plot for the normal target data

#Pie chart

plt.pie(target,)

#Histogram
plt.hist(target)
plt.title('SalePrice')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

#Q-Q Plots

stats.probplot(target, plot=plt)
plt.title('Q-Q Plot')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('SalePrice Quantiles')
plt.show()





#We will now make the histogram and Q-Q plot for the log transformed Data

#Histogram
plt.hist(target_log, fit = stats.norm)
plt.title('SalePrice')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

#Q-Q Plots
stats.probplot(target_log, plot=plt)
plt.title('Q-Q Plot')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('SalePrice Quantiles')
plt.show()

#Observation: The log transformed target values are much closer to a 
#normal distribution as indicated by both QQ plots and Histogram

#We will merge the train and test data within a single Dataframe in order to operate with it

data = pd.concat([train, test], axis=0, ignore_index=True)


#Now, we will divide de columns from the DataFrame in Numerical, Categorical and Ordinal Columns

# We will firstly find the Numerical Columns and save them in a list

num_col = data.select_dtypes (include=[np.number]).columns.tolist()

#Now we will find the Catgorical Columns

cat_col = data.select_dtypes("object").columns.to_list()

#Now we will find the Ordinal Columns

n = data['YearBuilt'].nunique()

print(n)

ord_col = [col for col in num_col if data[col].nunique() < n 

#We chose year built as a unique valuye threshold because...

#We will now correct the number of numerical columns

num_col = [col for col in num_col if col not in ord_col]

print(f"Num Cols: {num_col}", end="\n\n")
print(f"Cat Cols: {cat_col}", end="\n\n")
print(f"Ordinal Cols: {ord_col}")

#We will now check if there are any duplicated rows

data.duplicated().sum()

#There are no duplicated rows

#We will now explore distribution of the Target Variable (SalesPrice)
#On different Categories from the Categorical Rows
#We will also compute the mean values of the target variable grouped by the categorical variables

#V1, which we tested
    
    for col in cat_col :
        plt.figure(figsize=(10,6))
        sns.barplot(x = col, y= train['SalePrice'], data = train, ci=None)
        plt.show()

#V2, still needs work

#target_mean_col = train.groupby('cat_col')['target'].mean().sort_values()

#for col in cat_col :
#    plt.figure(figsize=(10,6))
#   sns.barplot(x = col, y= target, data = train, ci=None, order = target_mean_col.index)
#  plt.show()

# Now, we will analyse the descriptive statistics, distribution of both normal and transformed values of the other variables 

#We will have to work here

for idx, column in enumerate (num_col):
    
    column.describe()
    
    sns.boxplot(y=column, data=train)
    plt.show()
    
    plt.hist(train[column], bins=10, edgecolor='black')
    plt.title(f'{column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
    
    
    stats.probplot(column, plot=plt)
    plt.title('Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('{column} Quantiles')
    plt.show()
    

#Now, we will repeat the process for transormed variables



#we will make correlations, heatmaps, and scatterplots between the target variable and the other variables



#We will check Homoscedasticity for important numerical variables before and after transformation


