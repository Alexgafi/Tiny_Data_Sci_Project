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

import statsmodels.api as sm

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

#We will look at information about the train dataset features and target variable

train.info()

train.head()

#We will now look at the descriptiv statistics of the target variable 'SalePrice'

train['SalePrice'].describe()

#We will check if there are any missing values in the target data:
    
    train['SalePrice'].isnull().sum()
#The result is 0, thus, there are no missing values in the target Data


#We will check the quantiles of SalesPrice

train['SalePrice'].quantile([0, 0.25, 0.50, 0.75, 0.99])

#Now we will explore the Data Distribution of the Target Value as well as the outliers of the Target Variable 'SalePrice'

sns.boxplot (data = train, x='SalePrice')

# Observations: We can observe a considerable number of Outliers

plt.hist(train['SalePrice'])

# The data is also positively Skewed

#Now we will try to apply the log transformation on the target data and see how it influences its distribution
#We will compare the distribution before and after transformation
#We will compare both histograms and Q-Q plots or probability plots before and after transformation

target = train['SalePrice']

target_log = np.log(target)

#We will make the histogram and Q-Q plot for the normal target data

#Pie chart

plt.pie(target)

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
plt.hist(target_log)
plt.title('SalePrice After Transformation')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

#Q-Q Plots
stats.probplot(target_log, plot=plt)
plt.title('Q-Q Plot After Transformation')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('SalePrice Quantiles')
plt.show()

#Observation: The log transformed target values are much closer to a 
#normal distribution as indicated by both QQ plots and Histogram
#Thus, for future regression analyses, we will use the log transformed Target Variable


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

ord_col = [col for col in num_col if data[col].nunique() < n ]

#To get the ordinal features, we counted the number of unique values in the ordinal column which has the biggest number of
#Different unique values, which is the YearBuilt. Thus, we assume that columns which have numbers of unique values below this number
#To be other ordinal variables

#We will now correct the number of numerical columns by extracting the no of ordinal columns from the total number of numerical columns

num_col = [col for col in num_col if col not in ord_col]

print(f"Num Cols: {num_col}", end="\n\n")
print(f"Cat Cols: {cat_col}", end="\n\n")
print(f"Ordinal Cols: {ord_col}")

#We will now check if there are any duplicated rows

data.duplicated().sum()

#There are no duplicated rows

#We will now explore distribution of the Target Variable (SalesPrice)
#On different Categories from the Categorical Rows


#V1, which we tested
    
    for col in cat_col :
        plt.figure(figsize=(10,6))
        sns.barplot(x = col, y= train['SalePrice'], data = train, ci=None)
        plt.show()



# Now, we will analyse the descriptive statistics, distribution of both normal and transformed values of the other variables 

#We will do it for the current numerical columns

for column in train:
    
    if column in num_col:
         
        train [column].describe()
         
        sns.boxplot(y=train[column], data=train)
        plt.title(f'{column} Distribution')
        plt.show()
             
        plt.hist(train[column], edgecolor='black')
        plt.title(f'{column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
             
        stats.probplot(train[column], plot=plt)
        plt.title('Q-Q Plot')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel(f'{column} Quantiles')
        plt.show()


    
print ("These are the descriptive statistics, boxplots, histograms and Q-Q Plots")
  

 # Few Features show normal distributions as indicated by both Histograms, QQ plots and Boxplots,
 # Moreover, a considerable number of outliers is observed. Thus, we will try the same process but with the log-transformed features
   
    

#Now, we will repeat the process for the log transormed variables

for column in train:
    
    if column in num_col:
         
        np.log(train[column]).describe()
         
        sns.boxplot(y=np.log(train[column]), data=train)
        plt.title(f'{column} Distribution')
        plt.show()
             
        plt.hist(np.log(train[column]), edgecolor='black')
        plt.title(f'{column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
             
        stats.probplot(np.log(train[column]), plot=plt)
        plt.title('Q-Q Plot')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel(f'{column} Quantiles')
        plt.show()

#Some features show distributions closer to normal ones as  result of the log transformation


#we will make correlations, heatmaps, and scatterplots between the target variable and the other numeric variables

#We will make a heatmap with the correlations of the numerical Variables in the train DataFrame
#With both numerical and ordinal columns and the target variable

n_col = train.select_dtypes(include='number')

    
    

#Heatmap of all the numeric Variables
plt.figure(figsize = (15, 15))
sns.heatmap(n_col.corr())


#Correlations between numerical columns and target value
#Give reference: https://www.kaggle.com/code/ericliu365/house-price-common-models-stacking-voting-and-nn

plt.figure(figsize=(15, 8))

n_col.corr()['SalePrice'].sort_values(ascending = False).drop('SalePrice').plot(kind = 'bar')

plt.suptitle('Correlation between numerical features and Sales Price', fontsize=16)



#We will check the most important 10 features based on correlation for the Target variable
#Give reference: https://www.kaggle.com/code/ericliu365/house-price-common-models-stacking-voting-and-nn


top_feat_num_corr = n_col.corr()['SalePrice'].drop('SalePrice').sort_values(ascending = False).head(10)

print(top_feat_num_corr)

#Top 10 most correlated features with the target variable, which are also likely to be important for the target variable are:

# OverallQual     0.790982, GrLivArea       0.708624, GarageCars      0.640409; GarageArea      0.623431
#TotalBsmtSF     0.613581, #1stFlrSF        0.605852, #FullBath        0.560664, TotRmsAbvGrd    0.533723
#YearBuilt       0.522897, YearRemodAdd    0.507101    

#We will check Homoscedasticity for important numerical variables before and after transformation
#We will make scatterplots with linear regression fit models between numerical columns and the target variable
#Give Reference: https://www.kaggle.com/code/ericliu365/house-price-common-models-stacking-voting-and-nn

for column in train:
    if column in train.select_dtypes (include=[np.number]).columns:
        sns.regplot(x=column, y= train['SalePrice'], data=train, scatter_kws={'s':10})
        plt.title(f'Regression Plot for {column} and SalePrice')
        plt.show()

#Now, we will do the same for the log Transformed Values

for column in train:
    if column in train.select_dtypes (include=[np.number]).columns:
        sns.regplot(x=np.log(train[column]), y= np.log(train['SalePrice']), data=train, scatter_kws={'s':10})
        plt.title(f'Regression Plot for {column} and SalePrice After Log Transformation')
        plt.show()


#We will now do it for when only the target variable is transformed:
    
  
    for column in train:
        if column in train.select_dtypes (include=[np.number]).columns:
            sns.regplot(x= train[column], y= np.log(train['SalePrice']), data=train, scatter_kws={'s':10})
            plt.title(f'Regression Plot for {column} and SalePrice after Log Transformation')
            plt.show()

    

# It seems like the log transfomration has improved scatterplots distribution and visualization, which is consistent with the normality distributions assumptions
#Thus, we might use log transformaion for future regression analyses
    
# FEATURE ENGINEERING AND DATA WRAGGLING

#Firstly, we will check for Outliers in our data and see what variables have the most outliers

for column in train.select_dtypes (include=[np.number]).columns:
    Q1 = train[column].quantile(0.25)
    Q3 = train[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train[(train[column] < lower_bound) | (train[column] > upper_bound)]
    percentage_outliers = (len(outliers) / len(train[column])) * 100
    
    print (f"Column:{column}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Percentage of outliers: {percentage_outliers: .2f}%")
    print("Outliers:")
    print(outliers)
    print("\n")
    
#There is a considerable number of features witth outliers. We will remove the outliers so that they don't interfere with our regression analysis
 

#Remove the ourliers in the train dataframe, both target and the other numerical variables

for column in train:
    if column in train.select_dtypes (include=[np.number]).columns:
        Q1_train = train[column].quantile(0.25)
        Q3_train = train[column].quantile(0.75)
        IQR_train = Q3_train - Q1_train
        
        lower_bound_train = Q1_train - 1.5 * IQR_train
        upper_bound_train = Q3_train + 1.5 * IQR_train
        
        train_noot = train[(train[column] >= lower_bound_train) & (train[column] <= upper_bound_train ) ]

 train = train_noot
 
 #We will now eliminate the outliers in test as well

 for column in test:
     if column in test.select_dtypes (include=[np.number]).columns:
         Q1_test = test[column].quantile(0.25)
         Q3_test = test[column].quantile(0.75)
         IQR_test = Q3_test - Q1_test
         
         lower_bound_test = Q1_test - 1.5 * IQR_test
         upper_bound_test = Q3_test + 1.5 * IQR_test
         test_noot = test[(test[column] >= lower_bound_test) & (test[column] <= upper_bound_test ) ]

 test = test_noot
 
 
 
#We will look at the features which have the most missing values and are correlated the least with the target Variables and 
#We will consider eliminating them from the analysis as they most likely represent noise


missing_values = train.isnull().sum()
print(missing_values)

#We decided to Drop the following features: Alley, PoolArea(none of the remaining houses had a pool),
#MiscFeature, Fence, FireplaceQu. These are feature which have considerable missing values and are poorly correlated with our target variable 

train = train.drop(['Alley', 'PoolArea', 'MiscFeature', 'Fence', 'FireplaceQu', 'Street', 'Utilities','Condition2', 'RoofMatl', 'Heating','PoolQC'], axis=1)

#We will now eliminate the same columns in test

test = test.drop(['Alley', 'PoolArea', 'MiscFeature', 'Fence', 'FireplaceQu','Street', 'Utilities','Condition2', 'RoofMatl', 'Heating', 'PoolQC' ], axis=1)

#We will eliminate variables that have a dominant variable that covers a high percentage of the datatset(over 95%)
#As they don't explain variation in our analysis, but rather represent noise

for column in train:
    if (train[column].value_counts().sort_values(ascending=False)/ len(train[column]) * 100).iloc[0] > 95:
       train = train.drop([column], axis=1)
       print(column)
        
     

#Now we will eliminate the same variables from the test

test = test.drop(['LowQualFinSF','KitchenAbvGr','3SsnPorch', 'MiscVal'], axis = 1)

#We will first impute the variables for the categorical featrures
#firstly, we will do that for train

train.loc[train["MasVnrArea"].isnull(), ["MasVnrArea"]] = 0
train.loc[train["BsmtFinSF1"].isnull(),["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF"]] = 0
train.loc[train["GarageYrBlt"].isnull(), "GarageYrBlt"] = 0

#We set the variables above to 0 as we assumed that the missing values represents the lack of the variable indicated
# For example in: "MasVnrArea", we assumed that the missing values represent the absence of masonry veneer
#Same for Basement related columns and Garage Areas

for column in train:
    train.loc[train["GarageArea"].isnull(),"GarageArea"] = 0
    
    if train[column].dtype =='object':
        train[column].fillna("None")


cat_del = ['Street', 'Alley', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

for element in cat_del:
    if element in cat_col:
        cat_col.remove(element) 



ord_del = ['LowQualFinSF', 'KitchenAbvGr', '3SsnPorch', 'PoolArea', 'MiscVal']

for element in ord_del:
    if element in ord_col:
        ord_col.remove(element)

train[ord_col] = train[ord_col].fillna(0)

#We then filled the categorical and ordinal missing values with either '0' or 'None, as there remained few such features, which were not strongly correlated to the target variable


#Now for test

test.loc[test["MasVnrArea"].isnull(), ["MasVnrArea"]] = 0
test.loc[test["BsmtFinSF1"].isnull(),["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF"]] = 0
test.loc[test["GarageYrBlt"].isnull(), "GarageYrBlt"] = 0
test.loc[test["GarageArea"].isnull(),"GarageArea"] = 0

for column in test:
    if test[column].dtype =='object':
        test[column].fillna("None")


test[ord_col] = test[ord_col].fillna(0)

#Here, we did the same for the test



#In the ordinal Columns, we assume that the missing values correspond to the lowest ordinal value
#In the categorical Values, we fill the missing values with the string 'None;

#We will Impute those with some missing variables but high correlation with the target Variable:
#The method chosen for Imputation is MICE (Multiple Imputation by Chained Equations) algorithm
#Which imputes the missing values considering the relationships between related variables variables
#We will make a function which does that and then we will choose the feature on which we will perform the imputation
#Give reference:  https://www.kaggle.com/code/dumanmesut/house-prices-prediction-cat-lgbm-xgb

def mice_imput(data:pd.DataFrame, fill:str, based:list) -> pd.Series :

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    
    
    categoric_cols = [col for col in based if data[col].dtype == "0"]
    
    temp_df = pd.get_dummies(data[[fill] + based].copy(), columns = categoric_cols)
    
    missing_mask = temp_df.isna()
    
    imputer = IterativeImputer(max_iter=10, random_state=42)
    
    imputed_values = imputer.fit_transform(temp_df)
    
    temp_df[fill][temp_df[fill].isnull()] = imputed_values[missing_mask]
    
    return temp_df[fill]




#Now, we will impute the variables in our chosen features with the MICE method
#We chose the 'LotFrontage' feature as it is highly correlayted with the target variable and has numerous missing values
#We will impute this feature with a numerical Variable which is directly correlated to it: 'LotArea'

train["LotFrontage"] = mice_imput(train, fill="LotFrontage", based=["LotArea"])
    
test["LotFrontage"] = mice_imput(test, fill="LotFrontage", based=["LotArea"])
    

#We will now drop the categorical columns with a dominant value that covers a high percentage of the Dataset
#Because this does not helpo us explain variation, but rather represent noise


for column in train.select_dtypes("object").columns:
    if (train[column].value_counts().sort_values(ascending=False) / len(train[column])*100).iloc[0] > 95: 
        train = train.drop(column, axis=1)
        print(column)
        
#We dropped the following features: Street, Utilities, Condition2, RoofMat1, Heating

#We will drop the same from the test



#We will add some nerw features that we consider relevant for the analysis based on the ones we have now:
#The following features can be compounded into larger features which are relevant for the target variable, thus, making new features can aid us explain more variance
    
data['Total Area'] = data['TotalBsmtSF'] + data['GrLivArea']
    
 data['TotalBathrooms'] = data['FullBath'] + data['HalfBath']*0.5 + data['BsmtHalfBath']*0.5 + data['BsmtFullBath']
    
    # Calculate the total room count
    data['TotalRooms'] = data['BedroomAbvGr'] + data['TotRmsAbvGrd']
    

    # Calculate the total porch area
    data['TotalPorchArea'] = data['OpenPorchSF'] + data['EnclosedPorch'] + \
                            data["3SsnPorch"] + data["ScreenPorch"] + data["WoodDeckSF"]
    
    # Has Garage?
    data['HasGarage'] = [1 if gar > 0 else 0 for gar in data["GarageYrBlt"]]
    
    # House Overall
    data['Overal'] = data['OverallQual'] + data['OverallCond']

#We will now encode the categorical columns in the DataFrame with Binary values using the get_dummies function

data = pd.get_dummies(data, columns=cat_col, dtype=int)

# We will do the same for train

train['Total Area'] = train['TotalBsmtSF'] + train['GrLivArea']
    
train['TotalBathrooms'] = train['FullBath'] + train['HalfBath']*0.5 + train['BsmtHalfBath']*0.5 + train['BsmtFullBath']
    
    # Calculate the total room count
    train['TotalRooms'] = train['BedroomAbvGr'] + train['TotRmsAbvGrd']
    

    # Calculate the total porch area
    train['TotalPorchArea'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF']
    
    # Has Garage?
    train['HasGarage'] = [1 if gar > 0 else 0 for gar in train["GarageYrBlt"]]
    
    # House Overall
    train['Overal'] = train['OverallQual'] + train['OverallCond']


train = pd.get_dummies(train, columns=cat_col, dtype=int)

#Now for test

test['Total Area'] = test['TotalBsmtSF'] + test['GrLivArea']
    
test['TotalBathrooms'] = test['FullBath'] + test['HalfBath']*0.5 + test['BsmtHalfBath']*0.5 + test['BsmtFullBath']
    
    # Calculate the total room count
    test['TotalRooms'] = test['BedroomAbvGr'] + test['TotRmsAbvGrd']
    

    # Calculate the total porch area
    test['TotalPorchArea'] = test['OpenPorchSF'] + test['EnclosedPorch']  + test["ScreenPorch"] + test["WoodDeckSF"]
    
    # Has Garage?
    test['HasGarage'] = [1 if gar > 0 else 0 for gar in test["GarageYrBlt"]]
    
    # House Overall
    test['Overal'] = test['OverallQual'] + test['OverallCond']


test = pd.get_dummies(test, columns=cat_col, dtype=int)




                            # MODELING #
                            
# Firstly, we will split the train data into train and test for the models 
#We will also use log transformation of our target value as it showed to improve the regression assumptions for our Dataset
#We will use 80% of the data for training and 20% for testing
#The Tests conducted will be lgbm, xgboost, Catboost, KNN, SVR. We chose these regressors as they are appropiate for regressions with features that include both categorical and numerical features



for column in train: 
    if train[column].dtype == 'object':
        train[column] = train[column].astype('category')

for column in test: 
    if test[column].dtype == 'object':
        test[column] = test[column].astype('category')

#We transformed the object datatypes in 'category' as the tests require us to do so

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 

X = train.drop(["SalePrice"], axis=1)
Y = np.log(train["SalePrice"])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1000)




# We will firstly test the importance of features using Light Gradient Boosting Machine (LGBM) as our data contains numerical, categorical and ordinal vectors
#Give reference: https://www.kaggle.com/code/dumanmesut/house-prices-prediction-cat-lgbm-xgb

pip install lightgbm

pip install --upgrade lightgbm
pip install --upgrade pandas

#///          ///
import lightgbm as lgb

from lightgbm import LGBMRegressor

lgb = lightgbm.LGBMRegressor(objective = 'root_mean_squared_error')
lgb.fit(X_train, Y_train)
lightgbm.plot_importance(lgb, max_num_features = 15);
y_pred = lgb.predict(X_test)
mean_squared_error(Y_test, Y_pred, squared=False)

#We will analyze the importance of the features using an extreme gradient boosting (xgboost) 
#Give reference: https://www.kaggle.com/code/dumanmesut/house-prices-prediction-cat-lgbm-xgb
    


import xgboost

xgb = xgboost.XGBRegressor(objective = 'reg:squarederror')
xgb.fit(X_train, Y_train)
xgboost.plot_importance(xgb, max_num_features = 15);
Y_pred = xgb.predict(X_test)
mean_squared_error(Y_test,Y_pred, squared=False)

#We will analyze the importance of features doing Catboost


from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

regressor = CatBoostRegressor(loss_function='RMSE')
regressor.fit(X_train, Y_train)
importances = regressor.get_feature_importance()
Y_pred = regressor.predict(X_test)
mean_squared_error(Y_test, Y_pred, squared=False)
print(mean_squared_error)

#Now we will draw the bar plot which indicates the top 15 features based on importance for catboost


feature_names = X.columns

sorted_importances = sorted(range(len(importances)), key=lambda k:importances[k], reverse=True)

top_features_names = [feature_names[i] for i in sorted_importances[:15]]

top_feature_importances = [importances[i] for i in sorted_importances[:15]]

plt.barh(top_features_names, top_feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title(f'Top 15 Features by Importance')
plt.show()

print(top_features_names)

#We will now select the first 15 features to use based on the xgboost and catboost importance features
#We will use the standardization method MinMAx Scaler as XGboost and Catboost have different importance tests values

from sklearn.preprocessing import MinMaxScaler

xgb_importances = pd.DataFrame(dict(xgb = xgb.feature_importances_), index=xgb.feature_names_in_)

catb_importances = pd.DataFrame(dict( catb = importances),  index=feature_names )

final_importances = pd.concat([xgb_importances, catb_importances], axis=1)
min_max = MinMaxScaler((0.1,1))

final_importances["cross"] = min_max.fit_transform(final_importances[["xgb"]]) * min_max.fit_transform(final_importances[["catb"]])

sorted = final_importances.sort_values(by="cross", ascending=False).reset_index()

sorted = pd.DataFrame(sorted)

#We made a Dataframe with the most important features arranged in a descending manner. The 15 most important features based on the standardized test are:

# 	Total Area, OverallQual, CentralAir_N, Overal, KitchenQual_TA, TotalBathrooms, GarageCars, Fireplaces, YearBuilt, GrLivArea, MSZoning_RM, GarageArea, GarageYrBlt, YearRemodAdd, LotArea, TotalBsmtSF


#Checking the cross values, we decided to stick with the first 150 features, thus, eliminate the last 94 features.

X_train.drop(sorted.tail(94)["index"], axis=1, inplace=True)
X_test.drop(sorted.tail(94)["index"],axis=1, inplace=True)
test.drop(sorted.tail(94)["index"],axis=1, inplace=True)


#REGRESSOR Tests

#We will conduct XGBoosting, LGB,Catboost, KNN, SVR, 

   #CatBoost Regressor
   #We will use optuna to test hyperparameters ranges in order to adjust the right fit ones for our test
   #Give Reference: https://www.kaggle.com/code/dumanmesut/house-prices-prediction-cat-lgbm-xgb
   #Using optuna, we will define a function which suggests the hyperparameters' range that optuna tests and optimizes
   #we will train the model for the parameters range, then obtain the best parameters as indicated by the root mean squared errors
   #We wil then run the model using the best parameters as indicated by optuna. 
   #This method allows us to efficiently optimize hyperparameters without runing the models multiple times
   #We will apply the same method for all the models we run
   #We might change our approach for SVR as they are slower and require more computing power, for them, we might use predetermined parameters
   
from catboost import CatBoostRegressor
import optuna

def objective_cat(trial):
    
    params = {
        
        'objective' : trial.suggest_categorical('objective', ['RMSE']),
        
        'logging_level' : trial.suggest_categorical('logging_level', ['Silent']),
        
        "random_seed" : trial.suggest_categorical('random_seed', [42]),
        
        "iterations" : trial.suggest_int("iterations", 500, 1500),
        
        'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.15),
        
        "depth" : trial.suggest_int("depth", 5, 8),
        
        'subsample': trial.suggest_float("subsample", 0.6, 0.9),
        
        "colsample_bylevel" : trial.suggest_float("colsample_bylevel", 0.1, 0.5),
        
        'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 5, 20),
        
        'bagging_temperature' : trial.suggest_loguniform('bagging_temperature', 0.01, 1),
        
        'leaf_estimation_iterations' : trial.suggest_int('leaf_estimation_iterations',10,30),
        
        'reg_lambda': trial.suggest_uniform('reg_lambda',50,100),
        
        }
        
    model_cat = CatBoostRegressor(**params)
    model_cat.fit(X_train, Y_train)
    Y_pred = model_cat.predict(X_test)
    return mean_squared_error(Y_test, Y_pred, squared=False)
        
    
  study_cat = optuna.create_study(direction='minimize')
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  study_cat.optimize(objective_cat, n_trials=50,show_progress_bar=True)
       
   print('Best parameters', study_cat.best_params)
 
    #Now, we will run the model using the best parameters
    
 cat = CatBoostRegressor(**study_cat.best_params)
cat.fit(X_train, Y_train)
y_pred = cat.predict(X_test)

print('Error: ', mean_squared_error(Y_test, Y_pred, squared=False))
 
cat_mse = mean_squared_error(Y_test, Y_pred, squared=False)


#The mean squared error of our model is Error:  0.12031410369339221

   
   #XGBoost Regressor
   
   from xgboost import XGBRegressor
   import optuna
   
   def objective_xg(trial):
       
       params = {
           
           'booster' : trial.suggest_categorical('booster', ['gbtree']),
           'max_depth' : trial.suggest_int('max_depth', 3, 12),
           'max_leaves' : trial.suggest_int('max_leaves', 8, 1024),
           'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
           'n_estimators' : trial.suggest_int('n_estimators', 400, 1500),
           'min_child_weight': trial.suggest_int('min_child_weight', 10, 20),
           'subsample' : trial.suggest_float('subsample', 0.3, 0.9),
           'reg_alpha' : trial.suggest_float('reg_alpha', 0.01, 0.5),
           'reg_lambda' : trial.suggest_float('reg_lambda', 0.5, 1.0),
           'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.3, 0.8),
           'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1),
           'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.01, 0.5),
           "random_state" : trial.suggest_categorical('random_state', [42]),
           'objective' : trial.suggest_categorical('objective' , ['reg:squarederror']),
           "n_jobs" : trial.suggest_categorical('n_jobs', [-1]),
                                                                    
           
       }
        
       model_xgb = XGBRegressor(**params)
       model_xgb.fit(X_train, Y_train)
       Y_pred = model_xgb.predict(X_test)
       return mean_squared_error(Y_test, Y_pred, squared=False)
       
        
       study_xgb = optuna.create_study(direction='minimize')
       optuna.logging.set_verbosity(optuna.logging.WARNING)
       study_xgb.optimize(objective_xg, n_trials=50, show_progress_bar=True)
       
   print('Best parameters', study_xgb.best_params)
   
   
   xgb = XGBRegressor(**study_xgb.best_params)
   xgb.fit(X_train, Y_train)
   Y_pred = xgb.predict(X_test)
   print('Error:', mean_squared_error(Y_test, Y_pred, squared=False))
   
  xgb _mse = mean_squared_error(Y_test, Y_pred, squared=False)
   
#The mean squared error for the xgb regressor is: Error: 0.11867088180370258     
                                            
   #LightGBM Regressor
   
   
   from lightgbm import LGBMRegressor
import optuna

def objective_lgb(trial):
    """Define the objective function"""

    params = {
        'objective': trial.suggest_categorical('objective', ['root_mean_squared_error']),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 8, 1024),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 700, 1600),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 25),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.0),
        "random_state" : trial.suggest_categorical('random_state', [42]),
        "extra_trees" : trial.suggest_categorical('extra_trees', [True]),
        
    }


    model_lgb = LGBMRegressor(**params)
    model_lgb.fit(X_train, y_train)
    y_pred = model_lgb.predict(X_test)
    return mean_squared_error(y_test,y_pred, squared=False)
   
    study_lgb = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lgb.optimize(objective_lgb, n_trials=50,show_progress_bar=True)
   

     print('Best parameters', study_lgb.best_params)
     
     lgb = LGBMRegressor(**study_lgb.best_params)
lgb.fit(X_train, y_train)
y_pred = lgb.predict(X_test)

print('Error: ', mean_squared_error(y_test,y_pred, squared=False))


   #Support Vector Regression(SVR)
   
#The SVR took too long due to the high computing power requirements, thus, we set smaller range parameters for fewer parameters after we experienced several crashes
   

   from sklearn.svm import SVR
   import optuna
   
   def objective_svr(trial):
       
       params = {
           
           
           'C' : trial.suggest_loguniform('C', 0.5, 1.5),
           
           
        }
       
       model_svr = SVR(**params)
       model_svr.fit(X_train, Y_train)
       Y_pred = model_svr.predict(X_test)
       return mean_squared_error(Y_test, Y_pred, squared=False)
    
    study_svr = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_svr.optimize(objective_svr, n_trials=50, show_progress_bar=True)
    
   print('Best parameters', study_svr.best_params)
   
   svr = SVR(**study_svr.best_params)
   svr.fit(X_train, Y_train)
   Y_pred = svr.predict(X_test)
   print('Error:', mean_squared_error(Y_test, Y_pred, squared=False))
           
   svr_mse = mean_squared_error(Y_test, Y_pred, squared=False)
         
   #SVR mean squared error = Error: 0.19482051252137766
    
  # Predefined 



   
   #K-Nearest Neighbors, KNN
   
   from sklearn.neighbors import KNeighborsRegressor
   import optuna
   
   def objective_knn(trial):
       
       params = {
           
          'n_neighbors' : trial.suggest_int('n_neighbors', 1, 100),
          'weights' : trial.suggest_categorical('weights', ['uniform', 'distance']),
          'algorithm' : trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
          'leaf_size' : trial.suggest_int('leaf_size', 10, 50),
          'p' : trial.suggest_int('p', 1, 2),
     
           }
       
       model_knn = KNeighborsRegressor(**params)
       model_knn.fit(X_train, Y_train)
       Y_pred = model_knn.predict(X_test)
       return mean_squared_error(Y_test, Y_pred, squared=False)
   
    study_knn = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_knn.optimize(objective_knn, n_trials=50, show_progress_bar=True)
    
    print('Best parameters', study_knn.best_params)
    
    knn = KNeighborsRegressor(**study_knn.best_params)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    print('Error:', mean_squared_error(Y_test, Y_pred, squared=False))
       
      #The error for the KNN model is: Error: 0.19179481080819671
   knn_mse = mean_squared_error(Y_test, Y_pred, squared=False)
   
   #Voting Regressor: model that combines multiple regression models to make predicitons. It takes a weighted average of the predicitons made which leads to more robust and accurate predictions
   #We will make the voting regressor on the cat, xgb and knn models for now.
#Give Reference: https://www.kaggle.com/code/dumanmesut/house-prices-prediction-cat-lgbm-xgb

!pip install sklego

from sklego.linear_model import LADRegression
models = pd.DataFrame()
models["cat"] = cat.predict(X_test) 
models["svr"] = svr.predict(X_test)  
models["xgb"] = xgb.predict(X_test)
models["knn"] = knn.predict(X_test)
weights = LADRegression().fit(models, Y_test).coef_
pd.DataFrame(weights, index = models.columns, columns = ["weights"])
    
#this regression fit automatically distributes the weights on each model based on the variance they account for


from sklearn.ensemble import VotingRegressor
voting = VotingRegressor(estimators=[ ('cat', cat),
                                      ('svr', svr),
                                      ('xgb', xgb),
                                      
                                      ('knn', knn)], weights=weights)

voting.fit(X_train,Y_train)
voting_pred = voting.predict(X_test)

print('Error: ', mean_squared_error(Y_test,Y_pred, squared=False))

vot_mse = mean_squared_error(Y_test,Y_pred, squared=False)

#Error:  0.19482051252137766



    #Model Performance

#We will make a plot to compare the model performance of each model based on the mean squared erro value

performance = pd.DataFrame({ 
    
    'Model': ['cat_mse', 'xgb_mse', 'svr_mse', 'knn_mse', 'vot_mse'],
    'Value' : [cat_mse, xgb_mse, svr_mse, knn_mse, vot_mse]
    })

plt.figure(figsize=(8,5))
plt.bar(performance['Model'], performance['Value'])
plt.title('Errors of the models')
plt.xlabel('Model')
plt.ylabel('Value')
plt.show()


    #Prediction
    
    
    
    #For Voting Regressor 
    
    sub["SalePrice"]=np.exp(voting.predict(test))
sub.to_csv('submission.csv',index=False)
sub
    
    #For Catboost
    
    #For XGBoost
    
    #For SVR
    
    #For KNN
    
    