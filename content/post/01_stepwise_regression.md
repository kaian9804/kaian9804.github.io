---
layout:     post
title:      "Feature Engineering with Stepwise Regression"
subtitle:   "Coding about finding the importances of features"
date:       2023-10-08T21:16:00+08:00
author:     "Ka Ian"
#image:      "https://img.zhaohuabing.com/post-bg-2015.jpg"
draft:      true
tags:
  - "statistics"
  - "coding"
  - "python"
---

> When I was studying statistics, I noticed an algorithm called "stepwise regression". Generally, this is a method that ranks  features based on their importances and finds out how important each feature is to the prediction.

The following coding will show the process of finding the features importances. (Assumed that this is a binary classification)

## Import the tools
```
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
 
import warnings
warnings.filterwarnings('ignore')
```

## Get Data
Need to select / drop out those unnecessary features.
```
df = pd.read_csv('df.csv')
df = df.drop(['fea1', 'fea3', 'fea4'], axis=1)
```

## Define the type of features
To see if these features belong to CATEGORY or NUMERIC. In this part, only the x features need to be defined, the label is not needed here.
```
CATEGORY = ['fea5', 'fea6', 'fea7']
NUMERIC = ['fea2', 'fea8', 'fea9', 'fea10']
```

## Data Preprocessing
Use label encoder to CATEGORY data. This is because using a one-hot encoder may result in finding useful features that are all derived from the same original feature, suck kind of result is not matched what I want in finding the important features.

Use the same scale to scale the features. The common methods are Min-Max Scaler, Standard Scaler, etc.

```
def preprocessing(df):
  # label encoder
  labelencoder = LabelEncoder()
  for c in CATEGORY:
    df[c] = labelencoder.fit_transform(df[c])
  
  # MinMaxScaler
  scaler = MinMaxScaler()
  df[NUMERIC] = scaler.fit_transform(df[NUMERIC])
 
  return df
```

## Predict the Probability and Setting Threshold

Since this coding is applied in binary classification and the prediction resulted in propability, it is needed to changed the result into the binary value. The threshold is setted as 0.5.

```
def y_pred_to_binary(y_pred):
  y_pred_bin_list = []
  
  for i in y_pred:
    if i >= 0.5:
      y_pred_bin = 1
    else:
      y_pred_bin = 0
    y_pred_bin_list.append(y_pred_bin)
    
  return y_pred_bin_list
```

## Calculating AUC for Selecting Features

```
def auc(variables, target, basetable):
  y_pred_binary_list = []
 
  x = basetable[variables]
  y = basetable[target]
 
  log_reg = linear_model.LogisticRegression()
  log_reg.fit(x, y)
 
  y_pred = log_reg.predict_proba(x)[:, 1]
  y_pred_bin = y_pred_to_binary(y_pred) # 轉做binary
 
  auc = roc_auc_score(y, y_pred_bin)
  return auc
```

## Finding the BEST
Iterating over all candidate_variables and keep track of which var is the best (auc is related to best_var). Finally, combining the current variable to return the next best variable.

```
def next_best(current_variables, candidate_variables, target, basetable):
  best_auc = -1
  best_var = None
  for v in candidate_variables:
    auc_v = auc(current_variables + [v], target, basetable)
    # Adjust the best_auc
    if auc_v >= best_auc:
      best_auc = auc_v
      best_var = v
  return best_var
```

## Applying Data Preprocessing to the dataset
```
# preprocess the dataset
df = preprocessing(df)
 
# print the information
print(f'features in df:\n{df.columns.tolist()}')
print(f'num of x_features: {len(df.columns) - 1}')
print(f'num of y_features: {1}, and the feature is {df.columns[-1]}')
```

## Implement the Coding Process
Implementing the functions into the dataset adn finding out the results.

```
candidate_variables = df.columns[:-1].to_list()
current_variables = []
target = df.columns[-1]
 
max_num_var = 5 # this value can be decided by myself
num_iterations = min(max_num_var, len(candidate_variables))
print(f'num_iterations: {num_iterations}')
 
for i in range(0, num_iterations):
  next_var = next_best(current_variables, candidate_variables, target, df)
  current_variables = current_variables + [next_var]
  candidate_variables.remove(next_var)
 
# The sorting of important features
print(f'current_variables: {current_variables}')
```

## Reference
1. The module "Statistics II"
2. Python Coding from DataCamp

> **_Please note that_** this is the translated post of my original post in CSDN (Original post on 03 Apr 2023). The post are all my ideas and thoughts, please **do not reproduce** my post. Thank you very much!