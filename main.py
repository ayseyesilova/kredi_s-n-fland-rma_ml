# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:36:22 2024

@author: HP
"""

    
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
""" import theano
import keras
import tensorflow
import matplotlib.pyplot as plt"""
import seaborn as sns
sns.set()
working_directory = os.getcwd()
datapath= f"{working_directory}\loan.csv"
data = pd.read_csv(datapath)

data.info()
data.head()

age_dummies = pd.get_dummies(data=data,prefix="age",columns=["age"])
gender_dummies = age_dummies.replace(to_replace={'gender': {'Female': 1,'Male':0}})
data = gender_dummies.replace(to_replace={"loan_status": {"Approved":1,"Denied":0}})
sns.countplot(y=data.loan_status ,data=data)
plt.show()

data.hist(figsize=(15,12),bins = 15)
plt.title("Features Distribution")
plt.show()

plt.figure(figsize=(15,15))
p=sns.heatmap(data.corr(), annot=True,cmap='RdYlGn',center=0)

X=data.drop(["loan_status"],axis=1)
Y = data.loan_status

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)


# Feature Scaling because yes we don't want one independent variable dominating the other and it makes computations easy
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





