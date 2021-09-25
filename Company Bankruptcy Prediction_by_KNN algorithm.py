# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 20:33:27 2021

@author: jagadeesan
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#IMPORTING THE DATASET
data=pd.read_csv("data.csv")

data.info()
data.describe()
data.isnull().sum()

#INPUT AND TARGET SPLIT
X = data.iloc[:,1:].values
Y = data.iloc[:,0].values

#TRAIN AND TEST SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state = 42)

#SINCE IT HAS SO MUCH OF FEATURES(COLUMNS) WE CAN ANALYSE THE IMPORTANCE OF EACH FEATURE

#WE CAN USE RFE FEATURE SELECTION TOOL WITH RANDOM FOREST CLASSIFIER

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

n_features = [5,10,15,20,25,30]
for i in n_features:
    select =  RFE(RandomForestClassifier(n_estimators=100,random_state=42),n_features_to_select=i)
    select.fit(x_train,y_train)
    x_train_rfe = select.transform(x_train)
    x_test_rfe = select.transform(x_test)
    score = RandomForestClassifier().fit(x_train_rfe,y_train).score(x_test_rfe,y_test)
    print("Test score: {:.3f}".format(score), " number of features: {}".format(i))
    
#AFTER RUNNING THIS WE CAN WORK WITH 15 FEATURES SINCE IT EXHIBITS 97% ACCURACY.
#LETS RUN THE ALGORITHM AGAIN TO GET THE FEATURE DETAILS
    
select = RFE(RandomForestClassifier(n_estimators = 100, random_state = 42),n_features_to_select=15)
select.fit(x_train,y_train)
mask = select.get_support()  #TO KNOW THE NAMES FOR SELECTABLE FEATURES
x_train_rfe = select.transform(x_train)
x_test_rfe = select.transform(x_test)
score = RandomForestClassifier().fit(x_train_rfe,y_train).score(x_test_rfe,y_test)
print("Test score: {:.3f}".format(score), " number of features: {}".format(15))

#CRAETING NEW DATAFRAME WITH SELECTED FEATURES

features = pd.DataFrame({'features':list(data.iloc[:,1:].keys()),'select':list(mask)})
features = list(features[features['select']==True]['features'])
features.append(' Total debt/Total net worth')
features.append(' Cash/Total Assets')
features.append(' Fixed Assets Turnover Frequency')
features.append('Bankrupt?')


data = data[features]

    
#WE GOT THE DATAFRAME WITH SELECTED FEATURES WE CAN DO FURTHER ANALYSIS

sns.countplot(data=data, x='Bankrupt?', palette='bwr')
plt.show()

data.groupby('Bankrupt?').size()

#WE CAN SEE IT IS HIGHLY INBALANCED  DATA THATS WHY WE GOT GOOD RESULTS WHEN RANDOMFOREST TECHNIQUE APPLIED
#SO WE CAN DO ADDITIONAL DATA ANALYIS TO FIND THE TRUTH

data.hist(figsize=(10,10),edgecolor="white")
plt.show()

#LOT OF OUTLIERS!!!AND MOST OF THE VALUES IN MANY COLUMNS IN TINY BIN REANGE
#LETS A FEATURE Non-industry income and expenditure/revenue
#SEGRIGATING INTO BINS

segments = pd.cut(data[" Non-industry income and expenditure/revenue"],bins=10)
segments.value_counts()

lower = data[' Non-industry income and expenditure/revenue'] >0.3025
upper = data[' Non-industry income and expenditure/revenue'] <0.3045

close = data[lower & upper]
print('Rows with outliers: {}'.format(data.shape[0]))
print('Rows without outliers: {}'.format(close.shape[0]))
print('information lost = {} rows'.format(data.shape[0]-close.shape[0]))
close[' Non-industry income and expenditure/revenue'].hist(edgecolor='white')

data.describe()
data.shape


#CORRELATIONS CHEKINGS
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True), annot=True)


#WE GOT SOME CORRELATIONS WITH TARGET VALUES
#LETS PLOT FURTHER WITH TOP 3 

fig, ax = plt.subplots(1,3, figsize=(20, 6))

sns.scatterplot(data=data, x=" Net profit before tax/Paid-in capital", y=" Persistent EPS in the Last Four Seasons", hue="Bankrupt?", ax=ax[0])
sns.scatterplot(data=data, x=" Persistent EPS in the Last Four Seasons", y=" Net Value Per Share (B)", hue="Bankrupt?", ax=ax[1])
sns.scatterplot(data=data, x=" Net Income to Stockholder's Equity", y=" Borrowing dependency", hue="Bankrupt?", ax=ax[2])

#SO FROM SCATTERPLOT RESULTS Net profit before tax/Paid-in capital,
#Persistent EPS in the Last Four Seasons AND Net Value Per Share (B)
#HAVING THESE 3 VALUES AT LOW ARE MAY TO BANKRUPT WITH HIGH CHANCE

#FOR BORROWING TENDENCY AT 0.4 NO STRONG CHANCE FOR GOING BANKRUPT
#FOR NETINCOME TO STOCKHOLDERS EQUITY AT 0.8 NO STRONG CHANCE FOR GOING BANKRUPT
#BUT AT HIGHER VALUES GOING BANKUPT IS VERY STRONG

#companies with a low 'Net profit before tax/Paid-in capital',
#'Persistent EPS in the Last Four Seasons' and 'Net Value Per Share (A)' 
#tend to go bankrupt. A KNN algorithm would
# yield good results since the clusters are so evident


#COMPARING THE MEDIAN OF BANKRUPT AND NOT BANKRUPT COMPANIES OF FEATURES TO 
#FIND FURHTER TENDENCY 
central = data.groupby('Bankrupt?').median().reset_index()
features = list(central.keys()[1:])

fig, ax = plt.subplots(5,3, figsize=(20,20))

ax = ax.ravel()
position = 0

for i in features:
    sns.barplot(data=central, x='Bankrupt?', y=i, ax=ax[position], palette='bwr')
    position += 1
    
plt.show()

display(central)

#FINAL CONCLUSION
#Companies with:
#
#high "Interest-bearing debt interest rate" tend to go bankrupt (≈ 0.000499)
#high "Total debt/Total net worth" tend to go bankrupt (≈ 0.015723)
#high "Fixed Assets Turnover Frequency" tend to go bankrupt (≈ 0.001225)
#low "Cash/Total Assets" tend to go bankrupt (≈ 0.023755)
#low "Equity to Liability" tend to go bankrupt (≈ 0.018662)

#Also, These indicators should be enough to build a reliable model since
# the trend is very clear. Let's build our model

#MODEL BUILDING

model = ['Bankrupt?',' Net profit before tax/Paid-in capital',' Persistent EPS in the Last Four Seasons',' Interest-bearing debt interest rate',' Total debt/Total net worth',' Fixed Assets Turnover Frequency',' Cash/Total Assets',' Equity to Liability']
model = data[model]
X = model.iloc[:,1:].values
y = model.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

best_n = 0
best_training = 0
best_test = 0

from sklearn.neighbors import KNeighborsClassifier

for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    
    training = knn.score(X_train, y_train)
    test = knn.score(X_test, y_test)
    
    if test > best_test:
        best_n = i
        best_training = training
        best_test = test

print("best number of neighbors: {}".format(best_n))
print("best training set score : {:.3f}".format(best_training))
print("best test set score: {:.3f}".format(best_test))









    
    