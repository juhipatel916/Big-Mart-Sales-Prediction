import os
import pandas as pd
import numpy as np

# files:
train = pd.read_csv("train.csv", low_memory=False)
test = pd.read_csv("test.csv", low_memory=False)

train[:5]

# check different column between two file
train.columns.equals(test.columns)
train.columns.difference(test.columns)

#combine all data in one file
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True)
print train.shape, test.shape, data.shape

data.describe()
data.columns
print data.columns
data.dtypes

# check missing value
data.apply(lambda x: sum(x.isnull()))

# check variable levels
data.apply(lambda x: len(x.unique()))

# Filter variables
categorical_columns = [x for x in data.columns if data.dtypes[x] == 'object']

# Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]

# Frequency table 
print("Frequency Table")
for col in categorical_columns:
    print('\nFrequency table for varible %s' % col)
    print(data[col].value_counts(sort=True, dropna=False))

data['Item_Fat_Content'].replace({'LF': 'Low Fat', 
                                  'low fat': 'Low Fat', 
                                  'reg':'Regular'}, inplace=True)
data["Item_Fat_Content"].value_counts(sort=True, dropna=False)


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x="Item_Fat_Content",  hue="Item_Type", data=data);
data['Item_MRP'].hist()
item_avg_weight = data.groupby("Item_Identifier").Item_Weight.mean()

miss_idx = data['Item_Weight'].isnull() 

print('Orignal #missing: %d' % sum(miss_idx))

# input missing data
data.loc[miss_idx, 'Item_Weight'] = data.loc[miss_idx, 'Item_Identifier'].apply(lambda x: item_avg_weight[x])
print('Final # missing: %d' % sum(data['Item_Weight'].isnull()))


from scipy.stats import mode

# get mode
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x:mode(x).mode[0]))
print('Mode for each Outlet_Type:')
print(outlet_size_mode)

# get null index
miss_idx = data['Outlet_Size'].isnull() 

# input missing data
print('\nOrignal #missing: %d' % sum(miss_idx))
data.loc[miss_idx, 'Outlet_Size'] = data.loc[miss_idx, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print('Final # missing: %d' % sum(data['Outlet_Size'].isnull()))

data.groupby('Outlet_Type').Item_Outlet_Sales.mean()

visibility_avg = data.groupby('Item_Identifier').Item_Visibility.mean()

miss_idx = (data['Item_Visibility'] == 0)

print 'Number of 0 values initially: %d' % sum(miss_idx)
data.loc[miss_idx, 'Item_Visibility'] = data.loc[miss_idx, 'Item_Identifier'].apply(lambda x: visibility_avg[x])
print 'Number of 0 values after modification: %d' % sum(data['Item_Visibility'] == 0)

data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis=1)
print data['Item_Visibility_MeanRatio'].describe()

data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])

data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

train_idx = (data['source'] == 'train')

sub_data = data.loc[train_idx, ['Item_Identifier', 'Item_Outlet_Sales']]

item_avg_sales = sub_data.groupby("Item_Identifier").Item_Outlet_Sales.mean()

percentile = np.percentile(item_avg_sales, np.arange(0, 100, 25))
twentyfive = percentile[1]
fifty = percentile[2]
seventyfive = percentile[3]

first_idx = item_avg_sales.apply(lambda x: x < twentyfive)
second_idx = item_avg_sales.apply(lambda x: x >= twentyfive and x < fifty)
third_idx = item_avg_sales.apply(lambda x: x >= fifty and x < seventyfive)
fourth_idx = item_avg_sales.apply(lambda x: x > seventyfive)

first = item_avg_sales.loc[first_idx, ].index.values
second = item_avg_sales.loc[second_idx, ].index.values
thrid = item_avg_sales.loc[third_idx, ].index.values
fourth = item_avg_sales.loc[fourth_idx, ].index.values

def id_to_percentile(x):
    if x in first:
        return('first')
    elif x in second:
        return('second')
    elif x in thrid:
        return('thrid')
    elif x in fourth:
        return('fourth')

data['Percentile'] = data['Item_Identifier'].apply(lambda x: id_to_percentile(x))

data['Percentile'].value_counts()
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()
NC_idx = (data['Item_Type_Combined'] == "Non-Consumable")

data.loc[NC_idx, 'Item_Fat_Content'] = "Non-Edible"

data['Item_Fat_Content'].value_counts()

np.array(data.select_dtypes(include=["object_"]).columns)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet_Type', 'Outlet', 'Percentile']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])

data.head()

data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                              'Item_Type_Combined', 'Outlet', 'Percentile'])

data.dtypes
data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

test.drop(['Item_Outlet_Sales', 'source'], axis=1, inplace=True)
train.drop(['source'], axis=1, inplace=True)

train.to_csv("train_modified.csv", index=False)
test.to_csv("test_modified.csv", index=False)

from sklearn import cross_validation, metrics

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    alg.fit(dtrain[predictors], dtrain[target])
        
    dtrain_predictions = alg.predict(dtrain[predictors])

    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], 
                                                dtrain[target], 
                                                cv=20, 
                                                scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    print "\nModel Report"
    print "RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, 
                                                             dtrain_predictions))
    print "CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score), 
                                                                             np.std(cv_score),
                                                                             np.min(cv_score),
                                                                             np.max(cv_score))
    
    dtest[target] = alg.predict(dtest[predictors])
    
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


# Decision Tree

from sklearn.tree import DecisionTreeRegressor

predictors = [x for x in train.columns if x not in [target] + IDcol]

alg3 = DecisionTreeRegressor(max_depth=15, 
                             min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'decisionTree1.csv')

coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

predictors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years']

alg4 = DecisionTreeRegressor(max_depth=8, 
                             min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'decisionTree2.csv')

coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')

# Random Forest

from sklearn.ensemble import RandomForestRegressor

predictors = [x for x in train.columns if x not in [target] + IDcol]

alg5 = RandomForestRegressor(n_estimators=200, 
                             max_depth=5, 
                             min_samples_leaf=100, 
                             n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'randomForest1.csv')

coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')


predictors = [x for x in train.columns if x not in [target] + IDcol]

alg6 = RandomForestRegressor(n_estimators=400,
                             max_depth=6, 
                             min_samples_leaf=100,
                             n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDcol, 'randomForest2.csv')

coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')
