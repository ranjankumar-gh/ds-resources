
# coding: utf-8

# # Bigmart Sales Prediction

# ## Load required libraries and data

# In[59]:

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[25]:

#Read files:
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# ## Data Exploration
# ### Combine the train and test data for feature engineering.

# In[26]:

train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test],ignore_index=True)
print(train.shape, test.shape, data.shape)


# In[27]:

data.head()


# ### How many values are missing feature wise?

# In[28]:

data.apply(lambda x: sum(x.isnull()))


# ### Statistical overview of data

# In[29]:

data.describe()


# In[30]:

data.apply(lambda x: len(x.unique()))


# In[31]:

#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']

#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]

#Print frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories for varible %s'%col)
    print(data[col].value_counts())


# ## Data Cleaning

# ### Imputing Missing Values

# In[32]:

#Determine the average weight per item:
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print('Orignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight[x])
print('Final #missing: %d'% sum(data['Item_Weight'].isnull()))


# In[33]:

print(data['Outlet_Size'].unique())
outlet_size_mapping = {'Small': 1, 'Medium': 2, 'High': 3}
data['Outlet_Size'] = data['Outlet_Size'].map(outlet_size_mapping)
outlet_size_inverse_mapping = {1: 'Small', 2: 'Medium', 3: 'High'}


# In[34]:

#Import mode function:
from scipy.stats import mode

#Determing the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x:mode(x).mode[0]) )

print('Mode for each Outlet_Type:')
print(outlet_size_mode)

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Outlet_Size'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print('\nOrignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print(sum(data['Outlet_Size'].isnull()))

data['Outlet_Size'] = data['Outlet_Size'].map(outlet_size_inverse_mapping)
print(data['Outlet_Size'].head())


# ## Feature Engineering

# In[35]:

#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')

#Impute 0 values with mean visibility of that product:
miss_bool = (data['Item_Visibility'] == 0)

print('Number of 0 values initially: %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg[x])
print('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))


# In[36]:

#Determine another variable with means ratio
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis=1)
print(data['Item_Visibility_MeanRatio'].describe())


# In[37]:

#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[38]:

#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[39]:

#Change categories of low fat:
print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())

print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())


# In[40]:

#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


# In[41]:

#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[42]:

#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])


# In[43]:

data.dtypes


# In[44]:

#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)


# ## Model building

# In[45]:

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
features = [x for x in train.columns if x not in [target]+IDcol]
print(features)


# ### Baseline Model

# In[46]:

#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("alg0.csv",index=False)


# In[47]:

train.columns


# ### Linear Regression Model

# In[48]:

model_lr = LinearRegression(normalize=True)
model_lr.fit(train[features], train[target])


# In[49]:

#prediction on training data
model_lr_train_prediction = model_lr.predict(train[features])

#Perform cross-validation:
cv_score = cross_val_score(model_lr, train[features], train[target], cv=20, scoring='mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))


# In[50]:

#Print model report:
print("\nModel Report")
print("RMSE : %.4g" % np.sqrt(mean_squared_error(train[target].values, model_lr_train_prediction)))
print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),
                                                                  np.max(cv_score)))


# In[51]:

#prediction on test data
test[target] = model_lr.predict(test[features])

#Export submission file:
IDcol.append(target)
submission = pd.DataFrame({ x: test[x] for x in IDcol})
submission.to_csv("model_lr_01.csv", index=False)


# In[52]:

coef1 = pd.Series(model_lr.coef_, features).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
plt.show()


# ### Ridge Regression Model

# In[53]:

model_ridge = Ridge(alpha=0.05,normalize=True)
model_ridge.fit(train[features], train[target])

#prediction on training data
model_ridge_train_prediction = model_ridge.predict(train[features])

#Perform cross-validation:
cv_score = cross_val_score(model_ridge, train[features], train[target], cv=20, scoring='mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))

#Print model report:
print("\nModel Report")
print("RMSE : %.4g" % np.sqrt(mean_squared_error(train[target].values, model_ridge_train_prediction)))
print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),
                                                                  np.max(cv_score)))

#prediction on test data
test[target] = model_lr.predict(test[features])

#Export submission file:
IDcol.append(target)
submission = pd.DataFrame({ x: test[x] for x in IDcol})
submission.to_csv("model_ridge_01.csv", index=False)

coef2 = pd.Series(model_ridge.coef_, features).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')


# ### Decision Tree Model

# In[54]:

model_dt = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
model_dt.fit(train[features], train[target])

#prediction on training data
model_dt_train_prediction = model_dt.predict(train[features])

#Perform cross-validation:
cv_score = cross_val_score(model_dt, train[features], train[target], cv=20, scoring='mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))

#Print model report:
print("\nModel Report")
print("RMSE : %.4g" % np.sqrt(mean_squared_error(train[target].values, model_dt_train_prediction)))
print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),
                                                                  np.max(cv_score)))

#prediction on test data
test[target] = model_dt.predict(test[features])

#Export submission file:
IDcol.append(target)
submission = pd.DataFrame({ x: test[x] for x in IDcol})
submission.to_csv("model_dt_01.csv", index=False)

coef2 = pd.Series(model_dt.feature_importances_, features).sort_values()
coef2.plot(kind='bar', title='Feature Importances')


# In[55]:

features = ['Item_MRP','Outlet_Type_0','Item_Visibility','Outlet_Years']
model_dt2 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
model_dt2.fit(train[features], train[target])

#prediction on training data
model_dt_train_prediction2 = model_dt2.predict(train[features])

#Perform cross-validation:
cv_score = cross_val_score(model_dt2, train[features], train[target], cv=20, scoring='mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))

#Print model report:
print("\nModel Report")
print("RMSE : %.4g" % np.sqrt(mean_squared_error(train[target].values, model_dt_train_prediction2)))
print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),
                                                                  np.max(cv_score)))

#prediction on test data
test[target] = model_dt2.predict(test[features])

#Export submission file:
IDcol.append(target)
submission = pd.DataFrame({ x: test[x] for x in IDcol})
submission.to_csv("model_dt_02.csv", index=False)

coef2 = pd.Series(model_dt2.feature_importances_, features).sort_values()
coef2.plot(kind='bar', title='Feature Importances')


# ### Random Forest

# In[56]:

features = [x for x in train.columns if x not in [target]+IDcol]

model_rf = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
model_rf.fit(train[features], train[target])

#prediction on training data
model_rf_train_prediction = model_rf.predict(train[features])

#Perform cross-validation:
cv_score = cross_val_score(model_rf, train[features], train[target], cv=20, scoring='mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))

#Print model report:
print("\nModel Report")
print("RMSE : %.4g" % np.sqrt(mean_squared_error(train[target].values, model_rf_train_prediction)))
print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),
                                                                  np.max(cv_score)))

#prediction on test data
test[target] = model_rf.predict(test[features])

#Export submission file:
IDcol.append(target)
submission = pd.DataFrame({ x: test[x] for x in IDcol})
submission.to_csv("model_rf_01.csv", index=False)

coef2 = pd.Series(model_rf.feature_importances_, features).sort_values(ascending=False)
coef2.plot(kind='bar', title='Feature Importances')


# ## Fine tuning Random Forest with grid search

# In[67]:

pipe_rf = Pipeline([('clf', RandomForestRegressor(n_estimators=200, min_samples_leaf=100, random_state=1))])

#rf_gs = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)

max_depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

param_grid = [{'clf__max_depth': max_depth_range}]

gs = GridSearchCV(estimator=pipe_rf, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=4)
gs = gs.fit(train[features], train[target])
print( gs.best_score_)
print(gs.best_params_)

gs_train_prediction = gs.predict(train[features])

#Print model report:
print("\nModel Report")
print("RMSE : %.4g" % np.sqrt(mean_squared_error(train[target].values, gs_train_prediction)))
print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),
                                                                  np.max(cv_score)))


# In[69]:

#prediction on test data
test[target] = gs.predict(test[features])

#Export submission file:
IDcol.append(target)
submission = pd.DataFrame({ x: test[x] for x in IDcol})
submission.to_csv("model_gs_01.csv", index=False)

#coef2 = pd.Series(gs.feature_importances_, features).sort_values(ascending=False)
#coef2.plot(kind='bar', title='Feature Importances')

