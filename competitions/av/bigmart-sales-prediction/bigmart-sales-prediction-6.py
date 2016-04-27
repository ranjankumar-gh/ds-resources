
# coding: utf-8

# # AV: Bigmart Sales Prediction - Solution 6

# In[126]:

import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


# In[127]:

data = pd.read_csv('Train_UWu5bXk.csv')
test = pd.read_csv("Test_u94Q5KV.csv")


# In[128]:

data.head()


# In[129]:

features = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Location_Type',
       'Outlet_Type']
target = ['Item_Outlet_Sales']


# In[130]:

item_identifier_weight = data[['Item_Identifier', 'Item_Weight']]

unique_item_identifier = data['Item_Identifier'].unique()
item_identifier_weight_dic = {x: np.NaN for x in unique_item_identifier}

for j in item_identifier_weight.iterrows():
        if not np.isnan(j[1]['Item_Weight']):
            item_identifier_weight_dic[j[1]['Item_Identifier']] = j[1]['Item_Weight']
            
def f(x):
    a = x[0]
    b = x[1]
    if np.isnan(x[1]):
        b = item_identifier_weight_dic[x[0]]
    return pd.Series([a, b], ['Item_Identifier', 'Item_Weight'])
           
item_identifier_weight_data = item_identifier_weight.apply(f, axis=1)


# In[131]:

item_identifier_weight = test[['Item_Identifier', 'Item_Weight']]

#unique_item_identifier = test['Item_Identifier'].unique()
#item_identifier_weight_dic = {x: np.NaN for x in unique_item_identifier}

for j in item_identifier_weight.iterrows():
        if not np.isnan(j[1]['Item_Weight']):
            item_identifier_weight_dic[j[1]['Item_Identifier']] = j[1]['Item_Weight']
            
def f(x):
    a = x[0]
    b = x[1]
    if np.isnan(x[1]):
        b = item_identifier_weight_dic[x[0]]
    return pd.Series([a, b], ['Item_Identifier', 'Item_Weight'])
           
item_identifier_weight = item_identifier_weight.apply(f, axis=1)


# In[132]:

data = data[['Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Location_Type',
       'Outlet_Type', 'Item_Outlet_Sales']]
data = pd.concat([data, item_identifier_weight_data], axis=1)
data = data[data['Item_Weight'] >= 0]
data.head()


# In[133]:

test = test[['Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Location_Type',
       'Outlet_Type']]
test = pd.concat([test, item_identifier_weight_test], axis=1)
#test = test[test['Item_Weight'] >= 0]
test.head()


# In[134]:

test.isnull().sum()


# In[135]:

print(data.columns)
X = data[features]
y = data['Item_Outlet_Sales']
X_test = test[features]


# ## Imputing missing values
# Note: features with null values has not been taken in the first round

# In[136]:

X.isnull().sum()


# In[137]:

X_test.isnull().sum()


# In[138]:

X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X.head()


# In[139]:

X_test.head()


# ## Scaling the data

# In[140]:

sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


# ## Dividing the data in training and validation set

# In[141]:

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)


# ## Training

# In[142]:

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)


# ## Predictions

# In[143]:

predictions = model_lr.predict(X_val)
print(model_lr.score(X_val, y_val))
model_lr.score(X_train, y_train)


# ## Writing the submission file

# In[144]:

model_lr = LinearRegression()
model_lr.fit(X, y)


# In[145]:

predictions = model_lr.predict(X_test)


# In[146]:

result = test[['Item_Identifier', 'Outlet_Identifier']]
result['Item_Outlet_Sales'] = pd.Series(predictions, index=result.index)
result.to_csv('submission.csv', index = False, index_label = 'Id' )

