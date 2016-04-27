
# coding: utf-8

# # AV: Bigmart Sales Prediction - Solution 4

# In[285]:

import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


# In[286]:

data = pd.read_csv('Train_UWu5bXk.csv')
test = pd.read_csv("Test_u94Q5KV.csv")


# In[287]:

data.head()


# In[288]:

print(data.columns)
features = ['Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Location_Type',
       'Outlet_Type']
X = data[features]
y = data['Item_Outlet_Sales']
X_test = test[features]


# ## Imputing missing values
# Note: features with null values has not been taken in the first round

# In[289]:

X.isnull().sum()


# In[290]:

X_test.isnull().sum()


# In[291]:

X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X.head()


# In[292]:

X_test.head()


# ## Scaling the data

# In[293]:

sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


# ## Dividing the data in training and validation set

# In[294]:

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)


# ## Training

# In[295]:

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)


# ## Predictions

# In[296]:

predictions = model_lr.predict(X_val)
print(model_lr.score(X_val, y_val))
model_lr.score(X_train, y_train)


# ## Writing the submission file

# In[297]:

model_lr = LinearRegression()
model_lr.fit(X, y)


# In[298]:

predictions = model_lr.predict(X_test)


# In[299]:

result = test[['Item_Identifier', 'Outlet_Identifier']]
result['Item_Outlet_Sales'] = pd.Series(predictions, index=result.index)
result.to_csv('submission.csv', index = False, index_label = 'Id' )

