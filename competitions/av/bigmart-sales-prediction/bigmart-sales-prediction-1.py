
# coding: utf-8

# In[192]:

import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from collections import defaultdict
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


# In[178]:

train = pd.read_csv("Train_UWu5bXk.csv")
test = pd.read_csv("Test_u94Q5KV.csv")


# In[179]:

features = ['Item_Weight',
            'Item_Fat_Content',
            'Item_Visibility',
            'Item_Type',
            'Item_MRP',
            'Outlet_Establishment_Year',
            'Outlet_Size',
            'Outlet_Location_Type',
            'Outlet_Type']
target = ['Item_Outlet_Sales']
train_data = train[features + target].copy()
test_data = test[features].copy()


# Following is the function to convert all categorical 'str' features to numeric categorical features

# In[181]:

def dic_of_categories(data):
    print "dic_of_categories"
    dic_of_dics = {}
    for feature in data.columns:
        if type(data[feature][0]) is str:
            j = 1
            keys = [np.nan]
            values = [np.nan]
            for i in data[feature].value_counts().iteritems():
                #print i[0], i[1]
                keys.append(i[0])
                values.append(j)
                j += 1
            dic1 = dict(zip(keys, values))
            dic_of_dics.setdefault(feature, dic1) 
    return dic_of_dics


# In[182]:

def categorize_based_on_dic_of_categories(data, dic_of_dics):
    print "categorize_based_on_dic_of_categories"
    for feature in data.columns:
        if feature in dic_of_dics:
            #print dic_of_dics[feature]
            data[feature] = data[feature].apply(lambda x: dic_of_dics[feature][x])


# In[183]:

dic_of_dics = dic_of_categories(train[features]) 

categorize_based_on_dic_of_categories(train_data, dic_of_dics)
categorize_based_on_dic_of_categories(test_data, dic_of_dics)

print train_data.head()
print test_data.head()


# Imputation in train data

# In[184]:

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)

imp.fit(train_data)
train_data_np = imp.transform(train_data)
train_data = pd.DataFrame(train_data_np, columns=train_data.columns)

imp.fit(test_data)
test_data_np = imp.transform(test_data)
test_data = pd.DataFrame(test_data_np, columns=test_data.columns)


# In[185]:

print train_data.head()
print train_data.head()


# In[186]:

X = train_data[features]
y = train_data[target]


# In[187]:

model_lr = LinearRegression()
model_lr.fit(X, y)


# In[188]:

print X.shape, y.shape
print test_data.shape


# In[189]:

predictions = model_lr.predict(test_data)
print predictions


# ### Writing the submission file

# In[195]:

i = predictions
print i[2][0]
#for i in predictions.tolist():
    #print i[0]
    
print test['Item_Identifier'][0]


# In[196]:

with open('submission.csv', 'wb') as csvfile:
    submission_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    submission_writer.writerow(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
    for i in xrange(len(test)):
        submission_writer.writerow([test['Item_Identifier'][i], test['Outlet_Identifier'][i], predictions[i][0]])

