
# coding: utf-8

# # Applying K-Nearest Mode for Loan prediction

# In[61]:

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# In[62]:

loans = pd.read_csv('train_u6lujuX.csv')
test_data = pd.read_csv('test_Y3wMUE5.csv')
loans.head()


# ### Features and Target
# **Note:** Feature 'Dependents' has a value '3+' which is causing problem. Thats why it has been left for now.

# In[63]:

features = ['Gender', 
            'Married', 
            'Education',
            'Self_Employed',
            'ApplicantIncome', 
            'CoapplicantIncome',
            'LoanAmount',
            'Loan_Amount_Term',
            'Credit_History',
            'Property_Area']
target = 'Loan_Status'


# ## Preparing the data

# * **Is target column fully populated with proper data?**
# * **Replace the target values to 1 and -1**

# In[64]:

print "Is target column fully populated with proper data: ", (len(loans[loans[target] == 'Y']) + 
                                                              len(loans[loans[target] == 'N'])) == len(loans)
loans[target] = loans[target].apply(lambda x: 1 if x=='Y' else -1)


# * **Subsample dataset to make sure classes are balanced**
# * **Remove all the rows from the bigger set where null values are there**

# In[65]:

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

safe_loans_raw1 = safe_loans_raw[pd.isnull(safe_loans_raw['Gender']) != True]
print len(safe_loans_raw1)

safe_loans_raw2 = safe_loans_raw1[pd.isnull(safe_loans_raw1['Married']) != True]
print len(safe_loans_raw2)

safe_loans_raw3 = safe_loans_raw2[pd.isnull(safe_loans_raw2['Education']) != True]
print len(safe_loans_raw3)

safe_loans_raw4 = safe_loans_raw3[pd.isnull(safe_loans_raw3['Self_Employed']) != True]
print len(safe_loans_raw4)

safe_loans_raw5 = safe_loans_raw4[pd.isnull(safe_loans_raw4['ApplicantIncome']) != True]
print len(safe_loans_raw5)

safe_loans_raw6 = safe_loans_raw5[pd.isnull(safe_loans_raw5['CoapplicantIncome']) != True]
print len(safe_loans_raw6)

safe_loans_raw7 = safe_loans_raw6[pd.isnull(safe_loans_raw6['LoanAmount']) != True]
print len(safe_loans_raw7)

safe_loans_raw8 = safe_loans_raw7[pd.isnull(safe_loans_raw7['Loan_Amount_Term']) != True]
print len(safe_loans_raw8)

safe_loans_raw9 = safe_loans_raw8[pd.isnull(safe_loans_raw8['Credit_History']) != True]
print len(safe_loans_raw9)

safe_loans_raw10 = safe_loans_raw9[pd.isnull(safe_loans_raw9['Property_Area']) != True]
print len(safe_loans_raw10)


# In[67]:

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw10))
#safe_loans = safe_loans_raw.sample(percentage, seed = 1)
safe_loans = safe_loans_raw10.sample(frac=percentage)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)


# ### Changing categorical values to numerical values

# **Gender**

# In[68]:

def fun_Gender(x):
    if x == 'Male':
        return 1
    elif x == 'Female':
        return 2
    else:
        return 0   
loans_data['Gender'] = loans_data['Gender'].apply(lambda x: fun_Gender(x))
test_data['Gender'] = test_data['Gender'].apply(lambda x: fun_Gender(x))
print loans_data['Gender'].head()


# **Married**

# In[69]:

def fun_Married(x):
    if x == 'Yes':
        return 1
    elif x == 'No':
        return 2
    else:
        return 0
loans_data['Married'] = loans_data['Married'].apply(lambda x: fun_Married(x))
test_data['Married'] = test_data['Married'].apply(lambda x: fun_Married(x))
print loans_data['Married'].head()


# **Dependents**

# In[220]:

#loans_data['Dependents'] = loans_data['Dependents'].apply(lambda x: x if x >= 0 else 3 if str(x) == '3+' else 0)
#print loans_data['Dependents'].head()


# **Education**

# In[70]:

def fun_Education(x):
    if x == 'Graduate':
        return 1
    elif x == 'Not Graduate':
        return 2
    else:
        return 0
loans_data['Education'] = loans_data['Education'].apply(lambda x: fun_Education(x))
test_data['Education'] = test_data['Education'].apply(lambda x: fun_Education(x))
print loans_data['Education'].head()


# **Self_Employed**

# In[71]:

def fun_Self_Employed(x):
    if x == 'Yes':
        return 1
    elif x == 'No':
        return 2
    else:
        return 0
loans_data['Self_Employed'] = loans_data['Self_Employed'].apply(lambda x: fun_Self_Employed(x))
test_data['Self_Employed'] = test_data['Self_Employed'].apply(lambda x: fun_Self_Employed(x))
print loans_data['Self_Employed'].head()


# **ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term & Credit_History**

# In[72]:

loans_data['ApplicantIncome'] = loans_data['ApplicantIncome'].apply(lambda x: x if x >= 0 else 0)
test_data['ApplicantIncome'] = test_data['ApplicantIncome'].apply(lambda x: x if x >= 0 else 0)

loans_data['CoapplicantIncome'] = loans_data['CoapplicantIncome'].apply(lambda x: x if x >= 0 else 0)
test_data['CoapplicantIncome'] = test_data['CoapplicantIncome'].apply(lambda x: x if x >= 0 else 0)

loans_data['LoanAmount'] = loans_data['LoanAmount'].apply(lambda x: x if x >= 0 else 0)
test_data['LoanAmount'] = test_data['LoanAmount'].apply(lambda x: x if x >= 0 else 0)

loans_data['Loan_Amount_Term'] = loans_data['Loan_Amount_Term'].apply(lambda x: x if x >= 0 else 0)
test_data['Loan_Amount_Term'] = test_data['Loan_Amount_Term'].apply(lambda x: x if x >= 0 else 0)

loans_data['Credit_History'] = loans_data['Credit_History'].apply(lambda x: x if x >= 0 else 0)
test_data['Credit_History'] = test_data['Credit_History'].apply(lambda x: x if x >= 0 else 0)

print loans_data['ApplicantIncome'].head()
print loans_data['CoapplicantIncome'].head()
print loans_data['LoanAmount'].head()
print loans_data['Loan_Amount_Term'].head()
print loans_data['Credit_History'].head()


# **Property_Area**

# In[73]:

def fun_Property_Area(x):
    if x == 'Urban':
        return 1
    elif x == 'Rural':
        return 2
    elif x == 'Semiurban':
        return 3
    else:
        return 0
loans_data['Property_Area'] = loans_data['Property_Area'].apply(lambda x: fun_Property_Area(x))
test_data['Property_Area'] = test_data['Property_Area'].apply(lambda x: fun_Property_Area(x))

print loans_data['Property_Area'].head()


# ## Training the model

# In[74]:

X = loans_data[features]
y = loans_data[target]


# In[75]:

model = LogisticRegression()
model.fit(X, y)


# Test Data

# ## Making predictions using the model

# In[76]:

X_test = test_data[features]
predictions = model.predict(X_test)
#predictions_proba = model.predict_proba(X_test)
print predictions
#print predictions_proba


# ## Writing the submission file

# In[77]:

loan_id = test_data['Loan_ID']
predictionsS = pd.Series(predictions)
loan_status_predictions = predictionsS.apply(lambda x: 'Y' if x == 1 else 'N')


# In[78]:

with open('submission.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Loan_ID', 'Loan_Status'])
    for i in xrange(len(loan_id)):
        spamwriter.writerow([loan_id[i], loan_status_predictions[i]])


# In[ ]:

#print len(loans)

#x = np.arange(0, len(loans['ApplicantIncome'])) 
#x = loans['CoapplicantIncome']
#y = loans['ApplicantIncome']

#ll = plt.plot(x,y)
#plt.show()

