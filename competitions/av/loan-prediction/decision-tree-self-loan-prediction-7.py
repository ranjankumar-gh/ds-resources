
# coding: utf-8

# # Applying Decision Tree for Loan prediction

# In[34]:

import graphlab
import math


# In[35]:

loans = graphlab.SFrame.read_csv('train_u6lujuX.csv')
test_data = graphlab.SFrame.read_csv('test_Y3wMUE5.csv')
loans.head()


# ### Features and Target

# In[36]:

features = ['Gender', 
            'Married', 
            'Dependents',
            'Education',
            'Self_Employed',
            'ApplicantIncome', 
            'CoapplicantIncome',
            'LoanAmount',
            'Loan_Amount_Term',
            'Credit_History',
            'Property_Area']
target = 'Loan_Status'

loans = loans[features + [target]]


# ## Preparing the data

# * **Is target column fully populated with proper data?**
# * **Replace the target values to 1 and -1**

# In[37]:

print "Is target column fully populated with proper data: ", (len(loans[loans[target] == 'Y']) + 
                                                              len(loans[loans[target] == 'N'])) == len(loans)
loans[target] = loans[target].apply(lambda x: 1 if x=='Y' else -1)


# ## Subsample dataset to make sure classes are balanced

# In[38]:

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)
#loan_data = loans


# ## Transform categorical data into binary features

# ### Training Data

# Categorize 'ApplicantIncome'

# In[39]:

def fun_ApplicantIncome(x):
    if x < 1000:
        return 'range_0_1k'
    elif x < 2000:
        return 'range_1_2k'
    elif x < 3000:
        return 'range_2_3k'
    elif x < 4000:
        return 'range_3_4k'
    elif x < 5000:
        return 'range_4_5k'
    elif x < 6000:
        return 'range_5_6k'
    elif x < 7000:
        return 'range_6_7k'
    elif x < 8000:
        return 'range_7_8k'
    elif x < 9000:
        return 'range_8_9k'
    elif x < 10000:
        return 'range_9_10k'
    elif x < 11000:
        return 'range_10_11k'
    elif x < 12000:
        return 'range_11_12k'
    elif x < 13000:
        return 'range_12_13k'
    elif x < 14000:
        return 'range_13_14k'
    elif x < 15000:
        return 'range_14_15k'
    elif x < 16000:
        return 'range_15_16k'
    elif x < 17000:
        return 'range_16_17k'
    elif x < 18000:
        return 'range_17_18k'
    elif x < 19000:
        return 'range_18_19k'
    elif x < 20000:
        return 'range_19_20k'
    elif x < 30000:
        return 'range_20_21k'
    elif x >= 30000:
        return 'range_high'
loans_data['ApplicantIncome'] = loans_data['ApplicantIncome'].apply(lambda x: fun_ApplicantIncome(x))
test_data['ApplicantIncome'] = test_data['ApplicantIncome'].apply(lambda x: fun_ApplicantIncome(x))


# Categorize 'CoapplicantIncome'

# In[40]:

def fun_CoapplicantIncome(x):
    if x < 1000:
        return 'range_0_1k'
    elif x < 2000:
        return 'range_1_2k'
    elif x < 3000:
        return 'range_2_3k'
    elif x < 4000:
        return 'range_3_4k'
    elif x < 5000:
        return 'range_4_5k'
    elif x < 6000:
        return 'range_5_6k'
    elif x < 7000:
        return 'range_6_7k'
    elif x < 8000:
        return 'range_7_8k'
    elif x < 9000:
        return 'range_8_9k'
    elif x < 10000:
        return 'range_9_10k'
    elif x < 11000:
        return 'range_10_11k'
    elif x < 12000:
        return 'range_11_12k'
    elif x < 13000:
        return 'range_12_13k'
    elif x < 14000:
        return 'range_13_14k'
    elif x < 15000:
        return 'range_14_15k'
    elif x < 16000:
        return 'range_15_16k'
    elif x < 17000:
        return 'range_16_17k'
    elif x < 18000:
        return 'range_17_18k'
    elif x < 19000:
        return 'range_18_19k'
    elif x < 20000:
        return 'range_19_20k'
    elif x < 30000:
        return 'range_20_21k'
    elif x >= 30000:
        return 'range_high'
loans_data['CoapplicantIncome'] = loans_data['CoapplicantIncome'].apply(lambda x: fun_CoapplicantIncome(x))
test_data['CoapplicantIncome'] = test_data['CoapplicantIncome'].apply(lambda x: fun_CoapplicantIncome(x))


# Categorize 'LoanAmountTerm'

# In[41]:

#loan_amount_term = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]

def fun_LoanAmountTerm(x):
    if x in loans_data['Loan_Amount_Term']:
        return x
    else:
        return 0
    
loans_data['Loan_Amount_Term'] = loans_data['Loan_Amount_Term'].apply(lambda x: fun_LoanAmountTerm(x))
test_data['Loan_Amount_Term'] = test_data['Loan_Amount_Term'].apply(lambda x: fun_LoanAmountTerm(x))


# Categorize 'LoanAmount'

# In[42]:

def fun_LoanAmount(x):
    if x < 100:
        return 'range_0_1h'
    elif x < 200:
        return 'range_1_2h'
    elif x < 300:
        return 'range_2_3h'
    elif x < 400:
        return 'range_3_4'
    elif x < 500:
        return 'range_4_5'
    elif x < 600:
        return 'range_5_6'
    else:
        return 'range_high'
    
loans_data['LoanAmount'] = loans_data['LoanAmount'].apply(lambda x: fun_LoanAmount(x))
test_data['LoanAmount'] = test_data['LoanAmount'].apply(lambda x: fun_LoanAmount(x))


# Converting features to 0 and 1 based features

# In[43]:

#loans_data = risky_loans.append(safe_loans)
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})    
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)
    
    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)


# In[44]:

features


# ### Test Data

# In[45]:

for feature in features:
    test_data_one_hot_encoded = test_data[feature].apply(lambda x: {x: 1})    
    test_data_unpacked = test_data_one_hot_encoded.unpack(column_name_prefix=feature)
    
    # Change None's to 0's
    for column in test_data_unpacked.column_names():
        test_data_unpacked[column] = test_data_unpacked[column].fillna(0)

    test_data.remove_column(feature)
    test_data.add_columns(test_data_unpacked)


# Our test data does not have target values, so following not applicable.

# In[69]:

#features = loans_data.column_names()
#features.remove('Loan_Status')  # Remove the response variable
#features


# In[47]:

print "Number of features (after binarizing categorical variables) = %s" % len(features)


# Let's explore what one of these columns looks like:

# In[88]:

married_list = []
for i in xrange(len(test_data)):
    married_list.append(0)
married_array = graphlab.SArray(married_list)
test_data.add_column(married_array, 'Married.')


# In[89]:

#test_data.remove_column('Married.')
test_data['Married.']


# #Decision tree implementation

# ## Early stopping methods for decision trees
# 1. Reached a **maximum depth**. (set by parameter `max_depth`).
# 2. Reached a **minimum node size**. (set by parameter `min_node_size`).
# 3. Don't split if the **gain in error reduction** is too small. (set by parameter `min_error_reduction`).

# In[49]:

def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    if len(data) <= min_node_size:
        return True
    else:
        return False


# In[50]:

def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    return error_before_split - error_after_split


# In[51]:

def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    
    # Count the number of 1's (safe loans)
    count_of_1 = 0
    for label in labels_in_node:
        if label == 1:
            count_of_1 += 1
        
    # Count the number of -1's (risky loans)
    count_of_minus_1 = 0
    for label in labels_in_node:
        if label == -1:
            count_of_minus_1 += 1
                
    # Return the number of mistakes that the majority classifier makes.
    if count_of_1 > count_of_minus_1:
        return count_of_minus_1
    else:
        return count_of_1    


# In[52]:

# Test case 1
example_labels = graphlab.SArray([-1, -1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 1 failed... try again!'

# Test case 2
example_labels = graphlab.SArray([-1, -1, 1, 1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 2 failed... try again!'
    
# Test case 3
example_labels = graphlab.SArray([-1, -1, -1, -1, -1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 3 failed... try again!'


# ## Function to pick best feature to split on

# In[63]:

def best_splitting_feature(data, features, target):
    
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:

        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        right_split =  data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        left_mistakes = intermediate_node_num_mistakes(left_split[target])   

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error

        if error < best_error:
            best_error = error
            best_feature = feature
    
    return best_feature # Return the best feature we found


# ## Building the tree

# In[54]:

def create_leaf(target_values):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True    }   ## YOUR CODE HERE
    
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])
    
    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = +1         ## YOUR CODE HERE
    else:
        leaf['prediction'] = -1        ## YOUR CODE HERE
        
    # Return the leaf node        
    return leaf 


# ## Incorporating new early stopping conditions in binary decision tree implementation

# In[61]:

def decision_tree_create(data, features, target, current_depth = 0, 
                         max_depth = 10, min_node_size=1, 
                         min_error_reduction=0.0):
    
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    
    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached. All data points have the same target value."                
        return create_leaf(target_values)
    
    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."                
        return create_leaf(target_values)    
    
    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)
    
    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data, min_node_size):          ## YOUR CODE HERE 
        print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values) ## YOUR CODE HERE
    
    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Calculate the error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes = data[data[splitting_feature] == 0]   ## YOUR CODE HERE
    right_mistakes = data[data[splitting_feature] == 1]  ## YOUR CODE HERE
    error_after_split = (len(left_mistakes) + len(right_mistakes)) / float(len(data))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction <= min_error_reduction:        ## YOUR CODE HERE
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values)  ## YOUR CODE HERE 
    
    
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (                      splitting_feature, len(left_split), len(right_split))
    
    
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, 
                                      current_depth + 1, max_depth, min_node_size, min_error_reduction)
    
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


# Here is a function to count the nodes in your tree:

# In[56]:

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


# Run the following test code to check the implementation. Make sure you get 'Test passed' before proceeding.

# In[64]:

small_decision_tree = decision_tree_create(loans_data, features, target, max_depth = 2, 
                                        min_node_size = 10, min_error_reduction=0.0)
if count_nodes(small_decision_tree) == 7:
    print 'Test passed!'
else:
    print 'Test failed... try again!'
    print 'Number of nodes found                :', count_nodes(small_decision_tree)
    print 'Number of nodes that should be there : 5' 


# ##Build the tree!

# Now that our code is working, we will train a tree model on the train_data with
# * max_depth = 6
# * min_node_size = 100,
# * min_error_reduction = 0.0

# In[186]:

# Make sure to cap the depth at 6 by using max_depth = 6
my_decision_tree = decision_tree_create(loans_data, features, 'Loan_Status', max_depth = 1, min_node_size=1, min_error_reduction=0.0)
#data, features, target, current_depth = 0, max_depth = 10, min_node_size=1, min_error_reduction=0.0


# ## Making predictions with a decision tree

# In[187]:

def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)
               ### YOUR CODE HERE


# ## Making predictions using the tree

# In[188]:

#classify(my_decision_tree, test_data, annotate=True)


# In[189]:

prediction = test_data.apply(lambda x: classify(my_decision_tree, x))


# In[190]:

print prediction


# ## Writing the submission file

# In[191]:

import pandas as pd
loan_id = test_data['Loan_ID']
predictionsS = pd.Series(prediction)
loan_status_predictions = predictionsS.apply(lambda x: 'Y' if x == 1 else 'N')


# In[192]:

import csv
with open('submission.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Loan_ID', 'Loan_Status'])
    for i in xrange(len(loan_id)):
        spamwriter.writerow([loan_id[i], loan_status_predictions[i]])


# In[193]:

#print len(loans)

#x = np.arange(0, len(loans['ApplicantIncome'])) 
#x = loans['CoapplicantIncome']
#y = loans['Dependents']

#ll = plt.plot(x,y)
#plt.show()


# In[ ]:



