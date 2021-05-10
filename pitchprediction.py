"""
@author: gglas

Develop a model that predicts the probability that a pitch is:
    a 4-seam fastball, 2-seam fastball, Curveball, Slider, Changeup
    using logistic regression analysis for categorical predictions
"""


import pandas as pd
import sklearn 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style = 'white')
sns.set(style = 'whitegrid', color_codes = True)

pitches_train = pd.read_csv(r'C:\Users\gglas\Documents\Career Docs\Applications_Interviews\Mets\Q2_pitches_train.csv')
pitches_test = pd.read_csv(r'C:\Users\gglas\Documents\Career Docs\Applications_Interviews\Mets\Q2_pitches_test.csv')
pitches_test2 = pitches_test.drop(['FF','FT','CB','SL','CH'], axis=1)
total_pitches = len(pitches_train.index)
pitches_names = ['FF','FT','CB','SL','CH']

#Visualize dataset
#print(pitches_train.shape)
#print(list(pitches_train.columns))

#Create countplot to visualize pitch_type counts
sns.countplot(x = 'pitch_type', data = pitches_train, order = pitches_train['pitch_type'].value_counts().index)
plt.title('Pitch Type Counts from Train Data')
plt.show()

#Print percent of total pitches from train set
#This will be helpful to compare to the test set data
print('From training data: ')

for n in pitches_names:
    count = len(pitches_train[pitches_train['pitch_type']==n])
    pct = count / total_pitches
    print('Percent of ', n , round(pct*100,2), '%')

#Use dummies function to determine which pitches thrown in training data set
dummies = pd.get_dummies(pitches_train['pitch_type'])

#train model on subset of train data file
pitches_train2 = pitches_train.copy()
pitches_train2 = pitches_train2.drop('pitch_type', axis =1)
#pitches_train2 = pd.concat([pitches_train2, dummies], axis = 1)

#Train Data using logistic regression 
#Included all non-pitch_types variables as inputs to the model
#X = pitches_train2.drop(['inning', 'cid','is_lhp'], axis = 1)
#X =  pitches_train2

result = []
#temp = pd.DataFrame(pitches_test2[pitcherid])

print('From testing data:')

#Run logistic regression on each pitch type
for n in pitches_names:
    print('Running log regression on ', n)
    temp = pd.concat([pitches_train2, dummies[n]], axis = 1)
    y = temp[n]
    X = temp.drop([n], axis = 1)
    
    #use train_test_split to train the model with the train data
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    from sklearn.linear_model import LogisticRegression
    logmodel = LogisticRegression(max_iter = 1000)
    
    #use logistic regression to test the model on the test data
    logmodel.fit(X, y)
    predictions = logmodel.predict_proba(pitches_test2)
    
    #if running the training data run analysis on the quality of the model
    #from sklearn.metrics import classification_report
    #print(classification_report(pitches_test2, predictions))
    #from sklearn.metrics import confusion_matrix
    #print(confusion_matrix(y_test, predictions))
    
    #Display the percent of each pitch type thrown 
    print(round(sum(predictions[:,1])*100/len(pitches_test2),2), '%', n)
    result.append(predictions[:,1])
   

#combine the model results with the initial test data
dataframe = pd.DataFrame(result)
data2 = dataframe.transpose()
data2 = data2.rename(columns={0: 'FF', 1: 'FT', 
                          2: 'CB', 3: 'SL',
                          4: 'CH'})
pitches_test2 = pd.concat([pitches_test2, data2], axis = 1)

#Output dataframe to file "Q2_pitches_test_RESULTS.csv"
pitches_test2.to_csv(r'C:\Users\gglas\Documents\Career Docs\Applications_Interviews\Mets\Q2_pitches_test_RESULTS.csv',index = False)

totals = []
for n in pitches_names:
    temp = sum(data2[n])
    totals.append(temp)

totals.sort(reverse = True)
names_sorted = (['FF','SL','FT','CH','CB'])

#Figure showing pitch type counts sorted by number of counts
colors = ['steelblue','peru','olivedrab','indianred','mediumpurple']    
fig = plt.figure()
plt.bar(names_sorted,totals, color = colors)
plt.xlabel('pitch_type')
plt.ylabel('Counts')
plt.title('Pitch Types Counts from Test Data')
plt.show()