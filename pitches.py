# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:52:55 2021

@author: gglas
"""

import numpy as np, numpy.random
import matplotlib.pyplot as plt
import pandas as pd


pitches = []

total_days = 400
avg_throws = 170
std_throws =50

for d in range(total_days):
    pitches_each_day = np.random.normal(avg_throws,std_throws)
    if pitches_each_day < 0 :
        pitches_each_day = 0
    pitches.append(pitches_each_day)

plt.figure()
plt.hist(pitches, bins = 40)
plt.title('Histogram of Daily Number of Throws')
plt.xlabel('Number of Throws')
plt.ylabel('Number of Days')
plt.show()
    
four_types = 4
pitches_divided = []

for n in range(total_days):
    temp = np.random.dirichlet(np.ones(four_types), size = 1) * pitches[n]
    temp = np.round(temp)
    temp = temp.astype(int)
    pitches_divided.append(temp)
    
pitches_divided = np.reshape(pitches_divided,(total_days,four_types))

Nm_low = 30
Nm_med = 60
Nm_high = 80
Nm_veryhigh = 100

torques = np.column_stack((pitches_divided[:,0]*Nm_low, pitches_divided[:,1]*Nm_med, pitches_divided[:,2]*Nm_high, pitches_divided[:,3]*Nm_high))

#Create data frames from torques and pitches_divided
pitches_data = pd.DataFrame(data=pitches_divided, columns=["Pitches # Low", "Pitches # Med", "Pitches # High", "Pitches # Very High"])
torques_data = pd.DataFrame(data=torques, columns=["Torque Low", "Torque Med", "Torque High", "Torque Very High"])

pitches_data["Pitches Total"] = pitches_data.sum(axis=1)
torques_data["Torques Total"] = torques_data.sum(axis=1)

plt.figure()
plt.hist(torques_data.iloc[:,4], bins = 40)
plt.title('Histogram of Daily Total Workloads')
plt.xlabel('Workload (Nm)')
plt.ylabel('Number of Days')
plt.show()

work_ratio = []

for a in range(total_days):
    if a < 4:
        work_ratio.append(1)
    else: 
        temp = 4* torques_data.iloc[a,4] / (torques_data.iloc[a,4] + torques_data.iloc[a-1,4] + torques_data.iloc[a-2,4] + torques_data.iloc[a-3,4])
        if temp > 1.75:
            print('ALERT Day ', str(a), 'has a ACWR of: ', temp)
        work_ratio.append(temp)
    
torques_data.insert(5, "Work Ratios", work_ratio) 

plt.figure()
plt.plot(torques_data.iloc[:,5], '*')
plt.axhline(y=1.5, color = 'r', linestyle = '-')
plt.xlabel('Days')
plt.ylabel('ACWR')
plt.title('Daily Acute to Chronic Workload Ratios')
plt.show()

plt.figure()
plt.hist(torques_data.iloc[:,5], bins = 40)
plt.title('Histogram of Daily ACWRs')
plt.xlabel('ACWR')
plt.ylabel('Number of Days')
plt.show()

#Stack the data
bins = 40

plt.figure()
plt.hist([pitches_data.iloc[:,3],pitches_data.iloc[:,2],pitches_data.iloc[:,1], pitches_data.iloc[:,0]], bins, stacked=True, density=True, color = ['red', 'orange', 'green', 'blue'])
plt.xlabel('Daily Number of Throws')
plt.ylabel('Frequency')
plt.title('Number of Low, Med, High, and Very High Stress Throws')
plt.legend(['Very High', 'High', 'Med', 'Low'])

#plt.plot(pitches_data.iloc[:,0],'*', color = 'blue', alpha = .1)
#plt.plot(pitches_data.iloc[:,1],'*', color = 'green', alpha = .2)
#plt.plot(pitches_data.iloc[:,2],'*', color = 'orange', alpha = .5)
#plt.plot(pitches_data.iloc[:,3],'*', color = 'red', alpha = .7)
#plt.ylabel('Number of Throws')

