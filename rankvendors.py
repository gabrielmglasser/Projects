
"""
@author: gglas

Develop a model that ranks the hot dog vendor's sales abilities
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

vendors = pd.read_csv(r'C:\Users\gglas\Documents\Career Docs\Applications_Interviews\Mets\Q3_citi_vendors.csv')

#take avg number of values for each input: vendor, game, day, and section for preliminary analysis
vendor_group_m = vendors.groupby(by = 'vendor').mean() 
game_group_m = vendors.groupby(by = 'game').mean()
day_group_m = vendors.groupby(by = 'day').mean()
section_group_m = vendors.groupby(by = 'section').mean()

vendor_groups_c = vendors.groupby(by = 'vendor').count() #counts number of data points for each vendor

num_vendor = len(vendor_group_m)
num_game = len(game_group_m)
num_day = len(day_group_m)
num_section = len(section_group_m)

first_game = first_day = first_section = first_vendor = 1

colorlist = [(0,'darkorange'),(1,'darkblue')]
rvb = mcolors.LinearSegmentedColormap.from_list("", colorlist)

#Visualize data based on average number of hot dogs sold for each:
#Each Vendor
plt.figure()
plt.bar(np.arange(1,num_vendor+1),vendor_group_m['hot_dogs_sold'],color=rvb(np.arange(1,num_vendor+1)/num_vendor))
plt.xlabel('Vendors'),plt.ylabel('Hot Dogs Sold'),plt.title('Avg Number of Hot Dogs Sold by Vendor')
plt.show()
#We see that avg number of hot dogs sold is different for each vendor
#Also running a statistical analysis: 
k2_, p = stats.normaltest(vendor_group_m['hot_dogs_sold']) #we see that p < .05, 
# meaning the data is normally distributed and we can perform a t-test
t_test = stats.ttest_ind(np.arange(1,num_vendor+1),vendor_group_m['hot_dogs_sold']) #we see that p< .05,
# meaning we reject the null hypothesis and the data is statistically significant between vendors and avg number of hot dogs sold

#Each game
plt.figure()
plt.bar(np.arange(1,num_game+1),game_group_m['hot_dogs_sold'],color=rvb(np.arange(1,num_game+1)/num_game))
plt.xlabel('Games'),plt.ylabel('Hot Dogs Sold'),plt.title('Avg Number of Dogs Sold During Each Game')
plt.show()
#We see that avg number of hot dogs sold is different for each game 

#Each day of the week
plt.figure()
plt.bar(np.arange(1,num_day+1),day_group_m['hot_dogs_sold'],color=rvb(np.arange(1,num_day+1)/num_day))
plt.xlabel('Days'),plt.ylabel('Hot Dogs Sold'),plt.title('Avg Number of Dogs Sold During Each Day')
plt.show()
#We see that avg number of hot dogs sold is different for each day

#Each section
plt.figure()
plt.bar(np.arange(1,num_section+1),section_group_m['hot_dogs_sold'],color=rvb(np.arange(1,num_section+1)/num_section))
plt.xlabel('Sections'),plt.ylabel('Hot Dogs Sold'),plt.title('Avg Number of Dogs Sold In Each Section')
plt.show()
#We see that avg number of hot dogs sold is different for each section

#Create a DataFrame vendors_ranks which has columns bygame, byday, bysection
#It will group by the columns game, day, and section and then rank by hot_dogs_sold 
#And returns the percent of the number of hot_dogs_sold for each condition
vendors_ranks = pd.DataFrame(vendors['vendor'])
vendors_ranks['bygame'] = vendors.groupby(by = 'game')['hot_dogs_sold'].rank(ascending = True, pct = True)
vendors_ranks['byday'] = vendors.groupby(by = 'day')['hot_dogs_sold'].rank(ascending = True, pct = True)
vendors_ranks['bysection'] = vendors.groupby(by = 'section')['hot_dogs_sold'].rank(ascending = True, pct = True)

#Vendors_group takes the vendors_ranked data and then groups by vendor and takes the mean for each column (variable)
vendors_grouped = pd.DataFrame()
vendors_grouped = vendors_ranks.groupby(by = 'vendor').mean()
#means takes the mean of the row percents for each vendor 
vendors_grouped['means'] = vendors_grouped.mean(axis=1)
#overall_ranks ranks the vendors by the means values
vendors_grouped['overall_ranks'] = vendors_grouped['means'].rank(ascending = False)
vendors_grouped['percent_ranks'] = vendors_grouped['overall_ranks'].rank(ascending = False, pct = True)


#print the results
print (vendors_grouped.sort_values(by = 'overall_ranks',ascending=True))  

#Create a figure for Overall Percent Ranks for each hot dog vendor
plt.figure()
plt.bar(np.arange(1,num_vendor+1),vendors_grouped['percent_ranks'],color=rvb(np.arange(1,num_vendor+1)/num_vendor))
plt.xlabel('Vendors'),plt.ylabel('Percent Ranks'),plt.title('Overall Percent Ranks for Each Hot Dog Vendor')  
plt.show()

#Output dataframe to file "Q2_pitches_test_RESULTS.csv"
vendors_grouped.to_csv(r'C:\Users\gglas\Documents\Career Docs\Applications_Interviews\Mets\Q3_citi_vendors_RESULTS.csv',index = False)
    