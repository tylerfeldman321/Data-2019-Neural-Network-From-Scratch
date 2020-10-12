# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:31:57 2020

@author: TylerFeldman
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:52:41 2020

@author: TylerFeldman
"""


import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
np.set_printoptions(threshold=sys.maxsize)

    
def preprocess(training_data, validation_data):
    mean = np.mean(training_data, axis=0)
    std = np.std(training_data, axis=0)
    training_data -= mean
    training_data /= std
    validation_data -= mean
    validation_data /= std
    return training_data, validation_data  
  
'''
df = pd.read_csv('forestfires.csv')
print(df.head())
df.info()


print(df['area'].describe())
plt.figure(figsize=(6, 5))
sns.distplot(df['area'], color='r', bins=100, hist_kws={'alpha': 0.5});



df_num = df.select_dtypes(include = ['float64', 'int64'])

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


df_num_corr = df_num.corr()['area'][:] # -1 because the latest row is SalePrice
golden_features_list = df_num_corr[abs(df_num_corr) > 0.0].sort_values(ascending=False)
print("How the Inputs Correlate with area\n", golden_features_list)



for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['area'])'''



areas_days = np.zeros(7)
areas_months = np.zeros(12)

      
data = np.zeros((517, 30))
        
d_months = {'jan':0, 'feb':1, 'mar':2, 'apr':3, 'may':4, 'jun':5, 'jul':6, 'aug':7, 'sep':8, 'oct':9, 'nov':10, 'dec':11}
d_days = {'mon':0, 'tue':1, 'wed':2, 'thu':3, 'fri':4, 'sat':5, 'sun':6}
        
with open('forestfires.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    r = 0
    for row in readCSV:
        if (r == 0):
            r = 1
            continue
        c = 0
        for col in range(13):
            if (col == 2):
                index = d_months.get(row[col])
                areas_months[index] += float(row[12])
                data[r-1][c+index] = 1
                c += 12
            elif (col == 3):
                index = d_days.get(row[col])
                areas_days[index] += float(row[12])
                data[r-1][c + index] = 1
                c += 7 
            else:
                data[r-1][c] = float(row[col])
                c += 1
        r+=1  
        
        
print(areas_days)
print(areas_months)

days = list(d_days.keys())
months = list(d_months.keys())
   
plt.clf()
barlist = plt.bar(days, areas_days)
plt.xlabel("Day")
plt.ylabel("Area (ha)")
barlist[5].set_color('m')

plt.figure(2)
barlist2 = plt.bar(months, areas_months)
barlist2[7].set_color('m')
barlist2[8].set_color('m')
plt.xlabel("Month")
plt.ylabel("Area (ha)")

  
        
    

    
    
    
    
    
    
    
    
    
