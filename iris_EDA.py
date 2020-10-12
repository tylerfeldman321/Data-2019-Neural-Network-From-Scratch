# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:58:15 2020

@author: TylerFeldman
"""

import numpy as np
import matplotlib.pyplot as plt


    
data = np.zeros((150,7))
sepal_length = []
sepal_width = []
petal_length = []
petal_width = [] 
classes = []
      
d = {'Iris-setosa\n':0,'Iris-versicolor\n':1,'Iris-virginica\n': 2}
        
with open('iris.DATA', 'r') as reader:
    for row in range (0, 150):
        line = reader.readline();
        line_list = line.split(',')
        flower_name = line_list[4]
        classes.append(flower_name)
        class_index = d.get(flower_name)
        data[row][class_index+4] = 1;
        for col in range(4):
            data[row][col] = float(line_list[col])
            if (col == 0):
                sepal_length.append(data[row][col])
            elif (col == 1):
                sepal_width.append(data[row][col])
            elif (col == 2):
                petal_length.append(data[row][col])
            elif (col == 3):
                petal_width.append(data[row][col])

plt.clf()
plt.figure(1)
plt.scatter(sepal_length[:50], sepal_width[:50], label="Iris Setosa")
plt.scatter(sepal_length[50:100], sepal_width[50:100], label="Iris Versicolor")
plt.scatter(sepal_length[100:], sepal_width[100:], label="Iris Virginica")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(loc="best")


plt.figure(2)
plt.scatter(petal_length[:50], petal_width[:50], label="Iris Setosa")
plt.scatter(petal_length[50:100], petal_width[50:100], label="Iris Versicolor")
plt.scatter(petal_length[100:], petal_width[100:], label="Iris Virginica")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend(loc="best")
    
    
    
    