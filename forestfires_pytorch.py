# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:05:40 2020

@author: TylerFeldman
"""

import numpy as np
import torch.utils.data
import csv


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
                data[r-1][c+index] = float(1)
                c += 12
            elif (col == 3):
                index = d_days.get(row[col])
                data[r-1][c + index] = float(1)
                c += 7 
            else:
                data[r-1][c] = float(row[col])
                c += 1
        r+=1  


np.random.seed(0)
np.random.shuffle(data)

training_data = data[0:428, :]

mean = np.mean(training_data, axis=0)
std = np.std(training_data, axis=0)
data -= mean
data /= std + 1e-000001

x_data = data[:, :29]
y_data = data[:, 29]
y_data = y_data.reshape(-1, 1)

x_train = torch.tensor(x_data[:428, :], dtype=torch.float32)
y_train = torch.tensor(y_data[:428, :], dtype=torch.float).view(-1)

x_test = torch.tensor(x_data[428:, :], dtype=torch.float32)
y_test = torch.tensor(y_data[428:, :], dtype=torch.float).view(-1)

forestfire_train = torch.utils.data.TensorDataset(x_train, y_train)
forestfire_test = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(forestfire_train, batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(forestfire_test, batch_size=89, shuffle=False)


import torch.nn as nn

class ForestFireMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(29, 50)
        self.linear2 = nn.Linear(50, 1)
        self.LeakyReLU = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        h = self.linear1(x).clamp(min=0)
        #y_hat = self.linear2(h)
        y_hat = self.LeakyReLU(self.linear2(h))
        return y_hat
    
    
import torch.optim


model = ForestFireMLP()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(500):
    
    for x, y in train_loader:
        optimizer.zero_grad()
        
        x = x.view(-1, 29)
        y_hat = model(x)
        
        y = y.view(-1, 1)
        loss = criterion(y_hat, y)
        loss.backward()
        
        test_loss = 0
        for x, y in test_loader:
            x = x.view(-1, 29)
            y_hat = model(x)
            test_loss += criterion(y_hat, y)
        
        
        optimizer.step()
        
    if epoch % 25 == 0:
        print("Epoch: {}, Train Loss: {:.3f}, Test Loss: {:.3f}".format(epoch, loss, test_loss))
        

with torch.no_grad():
    for x, y in train_loader:
        
        x = x.view(-1, 29)
        y_hat = model(x)
        
        for i in range(y.shape[0]):
            print(y_hat[i], y[i])
        

    

        
