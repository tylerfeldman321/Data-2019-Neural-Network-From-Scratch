# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:31:57 2020

@author: TylerFeldman
"""


import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
np.set_printoptions(threshold=sys.maxsize)

class TwoLayerNeuralNetworkRegression:
    def __init__(self, input_nodes, hidden_neurons, output_nodes):

        stddev1 = 2 / np.sqrt(input_nodes + hidden_neurons)
        stddev2 = 2 / np.sqrt(hidden_neurons + output_nodes)
        self.w1 = np.random.randn(input_nodes, hidden_neurons) / np.sqrt(input_nodes/2)
        self.w2 = np.random.randn(hidden_neurons, output_nodes) / np.sqrt(input_nodes/2)
        self.input_nodes = input_nodes
        self.hidden_neurons = hidden_neurons
        self.output_nodes = output_nodes
    
    
    def get_forest_fire_data(self):
        
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
                        data[r-1][c+index] = 1
                        c += 12
                    elif (col == 3):
                        index = d_days.get(row[col])
                        data[r-1][c + index] = 1
                        c += 7 
                    else:
                        data[r-1][c] = float(row[col])
                        c += 1
                r+=1  
        
        
        training_data = data
        validation_data = np.zeros((78, 30))
        
        for i in range(78):
            index = np.random.randint(0, 517-i)
            validation_data[i] = training_data[index]
            training_data = np.delete(training_data, index, axis=0)
        
        training_data, validation_data = self.preprocess(training_data, validation_data)
        
        self.training_data, self.validation_data = training_data, validation_data


    def preprocess(self, training_data, validation_data):
        mean = np.mean(training_data, axis=0)
        std = np.std(training_data, axis=0)
        training_data -= mean
        training_data /= std
        validation_data -= mean
        validation_data /= std
        return training_data, validation_data
        
    def train(self, batch_size, learning_rate, max_epochs, tolerance, plot):
    
        training_loss_history = []
        validation_loss_history = []
        
        for i in range(max_epochs):
            
            # Shuffle the dataset
            np.random.shuffle(self.training_data) 
            
            # Get a mini batch
            mini_batch = self.training_data[:batch_size,:]
            x_batch = mini_batch[:, 0:self.input_nodes]
            y_batch = mini_batch[:, self.input_nodes:]
    
             # Get gradients based on mini batch
            avg_gradient_w1, avg_gradient_w2 = self.calcGradients(x_batch, y_batch)
            
            # Update weights
            self.w1 = self.w1 + (avg_gradient_w1 * learning_rate)
            self.w2 = self.w2 + (avg_gradient_w2 * learning_rate)
            
            # Get the average loss for each dataset, then record it for plotting and print it
            training_loss = self.avg_loss(self.training_data)
            validation_loss = self.avg_loss(self.validation_data)
            
            training_loss_history.append(training_loss)
            validation_loss_history.append(validation_loss)
            

            print("{}{:10.3f}{:10.3f}".format(i, training_loss, validation_loss))
        
            '''
            if ((i > 2) & (abs(validation_loss_history[i-1] - validation_loss) < tolerance)):
                break
            if (validation_loss < .01):
                break'''

        if (plot):
            self.plot_performance(training_loss_history, validation_loss_history, i+1)
        return training_loss, validation_loss        
    

    # Calculates average gradients for each of the weights in the batch
    def calcGradients(self, x_batch, y_batch):
        batch_size = x_batch.shape[0]
        gradient_w1_sum = np.zeros((self.input_nodes, self.hidden_neurons))
        gradient_w2_sum = np.zeros((self.hidden_neurons, self.output_nodes))
        
        # For each input, calculate the gradient and add it to the sum of gradients
        for i in range(batch_size):
            x = x_batch[i]
            y = y_batch[i]
            a1, z1, a2, y_hat, l2_loss = self.forwardPass(x, y)
            gradient_w1, gradient_w2 = self.backPropagation(x, y, a1, z1, a2, y_hat, l2_loss)
            gradient_w1_sum += gradient_w1
            gradient_w2_sum += gradient_w2
        
        # Get avg gradients by dividing by the batch size    
        avg_gradient_w1 = gradient_w1_sum / batch_size
        avg_gradient_w2 = gradient_w2_sum / batch_size
        
        # Return average gradients based on minibatch
        return avg_gradient_w1, avg_gradient_w2
    
    def forwardPass(self, x_vec, y_vec):
        x_vec = x_vec.reshape(1, -1)
        a1 = x_vec.dot(self.w1)
        z1 = self.relu(a1)
        a2 = z1.dot(self.w2)
        y_hat = self.leaky_relu(a2)
        #z2 = self.relu(a2)
        #y_hat = softmax(z2)
        l2_loss = self.l2Loss(y_vec, y_hat)
        return a1, z1, a2, y_hat, l2_loss
    
    def backPropagation(self, x, y, a1, z1, a2, y_hat, l2_loss):
        x = x.reshape(1, -1)
        
        delta_2 = ((y.reshape(1,1) - y_hat) * self.leaky_relu_derivative(a2))
        gradient_w2 = np.dot(z1.T, delta_2)        
        gradient_w1 = np.dot(x.T,  (np.dot(delta_2, (self.w2).T) * self.relu_derivative(a1)))
        
        return gradient_w1, gradient_w2
    
    def sigmoid(self, x):
        return (1.0 / (1.0 + np.exp(-x)))
    
    # Returns sigmoid local gradient based on the input to the function
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    def tanh_derivative(self, x):
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        dt = 1 - t**2
        return dt  
    
    
    def relu(self, x):
        x[x<=0] = 0
        x[x>0] = x[x>0]
        return x
    
    def relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def leaky_relu(self, x):
        x[x<=0] = 0.01 * x[x<=0]
        x[x>0] = x[x>0]
        return x
    
    def leaky_relu_derivative(self, x):
        x[x<=0] = 0.01
        x[x>0] = 1
        return x
        
    def l2Loss(self, y, y_hat):
        diff = abs(y_hat[0,0] - y[0])
        ret = 0.5 * (diff * diff)
        return ret
    
    def avg_loss(self, data):
        x_data = data[:, 0:self.input_nodes]
        y_data = data[:, self.input_nodes:]
        total_loss = 0
        for i in range(data.shape[0]):
            a1, z1, a2, y_hat, l2_loss = self.forwardPass(x_data[i], y_data[i])
            total_loss += l2_loss
        
        return total_loss / data.shape[0]
    
    def plot_performance(self, training, validation, epochs):
        plt.figure(figsize=(20,10))
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.plot(range(epochs), training, 'k', label="Training")
        plt.plot(range(epochs), validation, 'b', label="Validation")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    
    
    # Just as a sanity check to see that it correctly predicts the class for validation and traiing
    def predict_all(self):
        x_training = self.training_data[:, 0:self.input_nodes]
        y_training = self.training_data[:, self.input_nodes:]
        
        x_validation = self.validation_data[:, 0:self.input_nodes]
        y_validation = self.validation_data[:, self.input_nodes:]
        
        print("Training")
        for i in range(self.training_data.shape[0]):
            print("{:8.4f}{:8.4f}".format(self.predict(x_training[i]), y_training[i][0]))
        '''
        print("Validation")
        for i in range(self.validation_data.shape[0]):
            print("{:8.4f}{:8.4f}".format(self.predict(x_validation[i]), y_validation[i][0]))'''
        
    # Makes a prediction based on input
    def predict(self, x): 
        prediction = self.leaky_relu(self.relu((x.reshape(1,-1)).dot(self.w1)).dot(self.w2))
        return prediction[0,0]

    
    
    

    
if __name__ == "__main__":
    
    
    nn = TwoLayerNeuralNetworkRegression(input_nodes=29, hidden_neurons=50, output_nodes=1)
    nn.get_forest_fire_data()
    
    
    lr = .01
    nn.train(batch_size=50, learning_rate=lr, max_epochs=1000, tolerance=lr/10000000000, plot=True)
    nn.predict_all()

    
    
    
    
    
    
    
    
    
