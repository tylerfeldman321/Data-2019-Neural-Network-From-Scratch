# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:58:15 2020

@author: TylerFeldman
"""

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNeuralNetworkClassificationCustom:
    def __init__(self, input_nodes, hidden_neurons, output_nodes, activation_function, leak=0):

        self.w1 = np.random.randn(input_nodes, hidden_neurons) / np.sqrt(input_nodes/2)
        self.w2 = np.random.randn(hidden_neurons, output_nodes) / np.sqrt(input_nodes/2)
        self.input_nodes = input_nodes
        self.hidden_neurons = hidden_neurons
        self.output_nodes = output_nodes
        self.activation_function = activation_function
        self.leak = leak
    
    def get_iris_data(self):
    
        data = np.zeros((150,7))
        
        d = {'Iris-setosa\n':0,'Iris-versicolor\n':1,'Iris-virginica\n': 2}
        
        with open('iris.DATA', 'r') as reader:
            for row in range (0, 150):
                line = reader.readline();
                line_list = line.split(',')
                flower_name = line_list[4]
                class_index = d.get(flower_name)
                data[row][class_index+4] = 1;
                for col in range(4):
                    data[row][col] = float(line_list[col])
            
        training_data = data
        validation_data = np.zeros((22, 7))
        
        for i in range(22):
            index = np.random.randint(0, 150-i)
            validation_data[i] = training_data[index]
            training_data = np.delete(training_data, index, axis=0)
        
        self.m = training_data.shape[0]
        self.training_data, self.validation_data = training_data, validation_data

       
    def train(self, batch_size, learning_rate, max_epochs, tolerance, rho, lambd, print_loss=True, plot=True):
    
        self.lambd = lambd
        
        rho = 0
        vx_w1 = 0
        vx_w2 = 0
        
        training_loss_history = []
        validation_loss_history = []
        
        for i in range(max_epochs):
            
            if (rho):
                if (i < 50):
                    rho = 0.5 + (i)/50 * 0.4
                else:
                    rho = 0.9
            
            # Shuffle the dataset
            np.random.shuffle(self.training_data) 
            
            # Get a mini batch
            mini_batch = self.training_data[:batch_size,:]
            x_batch = mini_batch[:, 0:self.input_nodes]
            y_batch = mini_batch[:, self.input_nodes:]
    
             # Get gradients based on mini batch
            avg_gradient_w1, avg_gradient_w2 = self.calcGradients(x_batch, y_batch)
            
            # Update weights using momentum
            vx_w1 = rho * vx_w1 + avg_gradient_w1
            vx_w2 = rho * vx_w2 + avg_gradient_w2
            
            self.w1 += learning_rate * vx_w1
            self.w2 += learning_rate * vx_w2
            
            
            # Get the average loss for each dataset, then record it for plotting and print it
            training_loss = self.avg_loss(self.training_data)
            validation_loss = self.avg_loss(self.validation_data)
                
            training_loss_history.append(training_loss)
            validation_loss_history.append(validation_loss)
               
            if (print_loss):
                print("{:10}{:10.5f}{:10.5f}".format(i+1, training_loss, validation_loss))
            
            if ((i > 2) & (abs(validation_loss_history[i-1] - validation_loss) < tolerance)):
                break
            
            
        if (plot):
            self.plot_performance(training_loss_history, validation_loss_history, i+1)
        return training_loss, validation_loss

    # Calculates average gradients for each of the weights
    def calcGradients(self, x_batch, y_batch):
        batch_size = x_batch.shape[0]
        gradient_w1_sum = np.zeros((self.input_nodes, self.hidden_neurons))
        gradient_w2_sum = np.zeros((self.hidden_neurons, self.output_nodes))
        
        # For each input, calculate the gradient and add it to the sum of gradients
        for i in range(batch_size):
            x = x_batch[i]
            y = y_batch[i]
            a1, z1, a2, categorical_cross_entropy_loss_with_L2_reg = self.forwardPass(x, y)
            gradient_w1, gradient_w2 = self.back_propagation(x, y, a1, z1, a2, categorical_cross_entropy_loss_with_L2_reg)
            gradient_w1_sum -= gradient_w1
            gradient_w2_sum -= gradient_w2
        
        # Get avg gradients by dividing by the batch size    
        avg_gradient_w1 = gradient_w1_sum / batch_size
        avg_gradient_w2 = gradient_w2_sum / batch_size
        
        # Return average gradients based on minibatch
        return avg_gradient_w1, avg_gradient_w2
    
    def forwardPass(self, x_vec, y_vec):
        x_vec = x_vec.reshape(1, -1)
        a1 = x_vec.dot(self.w1)
        z1 = self.activation(a1)
        a2 = z1.dot(self.w2)
        categorical_cross_entropy_loss_with_L2_reg = self.categorical_cross_entropy_loss_with_L2_reg(y_vec, a2)
        return a1, z1, a2, categorical_cross_entropy_loss_with_L2_reg
    

    def back_propagation(self, x, y, a1, z1, a2, categorical_cross_entropy_loss_with_L2_reg):
        x = x.reshape(1, -1)

        delta_2 = (self.categorical_cross_entropy_loss_gradient(y, a2))
        gradient_w2 = np.dot(z1.T, delta_2) + ((self.lambd / self.m) * self.w2)
        gradient_w1 = np.dot(x.T,  (np.dot(delta_2, (self.w2).T) * self.activation_derivative(a1))) + ((self.lambd / self.m) * self.w1)
        
        return gradient_w1, gradient_w2
    
    def activation(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'relu':
            return self.relu(x)
        elif self.activation_function == 'tanh':
            return self.tanh(x)
        
    def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_function == 'relu':
            return self.relu_derivative(x)
        elif self.activation_function == 'tanh':
            return self.tanh_derivative(x)
    
    def sigmoid(self, x):
        return (1.0 / (1.0 + np.exp(-x)))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def relu(self, x):
        x[x<=0] = self.leak * x[x<=0]
        x[x>0] = x[x>0]
        return x
    
    def relu_derivative(self, x):
        x[x<=0] = self.leak
        x[x>0] = 1
        return x
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    def tanh_derivative(self, x):
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        dt = 1 - t**2
        return dt  
    
    def categorical_cross_entropy_loss(self, y, a2):
        correct_class_index = np.argmax(y)
        true_class_probability = self.individual_softmax_probability(a2, correct_class_index)
        CE = -np.log(true_class_probability)
        return CE
    
    def categorical_cross_entropy_loss_with_L2_reg(self, y, a2):
        correct_class_index = np.argmax(y)
        true_class_probability = self.individual_softmax_probability(a2, correct_class_index)
        CE = -np.log(true_class_probability)
        L2_regularization_loss = ((np.sum(np.square(self.w1))) + np.sum(np.square(self.w2))) * (self.lambd / (2 * self.m))
        return CE + L2_regularization_loss

    def categorical_cross_entropy_loss_gradient(self, y, a2):
        ret = np.zeros((a2.shape[0], a2.shape[1]))
        correct_class_index = np.argmax(y)
        for i in range(a2.shape[1]):
            if (i == correct_class_index):
                ret[0,i] = self.individual_softmax_probability(a2, i) -1
            else:
                ret[0,i] = self.individual_softmax_probability(a2, i)
            
        return ret    
        # should be 1x3 jacobian
    
    def individual_softmax_probability(self, a2, index):
        den = 0
        for i in range(a2.shape[1]):
            den += np.exp(a2[0,i])
        individual_class_probability = np.exp(a2[0,index]) / den
        return individual_class_probability      
    
    # This function uses categorical_cross_entropy_loss so that the printed results can be compared with values from not using L2 regularization 
    def avg_loss(self, data):
        x_data = data[:, 0:self.input_nodes]
        y_data = data[:, self.input_nodes:]
        total_loss = 0
        for i in range(data.shape[0]):
            a1, z1, a2, categorical_cross_entropy_loss_with_L2_reg = self.forwardPass(x_data[i], y_data[i])
            categorical_cross_entropy_loss = self.categorical_cross_entropy_loss(y_data[i], a2)
            total_loss += categorical_cross_entropy_loss
        
        return total_loss / data.shape[0]
    
    def plot_performance(self, training, validation, epochs):

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
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
        
        training_count=0
        validation_count=0

        for i in range(self.training_data.shape[0]):
            if (self.equal(self.classify(x_training[i]), y_training[i]) == True):
                training_count+=1
        
        for i in range(self.validation_data.shape[0]):
            if (self.equal(self.classify(x_validation[i]), y_validation[i]) == True):
                validation_count+=1
        
        print("Training: {}/{}".format(training_count, self.training_data.shape[0]))
        print("Validation: {}/{}".format(validation_count, self.validation_data.shape[0]))
        
    # Makes a classification based on input
    def classify(self, x): 
        probability_distribution = self.softmax(self.activation((x.reshape(1,-1)).dot(self.w1)).dot(self.w2))
        classification = np.zeros(3)
        index = np.argmax(probability_distribution)
        classification[index] = 1
        return classification
    
    # Will only be used for making predictions, not actually for the forward pass or back prop 
    # since the softmax is combined with CE in the categorical cross entropy loss
    def softmax(self, a2):
        ret = np.zeros((1, a2.shape[1]))
        den = 0
        for i in range(a2.shape[1]):
            den += np.exp(a2[0,i])
            
        for i in range(a2.shape[1]):
            ret[0][i] = (np.exp(a2[0,i])/den)
        
        return ret
    
    # Helper function for predict_all
    def equal(self, y_hat, y):
        for i in range(y_hat.shape[0]):
            if (y_hat[i] != y[i]):
                return False
        return True

    
if __name__ == "__main__":
    np.random.seed(0)
    nn = TwoLayerNeuralNetworkClassificationCustom(input_nodes=4, hidden_neurons=10, output_nodes=3, activation_function = 'relu', leak=0)
    nn.get_iris_data()
    
    lr = .01
    nn.train(batch_size=100, learning_rate=lr, max_epochs=500, tolerance=lr/1000000, rho=True, lambd=0.01, print_loss=True, plot=True)
    nn.predict_all()

    
    
    
    
    
    
    
    
    