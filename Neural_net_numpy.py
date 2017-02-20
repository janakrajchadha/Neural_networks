# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23:11:17 2017

@author: Janak
"""
#A simple program for a 3 layer neural network using only numpy as a dependency
from numpy import exp, array, random, dot

#class to create neuron layers for the network
class NeuronLayer():
    def __init__(self,number_of_neurons,number_of_inputs):
        self.synaptic_weights = 2 * random.random((number_of_inputs,number_of_neurons)) - 1

#class for the neural network
class Network():
  def __init__(self, layer1, layer2, layer3):
    self.layer1=layer1
    self.layer2=layer2
    self.layer3=layer3

	#Sigmoid function which normalises the input to a value between 0 and 1
  def __sigmoid(self,x):
    return 1 / (1+exp(-x))

	#Derivative of the sigmoid function
  def __der_sigmoid(self,x):
    return x*(1-x)

	#Training the neural network
	#Running it multiple times and adjusting the weights everytime
  def train(self, training_inputs, training_outputs, number_of_iterations):
    for iterator in range(number_of_iterations):

			#Passing the inputs through the Neural network and getting the outputs
      output_layer_1,output_layer_2,output_layer_3= self.think(training_inputs)

			#Calculating the error and delta for each layer
			#First for the output layer or layer 3
      layer_3_error = training_outputs - output_layer_3 
      layer_3_delta = layer_3_error * self.__der_sigmoid(output_layer_3)

			#Backpropagating through the layers and calculating error and delta for each
      layer_2_error = layer_3_delta.dot(self.layer3.synaptic_weights.T) 
      layer_2_delta = layer_2_error * self.__der_sigmoid(output_layer_2)

      layer_1_error = layer_2_delta.dot(self.layer2.synaptic_weights.T) 
      layer_1_delta = layer_1_error * self.__der_sigmoid(output_layer_1)

			#Calculation of layer adjustments
      layer_3_adjustment= output_layer_2.T.dot(layer_3_delta)
      layer_2_adjustment= output_layer_1.T.dot(layer_2_delta)
      layer_1_adjustment= training_inputs.T.dot(layer_1_delta)

			#Adjusting the weights
      self.layer3.synaptic_weights+=layer_3_adjustment
      self.layer2.synaptic_weights+=layer_2_adjustment
      self.layer1.synaptic_weights+=layer_1_adjustment

	#This is how the neural network thinks or in other words processes the inputs
	#Each input passes through a network of sigmoid transformations before reaching the output
  def think(self, training_inputs):
    output_layer_1= self.__sigmoid(dot(training_inputs,self.layer1.synaptic_weights))
    output_layer_2= self.__sigmoid(dot(output_layer_1, self.layer2.synaptic_weights))
    output_layer_3= self.__sigmoid(dot(output_layer_2, self.layer3.synaptic_weights))
    return output_layer_1,output_layer_2,output_layer_3
  
  #Printing weights   
  def print_weights(self):
    print("Layer 1:")
    print(self.layer1.synaptic_weights)
    print("Layer 2:")
    print(self.layer2.synaptic_weights)
    print("Layer 3:")
    print(self.layer2.synaptic_weights)



if __name__ == "__main__":

#Seeding the random number generator
  random.seed(1)

# Creating layer 1 
  layer1 = NeuronLayer(3, 3)

# Creating layer 2
  layer2 = NeuronLayer(4, 3)

# Creating layer 3
  layer3 = NeuronLayer(1, 4)


# Combining the layers to create a neural network
  neural_network = Network(layer1, layer2,layer3)

  print("Stage 1) Random starting synaptic weights: ")
  neural_network.print_weights()

# The training set - 5 examples, each consisting of 3 inputs and 1 output
  training_set_inputs = array([[0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]])  
  training_set_outputs = array([[0, 1, 1, 1, 0]]).T

# Training the neural network using the training inputs
  neural_network.train(training_set_inputs, training_set_outputs, 100000)

  print("Stage 2) New synaptic weights after training: ")
  neural_network.print_weights()

# Testing the neural network with a new situation.
  print("Stage 3) Considering a new situation [0, 1, 1] -> ")
  hidden_state1,hidden_state2, output = neural_network.think(array([1, 1, 0]))
  print(output)










