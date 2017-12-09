#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:57:20 2017

@author: deltau

network.py
"""

import numpy as np
import json
import random
import sys

def sigmoid(z):
    a = 1.0/(1.0+np.exp(-z))
    return a

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

class CrossEntropy(object):
    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z,a,y):
        return (a-y)
    
class QuatraticMethod(object):
    @staticmethod
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z,a,y):
        return (a-y)*sigmoid_prime(z)
        
class Network(object):    #initialize a neural network
    def __init__(self, sizes, cost=CrossEntropy):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.defaultWeightInitializer()
        self.cost = cost
        
    def defaultWeightInitializer(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        
    def largeWeightInitializer(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        
    def feedForward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, evaluation_data=None, lmbda = 0.0, monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False, monitor_training_cost=False, monitor_training_accuracy=False):
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost,evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in range(0,n,mini_batch_size):
                mini_batches.append(training_data[k:k+mini_batch_size])
            #mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta,lmbda,n)
            print ("Epoch %s training complete"% j)
            if monitor_training_cost:
                cost = 0
                cost = self.total_cost(training_data,lmbda)
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = 0
                accuracy = self.accuracy(training_data,convert=True)
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {}/{}".format(accuracy,n))
            if monitor_evaluation_cost:
                cost = 0
                cost = self.total_cost(evaluation_data,lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = 0
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print ("Accuracy on evaluation data: {}/{}".format(accuracy,n_data))
            print
        return evaluation_cost, evaluation_accuracy,training_cost, training_accuracy
                
    def update_mini_batch(self,mini_batch,eta,lmbda,n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            del_nabla_b,del_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, del_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, del_nabla_w)]
            
        #self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        #self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
        self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        x = np.array(x)
        #print(x.shape)
        y = np.array(y)
        #print(y.shape)
        activation = x
        #print(x.shape)
        activations = [x]
        zs = []
        count = 0
        for b, w in zip(self.biases, self.weights):
            #print(str(count))
            count += 1
            #print("w ",w.shape)
            activation = activation.reshape(-1,1)
            #print("a ",activation.shape)
            #print("b ",b.shape)
            z = np.dot(w, activation) + b
            #print ("z ",z.shape)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        delta = (self.cost).delta(zs[-1],activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2,self.num_layers):
            z = zs[-l]
            #print("z: ",z.shape)
            sp = sigmoid_prime(z)
            #print (z.shape)
            #print("weights[-l+1].T: ",self.weights[-l+1].transpose().shape)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            #print("delta: ",delta.shape)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose().reshape(1,-1))
        return (nabla_b, nabla_w)
    
    def accuracy(self,data,convert=False):
        if convert:
            results = [(np.argmax(self.feedForward(x)), np.argmax(y)) for (x,y) in data]
        else:
            results = [(np.argmax(self.feedForward(x)), y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results)
    
    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        l = len(data)
        for x,y in data:
            a = self.feedForward(x)
            if convert:
                y = vectorized_result(y)
                cost += self.cost.fn(a,y)/l
        cost += 0.5*(lmbda/l)*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def save(self, filename):
        data = {"sizes": self.sizes, "weights":[w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases], "cost":str(self.cost.__name__)}
        f.open("filename", f)
        f.close()
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
            
        
        
        
        
        
        
        
        
        
        
        