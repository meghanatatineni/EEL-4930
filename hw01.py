# -*- coding: utf-8 -*-
"""
File:   hw01.py
Author: Meghana Tatineni 
Date: 9/13/18
Desc: Machine Learning HW #1   
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """

def generateUniformData(N, l, u, gVar):
	'''generateUniformData(N, l, u, gVar): Generate N uniformly spaced data points 
    in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
	# x = np.random.uniform(l,u,N)
	step = (u-l)/(N);
	x = np.arange(l+step/2,u+step/2,step)
	e = np.random.normal(0,gVar,N)
	t = np.sinc(x) + e
	return x,t

def plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot training data

    #add title, legend and axes labels
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)
        
def fitdata(x,t,M):
    '''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t)''' 
    #This needs to be filled in
    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X)@X.T@t
    return w
        

""" ======================  Variable Declaration ========================== """

l = 0 #lower bound on x
u = 10 #upper bound on x
N = 15 #number of samples to generate
gVar = .001 #variance of error distribution
M =  15 #regression model order
""" =======================  Generate Training Data ======================= """
data_uniform  = np.array(generateUniformData(N, l, u, gVar)).T

x1 = data_uniform[:,0]
t1 = data_uniform[:,1]

x2 = np.arange(l,u,0.1)  #get equally spaced points in the xrange
t2 = np.sinc(x2) #compute the true function value
    
""" ========================  Train the Model ============================= """

w = fitdata(x1,t1,M)
print(w) 
x3 = np.arange(l,u,.01)  #get equally spaced points in the xrange
X = np.array([x3**m for m in range(w.size)]).T
t3 = X@w #compute the predicted value
actual_train = np.sinc(x3)
plotData(x1,t1,x2,t2,x3,t3,['Training Data', 'True Function', 'Estimated\nPolynomial'])

""" ======================== Generate Test Data =========================== """
N=50
data_uniform_test  = np.array(generateUniformData(N, l, u, gVar)).T

x1_test = data_uniform_test[:,0]
t1_test = data_uniform_test[:,1]

x2_test = np.arange(l,u,0.001)  #get equally spaced points in the xrange
t2_test = np.sinc(x2_test)

x3_test = np.arange(l,u,0.1)  #get equally spaced points in the xrange
actual_test = np.sinc(x3_test)

X_test = np.array([x3_test**m for m in range(w.size)]).T
t3_test = X_test@w

plotData(x1_test,t1_test,x2_test,t2_test,x3_test,t3_test,['Test Data', 'True Function', 'Estimated\nPolynomial'])
""" ========================  Test the Model ============================== """

#creating list of erms errors for training and test data 
model=[]
error_train=[]
error_test=[]
for M in range(0,15):
    model.append(M)
    w= fitdata(x1,t1,M)
    X = np.array([x3**m for m in range(w.size)]).T
    t3 = X@w 
    actual_train = np.sinc(x3)
    erms_train = np.linalg.norm(t3 - actual_train) / np.sqrt(len(actual_train))
    error_train.append(erms_train)
    X_test = np.array([x3_test**m for m in range(w.size)]).T
    t3_test = X_test@w
    actual_test = np.sinc(x3_test)
    erms_test = np.linalg.norm(t3_test - actual_test) / np.sqrt(len(actual_test))
    error_test.append(erms_test)
#Plotting Test vs. Training erms error
p1 = plt.plot(model, error_train,'-bo',label='Training') 
p2 = plt.plot(model, error_test,'-ro',label='Test')
plt.ylabel('ERMS') 
plt.xlabel('Model Order')
plt.legend()

"""
===============================================================================
===============================================================================
============================ Question 2 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Variable Declaration ========================== """

#True distribution mean and variance 
trueMu = 4
trueVar = 2

#Initial prior distribution mean and variance (You should change these parameters to see how they affect the ML and MAP solutions)
priorMu = 10
priorVar = 3
numDraws = 50 #Number of draws from the true distribution


"""========================== Plot the true distribution =================="""
#plot true Gaussian function
step = 0.01
l = -20
u = 20
x = np.arange(l+step/2,u+step/2,step)
plt.figure(0)
p1 = plt.plot(x, norm(trueMu,trueVar).pdf(x), color='b')
plt.title('Known "True" Distribution')

"""========================= Perform ML and MAP Estimates =================="""
#Calculate posterior and update prior for the given number of draws

data=[]
draws=[]
MLE=[]
MAP=[]

#loop for MLE and MAP estimates 
for draw in range(numDraws):
    data.append(np.random.normal(trueMu,trueVar,1)[0])
    print(data)
    print('Frequentist/MLE Estimate of Mean:' +str(sum(data)/len(data)))
    draws.append(draw)
    MLE.append((sum(data)/len(data)))
    #calculating MAP 
    a = ((priorVar ** 2)/(((len(data)*(priorVar ** 2)))+(trueVar ** 2)))
    c = ((priorMu*(trueVar ** 2))/((len(data)*(priorVar**2)+(trueVar**2))))
    print('Bayesian/MAP Estimate of Mean:' + str((a*(sum(data)))+c))
    MAP.append((a*(sum(data)))+c)
plt.close('all')

#Plotting MAP vs. MLE estimates for mean
p1 = plt.plot(draws, MLE,'-b',label='MLE') 
p2 = plt.plot(draws, MAP,'-r',label='MAP')
plt.ylabel('estimate of mean') 
plt.xlabel('sample size')
plt.title('True Mean:4 and True Variance:2 Prior Mean = 10 and Prior Variance = 3')
plt.legend()




