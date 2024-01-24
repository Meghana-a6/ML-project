# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:44:26 2023

@author: aeligeti greeshma
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

salaryData = pd.read_csv('EXPvsSALARY.csv')

def gradeint_descent(x,y):
    m_curr = b_curr = y_predicted = 0
    iterations = 100000
    n = len(x)
    learning_rate = 0.001
    
    for i in range(iterations):
        cost_i_1 = (1/n)* sum([val**2 for val in (y- y_predicted)])
        y_predicted = m_curr*x +b_curr
        cost_i = (1/n)* sum([val**2 for val in (y- y_predicted)])
        m_der = -(2/n)*sum(x*(y - y_predicted))
        b_der = -(2/n)*sum(y - y_predicted)
        m_curr = m_curr - learning_rate *m_der    
        b_curr = b_curr - learning_rate*b_der
        print ("m {}, b {}, cost {} at iteration {}".format(m_curr,b_curr,cost_i,i))
        if cost_i > cost_i_1 :
            break
    print ("Bias(b): {} \n Slope(m): {} ".format(b_curr, m_curr))      
    plt.figure(1);  
    plt.scatter(x,y)
    plt.suptitle('Experience Vs Salary')
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.grid(1,which='both')
    plt.axis('tight')
    plt.show()
    plt.plot(x, m_curr*x+b_curr)
        
#x = np.array([1,2,3,4,5])
#y = np.array([5,7,9,11,13])  
x = salaryData.iloc[:, 0]
y = salaryData.iloc[:, 1]


gradeint_descent(x, y)      



