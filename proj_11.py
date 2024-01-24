# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:15:54 2023

@author: aeligeti greeshma
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



BosData = pd.read_csv('BostonHousing.csv')
X = BosData.iloc[:,0:13]
y = BosData.iloc[:,13] # MEDV: Median value of owner-occupied homes in $1000s

ss = StandardScaler()
X = ss.fit_transform(X) # random state can be mentioned

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2)

model = Sequential()

model.add(Dense(20, input_dim = 13, activation="relu"))

model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

history = model.fit(Xtrain, ytrain, epochs=150, batch_size=10)

ypred = model.predict(Xtest)

ypred = ypred[:,0]

error = np.sum(np.abs(ytest-ypred))/np.sum(np.abs(ytest))*100
print('Prediction Error is',error,'%')