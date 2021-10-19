import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# load walmart data 

df=pd.read_csv('sales_train_validation.csv')
    
fooddf=df[df['cat_id']=='FOODS']
hobbiesdf=df[df['cat_id']=='HOBBIES']
housedf=df[df['cat_id']=='HOUSEHOLD']

food=np.zeros((1,fooddf.shape[1]-6))
hobbies=np.zeros((1,hobbiesdf.shape[1]-6))
house=np.zeros((1,housedf.shape[1]-6))


for i in range(1,(fooddf.shape[1])-6):
    food[:,i]=sum(fooddf['d_{}'.format(i)])
    
for i in range(1,(fooddf.shape[1])-6):
    hobbies[:,i]=sum(hobbiesdf['d_{}'.format(i)])

for i in range(1,(fooddf.shape[1])-6):
    house[:,i]=sum(housedf['d_{}'.format(i)])
    
# data cleaning
    
plt.figure(figsize=(10,10))
plt.plot(food.ravel())

error=np.argwhere(food<5000)[:,1]

food[:,1912]=food[:,1911]

error=np.argwhere(food<5000)[:,1]

for i in error:
    food[:,i]=(food[:,i-1]+food[:,i+1])/2
    
plt.figure(figsize=(10,10))
plt.plot(food.ravel())
  
# plt.figure(figsize=(10,10))
# plt.plot(food.ravel())
# plt.plot(hobbies.ravel())
# plt.plot(house.ravel())

# one day ahead product sales prediction based on past five days sales

# train=1800

# Xtrain=np.concatenate((food[:,0:train-5],food[:,1:train-4],food[:,2:train-3],food[:,3:train-2],food[:,4:train-1]))
# Ytrain=food[:,5:train]

# Xtest=np.concatenate((food[:,train-5:-10],food[:,train-4:-9],food[:,train-3:-8],food[:,train-2:-7],food[:,train-1:-6]))
# Ytest=food[:,train:-5]
    
# model = Sequential()
# model.add(Dense(50, input_dim=5, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(60,  activation='relu'))
# # number of layers here is the dimension of output, if trying to predict 28 days, then 28 
# model.add(Dense(1, activation='linear'))


# 30 day ahead product sales prediction based on past five datas sales 

pastdays=10
futuredays=30
train=1850

# GREAT TRICK FOR CHECKING OUR WORK IS TO CONVERT OUR DATA INTO INDEX SO WE CAN CLEARLY CHECK
# food=np.arange(food.shape[1]).reshape(1,food.shape[1])

Xtrain=np.zeros((pastdays,train-futuredays))
Ytrain=np.zeros((futuredays,train-futuredays))

for i in range(pastdays,train-futuredays):
    
    Xtrain[:,i-pastdays]=food[:,i-pastdays:i]
    Ytrain[:,i-pastdays]=food[:,i:i+futuredays]

Xtest=np.zeros((pastdays,food.shape[1]-futuredays-train))
Ytest=np.zeros((futuredays,food.shape[1]-futuredays-train))

for i in range(train,food.shape[1]-futuredays):
    
    Xtest[:,i-train]=food[:,i-pastdays:i]
    Ytest[:,i-train]=food[:,i:i+futuredays]

# create the NN
    
model = Sequential()
# input dimension: feature dimension of the Xtrain/Xtest
model.add(Dense(60, input_dim=pastdays, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60,  activation='relu'))
# number of layers here is the feature dimension of output
model.add(Dense(futuredays, activation='linear'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(Xtrain.T, Ytrain.T, epochs=20, verbose=1)

# predict 

Ypred = model.predict(Xtest.T).T

# r2 is a commonly used prediction accuracy measure, optimal=1, bad=0

R2=r2_score(Ytest,Ypred)

# Error=mean_squared_error(Ytest,Ypred)

for i in range(Ypred.shape[1]):
    
    plt.figure(figsize=(10,10))
    plt.title("30 days ahead prediction on the day %f" %(train+i))
    plt.plot(Ytest[:,i].ravel())
    plt.plot(Ypred[:,i].ravel())

# FUTURE WORK
    
# train a model for each item ( food, hobbies, household)
# create a model for each store or state at least, since CA stores work different than texas.
# include day of the week indicator, or at least a week/weekend indicator
# since new varibles included, data has to be normalized (z-score)
# we can cluster this data in different seasons
# include holiday indicator
    


