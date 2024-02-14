# Import Libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load data
from scipy.io import loadmat
d1=loadmat("ex3data1.mat")


#training and test set
x=np.array(d1["X"])
y=np.array(d1["y"])
Total=np.c_[x,y]
from sklearn.model_selection import train_test_split
train,test=train_test_split(Total, test_size=0.2, random_state=42)
x=train[:,:-1]
y=train[:,-1:]
m=len(y)
y[y==10]=0

#Feature Scaling
xv,xtt,xt=(x[0:500]/255),(x[500:1000]/255),(x[1000:])/255
yv,ytt,yt=(y[0:500]).flatten(),(y[500:1000]).flatten(),(y[1000:]).flatten()

#Visualize data
import matplotlib.image as mpimg
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
 for j in range(10):
  axis[i,j].imshow(x[np.random.randint(0,4001),:].reshape(20,20,order="F"),cmap="binary")
  axis[i,j].axis("off")
plt.show()

#create model using Sequential API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[400,1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(256, activation="elu",kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(256, activation="elu",kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(10, activation="softmax",kernel_initializer="glorot_normal"))

#Compile Model
model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(),metrics=["accuracy"])

#train and evaluate model
history=model.fit(xt,yt,epochs=15,validation_data=(xv,yv))
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

#test Set
model.evaluate(xtt,ytt)