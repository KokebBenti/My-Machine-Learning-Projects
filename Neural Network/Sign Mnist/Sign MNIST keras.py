# Import Libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load Data
df1=np.array(pd.read_csv("sign_mnist_train.csv"))
df2=np.array(pd.read_csv("sign_mnist_test.csv"))

x=df1[:,1:]
y=df1[:,0:1]
xtt=df1[:,1:]
ytt=df1[:,0:1]

#Feature Scaling
xv,xt=(x[0:5491]/255),(x[5491:])/255
yv,yt=(y[0:5491]).flatten(),(y[5491:]).flatten()
xtt=xtt/255
ytt=y.flatten()

#Visualize data
import matplotlib.image as mpimg
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
 for j in range(10):
  axis[i,j].imshow(xt[np.random.randint(0,21964),:].reshape(28,28,order="F"),cmap="binary")
  axis[i,j].axis("off")
plt.show()

#create model using Sequential API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[784,1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(256, activation="elu",kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(256, activation="elu",kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(25, activation="softmax",kernel_initializer="glorot_normal"))

#Compile Model
model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.RMSprop(),metrics=["accuracy"])

#train and evaluate model
history=model.fit(xt,yt,epochs=15,validation_data=(xv,yv))
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

#test Set
model.evaluate(xtt,ytt)