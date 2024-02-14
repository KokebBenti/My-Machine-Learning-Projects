# Import Libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load data
Fmnist=np.array(pd.read_csv("fashion-mnist_train.csv"))
x,y=Fmnist[:,1:],Fmnist[:,0:1]


#Feature Scaling
xv,xt=(x[0:5000]/255),(x[5000:])/255
yv,yt=(y[0:5000]).flatten(),(y[5000:]).flatten()
class_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

#visualize data
import matplotlib.image as mpimg
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
 for j in range(10):
  axis[i,j].imshow(x[np.random.randint(0,5001),:].reshape(28,28,order="F"),cmap="binary")
  axis[i,j].axis("off")
plt.show()

#create model using Sequential API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[784,1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(300, activation="elu",kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(300, activation="elu",kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(10, activation="softmax",kernel_initializer="glorot_normal"))


#Compile Model
model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(),metrics=["accuracy"])


#train and evaluate model
history=model.fit(xt,yt,epochs=15,validation_data=(xv,yv))
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()



#train and evaluate model using Tensorboard
import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
 import time
 run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
 return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history=model.fit(xt,yt,epochs=30,validation_data=(xv,yv),callbacks=[tensorboard_cb])
%load_ext tensorboard
%tensorboard --logdir=./my_logs --port=6006


#save model
model.save("my_keras_model.h5")