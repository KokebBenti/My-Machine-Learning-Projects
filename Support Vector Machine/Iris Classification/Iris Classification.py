#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load data
data=np.array(pd.read_csv("iris.csv"))
df1=data[:,1:]

#Create train and test set
from sklearn.model_selection import train_test_split
train,test=train_test_split(df1,test_size=0.2, random_state=42)

#Assign variables
x=train[:,0:4]
y=train[:,-1:]

#Change y values to integer values
from sklearn.preprocessing import OrdinalEncoder
ord=OrdinalEncoder()
ye=(ord.fit_transform(y)).flatten()


#Linear SVM Classification (c=0.5)
from sklearn.svm import SVC
svc=SVC(C=0.55,kernel="linear")
svc.fit(x,ye)
z=svc.predict(x)

#Performance Measuring
accuracy=np.sum(z==ye)*100/(120)
print("Accuracy is "+ str(accuracy)+" %")


#Test on test set
xt=test[:,0:4]
yt=test[:,-1:]
yte=(ord.fit_transform(yt)).flatten()
zt=svc.predict(xt)
accuracy=np.sum(zt==yte)*100/(30)
print("Accuracy is "+ str(accuracy)+" %")


#Not working
#Visualize Result (using petal width and petal length)
x1=pd.Series((x[:,0:1]).flatten())
x2=pd.Series((x[:,1:2]).flatten())
boundary_line=-((svc.coef_[:,0]*x1)/svc.coef_[:,1])-(svc.intercept_[0]/svc.coef_[:,1])
plt.scatter(x1.loc[ye==0],x2.loc[ye==0],c="y",marker="o")
plt.scatter(x1.loc[ye==1],x2.loc[ye==1],c="k",marker="o")
plt.scatter(x1.loc[ye==2],x2.loc[ye==2],c="b",marker="o")
#plt.plot(x1,boundary_line,label="c=1")
plt.legend()
plt.show()
