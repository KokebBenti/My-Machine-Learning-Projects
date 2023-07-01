#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read and understand data
d2=pd.read_csv('realest.csv')
d2.info()
d2.hist(bins=50,figsize=(20,15))
corr_matrix=d2.corr()
corr_matrix["Price"]

#Clean data
m1 = d2["Space"].median()
d2["Space"].fillna(m1, inplace=True)
m2 = d2["Lot"].median()
d2["Lot"].fillna(m2, inplace=True)
m3 = d2["Tax"].median()
d2["Tax"].fillna(m3, inplace=True)

#Create test set and training set
from sklearn.model_selection import train_test_split
tr,te=train_test_split(d2, test_size=0.2, random_state=42)


#Assign Variables
x1=np.array(tr["Space"])
x2=np.array(tr["Room"])
x3=np.array(tr["Bathroom"])
x4=np.array(tr["Garage"])
y=np.array(tr["Price"])
m=len(x1)

#Feature Normalization
x1_m=np.mean(x1)
x1_s=np.std(x1)
x2_m=np.mean(x2)
x2_s=np.std(x2)
x3_m=np.mean(x3)
x3_s=np.std(x3)
x4_m=np.mean(x4)
x4_s=np.std(x4)
y_m=np.mean(y)
y_s=np.std(y)
x1=(x1-x1_m)/x1_s
x2=(x2-x2_m)/x2_s
x3=(x3-x3_m)/x3_s
x4=(x4-x4_m)/x4_s
y=(y-y_m)/y_s


#Visualize Variables
plt.scatter(x1,y)
plt.xlabel("Space")
plt.show()
plt.scatter(x2,y)
plt.xlabel("Room")
plt.show()
plt.scatter(x3,y)
plt.xlabel("Bathroom")
plt.show()
plt.scatter(x4,y)
plt.xlabel("Garage")
plt.show()



#Apply Gradient Descent and Calculate Cost
x0=np.ones(m)
xt=np.c_[x0,x1,x2,x3,x4]
thetas=np.array([0,0,0,0,0])
h=np.dot(xt,thetas)
thetas=thetas-(0.1/m)*np.dot((h-y),xt)
cost=(np.sum((h-y)**2))/(2*m)
cost1=[]
l=[]
for i in range(15000):
 thetas=thetas-(0.01/m)*np.dot((h-y),xt)
 h = np.dot(xt,thetas)
 cost1.append((np.sum((h-y)**2))/(2*m))
 l.append(i)

#Visualize Result
plt.plot(l,cost1)
plt.show()
plt.scatter(x1,y,marker="o")
plt.scatter(x2,y,marker="o")
plt.plot(x1,h)
plt.plot(x2,h)
plt.show()

yreal=(y*y_s)+y_m
hp=np.dot(xt,thetas)
hreal=(hp*np.std(hp))+np.mean(hp)
cost=(np.sum((hreal-yreal)**2))/(2*m)
realc=np.sqrt(cost)

#using scikit-learn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(xt,y)

from sklearn.metrics import mean_squared_error
y_real=(y*y_s)+y_m
h_p = lin_reg.predict(xt)
h_real=(h_p*np.std(h_p))+np.mean(h_p)
lin_mse = mean_squared_error(y_real, h_real)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
