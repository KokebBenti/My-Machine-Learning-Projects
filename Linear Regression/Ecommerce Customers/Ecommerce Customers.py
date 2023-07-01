#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read and understand data
df1=pd.read_csv('Ecommerce Customers.csv')
df1.info()
df1.hist(bins=50,figsize=(20,15))
corr_matrix=df1.corr()
corr_matrix["Yearly Amount Spent"]

#Create test set and training set
from sklearn.model_selection import train_test_split
tr,te=train_test_split(df1, test_size=0.2, random_state=42)

#Assign Variables
x1=np.array(tr["Length of Membership"])
x2=np.array(tr["Time on App"])
x3=np.array(tr["Time on Website"])
y=np.array(tr["Yearly Amount Spent"])
m=len(x1)

#Feature Normalization
x1_m=np.mean(x1)
x1_s=np.std(x1)
x2_m=np.mean(x2)
x2_s=np.std(x2)
x3_m=np.mean(x3)
x3_s=np.std(x3)
y_m=np.mean(y)
y_s=np.std(y)
x1=(x1-x1_m)/x1_s
x2=(x2-x2_m)/x2_s
x3=(x3-x3_m)/x3_s
y=(y-y_m)/y_s


#Visualize Variables
plt.scatter(x1,y)
plt.xlabel("Length of Membership")
plt.show()
plt.scatter(x2,y)
plt.xlabel("Time on App")
plt.show()
plt.scatter(x3,y)
plt.xlabel("Time on Website")
plt.show()

#Apply Gradient Descent and Calculate Cost
x0=np.ones(m)
xt=np.c_[x0,x1,x2,x3]
thetas=np.array([0,0,0,0])
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

yreal=(y*y_s)+y_m
hp=np.dot(xt,thetas)
hreal=(hp*np.std(hp))+np.mean(hp)
cost=(np.sum((hreal-yreal)**2))/(2*m)
realc=np.sqrt(cost)

#test on test set
x1t=np.array(te["Length of Membership"])
x2t=np.array(te["Time on App"])
x3t=np.array(te["Time on Website"])
yt=np.array(te["Yearly Amount Spent"])
mt=len(x1t)

x1t_m=np.mean(x1t)
x1t_s=np.std(x1t)
x2t_m=np.mean(x2t)
x2t_s=np.std(x2t)
x3t_m=np.mean(x3t)
x3t_s=np.std(x3t)
yt_m=np.mean(yt)
yt_s=np.std(yt)
x1t=(x1t-x1t_m)/x1t_s
x2t=(x2t-x2t_m)/x2t_s
x3t=(x3t-x3t_m)/x3t_s
yt=(yt-yt_m)/yt_s

x0t=np.ones(mt)
xtt=np.c_[x0t,x1t,x2t,x3t]
ht=np.dot(xtt,thetas)
ytreal=(yt*yt_s)+yt_m
htreal=(ht*np.std(ht))+np.mean(ht)
costt=(np.sum((htreal-ytreal)**2))/(2*mt)
realct=np.sqrt(costt)


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

#using scikit-learn on test set
from sklearn.metrics import mean_squared_error
yt_real=(yt*yt_s)+yt_m
ht_p = lin_reg.predict(xtt)
ht_real=(ht_p*np.std(ht_p))+np.mean(ht_p)
lin_mset = mean_squared_error(yt_real,ht_real)
lin_rmset = np.sqrt(lin_mset)
lin_rmset