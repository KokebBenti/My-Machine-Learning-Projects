#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#visualze data
df1=pd.read_csv("framingham.csv")
df1.hist(bins=50,figsize=(20,15))

#correlation
corr_matrix=df1.corr()
corr_matrix["TenYearCHD"]

#Clean Data
median = df1["glucose"].median()
df1["glucose"].fillna(median, inplace=True)

#create train and test set
from sklearn.model_selection import train_test_split
train,test=train_test_split(df1, test_size=0.2, random_state=42)

#Assign Variables
x1=train["age"]
x2=train["prevalentHyp"]
x3=train["sysBP"]
x4=train["diaBP"]
x5=train["glucose"]
y=train["TenYearCHD"]
m=len(y)

#Feature Scaling
x1_m=np.mean(x1)
x1_s=np.std(x1)
x1=(x1-x1_m)/(x1_s)
x2_m=np.mean(x2)
x2_s=np.std(x2)
x2=(x2-x2_m)/(x2_s)
x3_m=np.mean(x3)
x3_s=np.std(x3)
x3=(x3-x3_m)/(x3_s)
x4_m=np.mean(x4)
x4_s=np.std(x4)
x4=(x4-x4_m)/(x4_s)
x5_m=np.mean(x5)
x5_s=np.std(x5)
x5=(x5-x5_m)/(x5_s)

#Apply Logistic regression
x0=np.ones(m)
xt=np.c_[x0,x1,x2,x3,x4,x5]
thetas=np.array([0,0,0,0,0,0])
s=np.dot(xt,thetas)
h=1/(1+np.exp(-s))
cost=np.sum((-y*np.log(h)-(1-y)*np.log(1-h))/m)
j=[]
costs=[]
for t in range(0,1500):
 thetas=thetas-(0.1/m)*(np.dot((h-y),xt))
 s = np.dot(xt,thetas)
 h = 1/(1+np.exp(-s))
 j.append(t)
 costs.append(np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m)

#visualize result
plt.plot(j,costs)
plt.show()

#test on test set
x1t=test["age"]
x2t=test["prevalentHyp"]
x3t=test["sysBP"]
x4t=test["diaBP"]
x5t=test["glucose"]
yt=test["TenYearCHD"]
mt=len(yt)

x1t_m=np.mean(x1t)
x1t_s=np.std(x1t)
x1t=(x1t-x1t_m)/(x1t_s)
x2t_m=np.mean(x2t)
x2t_s=np.std(x2t)
x2t=(x2t-x2t_m)/(x2t_s)
x3t_m=np.mean(x3t)
x3t_s=np.std(x3t)
x3t=(x3t-x3t_m)/(x3t_s)
x4t_m=np.mean(x4t)
x4t_s=np.std(x4t)
x4t=(x4t-x4t_m)/(x4t_s)
x5t_m=np.mean(x5t)
x5t_s=np.std(x5t)
x5t=(x5t-x5t_m)/(x5t_s)


x0t=np.ones(mt)
xtt=np.c_[x0t,x1t,x2t,x3t,x4t,x5t]
st=np.dot(xtt,thetas)
ht=1/(1+np.exp(-st))
costt=np.sum((-yt*np.log(ht)-(1-yt)*np.log(1-ht))/mt)
