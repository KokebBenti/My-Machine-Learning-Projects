#Diabetes prediction
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#visualze data
df2=pd.read_csv("diabetes2.csv")
df2.hist(bins=50,figsize=(20,15))

#correlation
corr_matrix=df2.corr()
corr_matrix["Outcome"]

#create train and test set
from sklearn.model_selection import train_test_split
train,test=train_test_split(df2, test_size=0.2, random_state=42)

#Assign Variables
x1=train["Glucose"]
x2=train["BMI"]
y=train["Outcome"]
m=len(y)

#Feature Scaling
x1_m=np.mean(x1)
x1_s=np.std(x1)
x1=(x1-x1_m)/(x1_s)
x2_m=np.mean(x2)
x2_s=np.std(x2)
x2=(x2-x2_m)/(x2_s)

#Apply Logistic regression
x0=np.ones(m)
xt=np.c_[x0,x1,x2]
thetas=np.array([0,0,0])
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
boundary_line=-(np.dot(thetas[1],x1)/thetas[2])-(thetas[0]/thetas[2])
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="m",marker="o",label="0")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="y",marker="+",label="1")
plt.plot(x1,boundary_line)
plt.legend()
plt.show()

#test on test set
x1t=test["Glucose"]
x2t=test["BMI"]
yt=test["Outcome"]
mt=len(yt)

x1t_m=np.mean(x1t)
x1t_s=np.std(x1t)
x1t=(x1t-x1t_m)/(x1t_s)
x2t_m=np.mean(x2t)
x2t_s=np.std(x2t)
x2t=(x2t-x2t_m)/(x2t_s)

x0t=np.ones(mt)
xtt=np.c_[x0t,x1t,x2t]
st=np.dot(xtt,thetas)
ht=1/(1+np.exp(-st))
costt=np.sum((-yt*np.log(ht)-(1-yt)*np.log(1-ht))/mt)

boundary_line=-(np.dot(thetas[1],x1t)/thetas[2])-(thetas[0]/thetas[2])
plt.scatter(x1t.loc[yt==0],x2t.loc[yt==0],c="m",marker="o",label="0")
plt.scatter(x1t.loc[yt==1],x2t.loc[yt==1],c="y",marker="+",label="1")
plt.plot(x1t,boundary_line)
plt.legend()
plt.show()