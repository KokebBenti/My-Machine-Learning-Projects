#Iris classification
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#visualize data
from sklearn import datasets
data1=datasets.load_iris()
x=pd.DataFrame(data1["data"])
y=pd.DataFrame(data1["target"])
iris=pd.DataFrame(np.c_[np.array(x),np.array(y)])
iris.hist(bins=50,figsize=(20,15))

#correlation
corr_matrix=iris.corr()
corr_matrix[4]

#create train and test set
from sklearn.model_selection import train_test_split
train,test=train_test_split(iris, test_size=0.2, random_state=42)


#Assign Variables
train=np.array(train)
x1=pd.Series(train[:,2])
x2=pd.Series(train[:,3])
y=train[:,-1:].flatten()
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
for t in range(0,15000):
 thetas=thetas-(.000032/m)*(np.dot((h-y),xt))
 s = np.dot(xt,thetas)
 h = 1/(1+np.exp(-s))
 j.append(t)
 costs.append(np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m)

#visualize result
plt.plot(j,costs)
plt.show()
boundary_line=-(np.dot(thetas[1],x1)/thetas[2])-(thetas[0]/thetas[2])
plt.scatter(x1.loc[h<=0.4],x2.loc[h<=0.4],c="m",marker="o",label="Iris Setosa")
plt.scatter(x1.loc[(h>0.4)&(h<=0.6)],x2.loc[(h>0.4)&(h<0.6)],c="y",marker="+",label="Iris Versicolor")
plt.scatter(x1[h>0.6],x2.loc[h>0.6],c="b",marker="o",label="Iris Virginica")
plt.plot(x1,boundary_line)
plt.legend()
plt.show()


#test on test set
test=np.array(test)
x1t=pd.Series(test[:,2])
x2t=pd.Series(test[:,3])
yt=test[:,-1:].flatten()
mt=len(yt)

x1t_m=np.mean(x1t)
x1t_s=np.std(x1t)
x1t=(x1t-x1t_m)/(x1t_s)
x2t_m=np.mean(x2t)
x2t_s=np.std(x2t)
x2t=(x2t-x2t_m)/(x2t_s)


xt0=np.ones(mt)
xtt=np.c_[xt0,x1t,x2t]
st=np.dot(xtt,thetas)
ht=1/(1+np.exp(-st))
costt=np.sum((-yt*np.log(ht)-(1-yt)*np.log(1-ht))/mt)
for i in range(len(yt)):
 if 0<=ht[i]<=0.4:
  if yt[i]==0: print( str(i) +" is predicted to be Setosa and it is Setosa")
  if yt[i]==1: print( str(i) +" is predicted to be Setosa and it is versicolor")
  if yt[i]==2: print( str(i) +" is predicted to be Setosa and it is virginica")
 if 0.4 <= ht[i] <= 0.66:
  if yt[i]==0: print(str(i) + " is predicted to be versicolor and it is Setosa")
  if yt[i]==1: print(str(i) + " is predicted to be versicolor and it is versicolor")
  if yt[i]==2: print(str(i) + " is predicted to be versicolor and it is virginica")
 if 0.66 <= ht[i] <= 1:
  if yt[i]==0: print(str(i) + " is predicted to be virginica and it is Setosa")
  if yt[i]==1: print(str(i) + " is predicted to be virginica and it is versicolor")
  if yt[i]==2: print(str(i) + " is predicted to be virginica and it is virginica")

boundary_line=-(np.dot(thetas[1],x1t)/thetas[2])-(thetas[0]/thetas[2])
plt.scatter(x1t.loc[ht<=0.4],x2t.loc[ht<=0.4],c="m",marker="o",label="Iris Setosa")
plt.scatter(x1t.loc[(ht>0.4)&(ht<=0.6)],x2t.loc[(ht>0.4)&(ht<0.6)],c="y",marker="+",label="Iris Versicolor")
plt.scatter(x1t[ht>0.6],x2t.loc[ht>0.6],c="b",marker="o",label="Iris Virginica")
#plt.plot(x1t,boundary_line)
plt.legend()
plt.show()