# Purchase Prediction
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Visualize data
df1=pd.read_csv("Social_Network_Ads.csv")
df1.info()
x1=df1["Age"]
x2=df1["EstimatedSalary"]
x3=df1[["Gender"]]
y=df1["Purchased"]
m=len(y)
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="m",marker="+",label="didnt purchase")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="y",marker="o",label="purchased")
plt.legend()
plt.show()


#Feature Normalization
x1_m=np.mean(x1)
x1_s=np.std(x1)
x2_m=np.mean(x2)
x2_s=np.std(x2)
x1=(x1-x1_m)/x1_s
x2=(x2-x2_m)/x2_s

#One-hot Encoding?
#needs 2D array look at x3 and wouldn't be concatenated
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
x3h2 = cat_encoder.fit_transform(x3)
x3h2.toarray()

#ordial encoding
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
x3h=ordinal_encoder.fit_transform(x3)
ordinal_encoder.categories_


#Apply Logistic regression
x0=np.ones(m)
xt=np.c_[x0,x1,x2,x3h]
thetas=np.array([0,0,0,0])
s=np.dot(xt,thetas)
h=1/(1+np.exp(-s))
cost=np.sum((-y*np.log(h)-(1-y)*np.log(1-h))/m)
j=[]
costs=[]
for t in range(0,10000):
 thetas=thetas-(0.1/m)*(np.dot((h-y),xt))
 s = np.dot(xt,thetas)
 h = 1/(1+np.exp(-s))
 j.append(t)
 costs.append(np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m)

#Visualize result
plt.plot(j,costs)
plt.show()

boundary_line=-(np.dot(thetas[1],x1)/thetas[2])-(thetas[0]/thetas[2])
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="m",marker="o",label="didnt purchase")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="y",marker="+",label="purchased")
plt.plot(x1,boundary_line)
plt.legend()
plt.show()


#using scikit learn
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xt,y)
w0,w1,w2,w3=model.coef_.T
b=model.intercept_[0]

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,h)
rmse=np.sqrt(mse)

boundary_line_sci=-((w1*x1)/w2)-(b/w2)
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="m",marker="o",label="didnt purchase")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="y",marker="+",label="purchased")
plt.plot(x1,boundary_line_sci)
plt.legend()
plt.show()