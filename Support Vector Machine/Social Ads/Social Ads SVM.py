#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
df1=pd.read_csv("Social_Network_Ads.csv")
df1.info()
x=df1[["Age","EstimatedSalary"]]
y=df1["Purchased"]

#Visualize Data
plt.scatter(x["Age"].loc[y==0],x["EstimatedSalary"].loc[y==0],c="k",marker="o")
plt.scatter(x["Age"].loc[y==1],x["EstimatedSalary"].loc[y==1],c="y",marker="+")
plt.show()

#Normalize Data
x=np.array(x)
y=np.array(y)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(x)

#Gaussian RBF Kernel
from sklearn.svm import SVC
svcrbf=SVC(C=10,kernel="rbf",gamma=0.5)
svcrbf.fit(x,y)
z3=svcrbf.predict(x)


#Visualize Result
x1_min,x1_max=x[:,0:1].min()-1,x[:,0:1].max()+1
x2_min,x2_max=x[:,1:2].min()-1,x[:,1:2].max()+1
axes=plt.gca()
axes.set_xlim([-3,4])
axes.set_ylim([-3,4])
h=0.01
x11,x22=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x1_min,x2_max,h))
zg=svcrbf.predict((np.c_[x11.ravel(),x22.ravel()]))
zg=zg.reshape(x11.shape)
plt.contourf(x11,x22,zg,cmap=plt.cm.Spectral)
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(pd.DataFrame(x[:,0:1]).loc[y==0],pd.DataFrame(x[:,1:2]).loc[y==0],c="k",marker="o")
plt.scatter(pd.DataFrame(x[:,0:1]).loc[y==1],pd.DataFrame(x[:,1:2]).loc[y==1],c="y",marker="+")
plt.show()

#Choose best values using grid search
from sklearn.model_selection import GridSearchCV
param_grid=[{'C':np.array([0.01,0.03,0.1,0.3,1,3,10,30]),'gamma':np.array([5000,555.5,50,5.55,0.5,0.055,0.005,0.000556])}]
grid_search = GridSearchCV(svcrbf,param_grid,scoring='accuracy',cv=2,return_train_score=True)
grid_search.fit(x,y)
grid_search.best_params_
grid_search.cv_results_

#Performance Measuring
accuracy=np.sum(z3==y)*100/(len(y))
print("Accuracy is "+ str(accuracy)+" %")