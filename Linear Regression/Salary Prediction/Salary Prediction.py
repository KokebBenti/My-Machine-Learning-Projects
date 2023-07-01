#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Assign Variables
d1=pd.read_csv('Salary Data.csv')

#Visualize Variables
x=np.array(d1["YearsExperience"])
y=np.array(d1["Salary"])
m=len(x)
plt.scatter(x,y)
plt.show()


#Apply Gradient Descent and Calculate Cost
x0=np.ones(m)
x1=np.c_[x0,x]
thetas=np.array([0,0])
h=np.dot(x1,thetas)
thetas=thetas-(0.01/m)*np.dot((h-y),x1)
cost=(np.sum((h-y)**2))/(2*m)
cost1=[]
l=[]
for i in range(15000):
 thetas=thetas-(0.01/m)*np.dot((h-y),x1)
 h = np.dot(x1,thetas)
 cost1.append((np.sum((h-y)**2))/(2*m))
 l.append(i)


#Visulize Result
plt.scatter(x,y,marker="o")
plt.plot(x,h)
plt.show()

plt.plot(l,cost1)
plt.show()