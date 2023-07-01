# California House Price
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Visualize data
df1=pd.read_csv('housing.csv')
df1.info()
df1.hist(bins=50,figsize=(20,15))
plt.scatter(df1["longitude"],df1["latitude"],alpha=0.1,c=df1["median_house_value"],cmap=plt.get_cmap("jet"))
plt.show()

#Create test set and training set
from sklearn.model_selection import train_test_split
d1,d2=train_test_split(df1, test_size=0.2, random_state=42)

#Correlation
d1["rooms_per_household"]=d1["total_rooms"]/d1["households"]
d1["bedroom_per_room"]=d1["total_bedrooms"]/d1["total_rooms"]
d1["population_per_household"]=d1["population"]/d1["households"]
corr_matrix=d1.corr()
corr_matrix["median_house_value"]

#Clean Data
median = d1["total_bedrooms"].median()
d1["total_bedrooms"].fillna(median, inplace=True)

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
d1c=d1[["ocean_proximity"]]
d1c_1hot = cat_encoder.fit_transform(d1c)
d1c_1hot=d1c_1hot.toarray()

#Assign Variables
x1=d1["housing_median_age"]
x2=d1["bedroom_per_room"]
x3=d1["median_income"]
y=d1["median_house_value"]
m=len(y)

#Feature scaling
x1_m=np.mean(x1)
x1_s=np.std(x1)
x1=(x1-x1_m)/(x1_s)
x2_m=np.mean(x2)
x2_s=np.std(x2)
x2=(x2-x2_m)/(x2_s)
x3_m=np.mean(x3)
x3_s=np.std(x3)
x3=(x3-x3_m)/(x3_s)
y_m=np.mean(y)
y_s=np.std(y)
y=(y-y_m)/(y_s)

#Apply Gradient Descent and Calculate Cost
x0=np.ones(m)
xt=np.c_[x0,x1,x2,x3]
thetas=np.array([0,0,0,0])
h=np.dot(xt,thetas)
thetas=thetas-(0.01/m)*np.dot((h-y),xt)
cost=(np.sum((h-y)**2))/(2*m)
cost1=[]
l=[]
for i in range(1500):
 thetas=thetas-(0.01/m)*np.dot((h-y),xt)
 h = np.dot(xt,thetas)
 cost1.append((np.sum((h-y)**2))/(2*m))
 l.append(i)

#Visulize Result
plt.plot(l,cost1)
plt.show()

yreal=(y*y_s)+y_m
hp=np.dot(xt,thetas)
hreal=(hp*np.std(hp))+np.mean(hp)
cost=(np.sum((hreal-yreal)**2))/(2*m)
realc=np.sqrt(cost)

#using Scikit learn
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


#test result on test set
d2["rooms_per_household"]=d2["total_rooms"]/d2["households"]
d2["bedroom_per_room"]=d2["total_bedrooms"]/d2["total_rooms"]
d2["population_per_household"]=d2["population"]/d2["households"]

median = d2["total_bedrooms"].median()
d2["total_bedrooms"].fillna(median, inplace=True)

x1t=d2["housing_median_age"]
x2t=d2["bedroom_per_room"]
x3t=d2["median_income"]
yt=d2["median_house_value"]
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
yt_m=np.mean(y)
yt_s=np.std(yt)
yt=(yt-yt_m)/(yt_s)

x0t=np.ones(mt)
xtt=np.c_[x0t,x1t,x2t,x3t]
ht=np.dot(xtt,thetas)
ytreal=(yt*yt_s)+yt_m
htreal=(ht*np.std(ht))+np.mean(ht)
costt=(np.sum((hreal-yreal)**2))/(2*mt)
realct=np.sqrt(costt)


#using Scikit learn on test set
yt_real=(yt*yt_s)+yt_m
ht_p = lin_reg.predict(xtt)
ht_real=(ht_p*np.std(ht_p))+np.mean(ht_p)
lin_mset=mean_squared_error(yt_real,ht_real)
lin_rmset=np.sqrt(lin_mset)
lin_rmset