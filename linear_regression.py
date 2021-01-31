import pandas as pd
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt


data = datasets.load_boston()
print(data.keys())
print("info: ",data.DESCR)
#print("Data is: ",data.data)
#print("Targeted data is: ",data.target)
#print("Features name(column name): ",data.feature_names)

df = pd.DataFrame(np.c_[data.data,data.target],
                  columns=([list(data.feature_names)+['target']]))

print("How Many Null values:\n ",df.isnull().sum())
print("How Many Percent data missing:\n ",df.isnull().sum() / df.shape[0]*100)

# Data visulization
correlation_matrix = df.corr().round(2)
#sns.heatmap(data=correlation_matrix,annot=True)

x = df.drop('target',axis=1)
y = df[['target']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
comapare_y = pd.DataFrame(np.c_[y_pred,y_test],columns=['predicted data','Actual data'])
print("Accuracy : ",lr.score(x_test,y_test))
plt.scatter(y_test,y_pred,c='red')
plt.plot()
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MSE: ",mse)
print("How Many Percentage Model is Accurate: ",(100-mse).round(2))



#using standard scalar
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
print("Mean is: ",sc.mean_)
print("Scale is: ",sc.scale_)
x_train_sc = sc.transform(x_train)
x_test_sc = sc.transform(x_test)
# standard scalar return data in to 2d-array so those data convert into dataframe 
x_train = pd.DataFrame(x_train_sc,columns=(x_train.columns))
x_test = pd.DataFrame(x_test_sc,columns=(x_test.columns))

lr_sc = LinearRegression()
lr_sc.fit(x_train,y_train)
print("Accuracy : ",lr_sc.score(x_test,y_test))
