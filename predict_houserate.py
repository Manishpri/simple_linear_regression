
#import library 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model


#load csv file
df = pd.read_csv("Housing.csv")
df.head()
df.describe()
x = df['lotsize']
y = df['price']
x = np.array(x).reshape(-1, 1)
y = np.array(y)

#splite the data into training\testing sets
x_train = x[:-250]
x_test = x[-250:]

#spltite the target into training/testing sets
y_train = y[:-250]
y_test = y[-250:]

#plot scatter
plt.scatter(x_test,y_test,color='black')
plt.title('test data')
plt.xlabel('size')
plt.ylabel('price')
plt.xticks([])
plt.yticks([])
#create our linear regression model
regr = linear_model.LinearRegression()
##training the model using the training sets
regr.fit(x_train,y_train)
#plot output
plt.plot(x_test,regr.predict(x_test),color='red',linewidth=3)
plt.show()
