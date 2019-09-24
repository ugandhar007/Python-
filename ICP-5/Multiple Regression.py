import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns; sns.set(color_codes=True)

train = pd.read_csv('winequality-red.csv')


## Null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
#
# # ## Replacing null values with mean values
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
# print(data)
print(sum(data.isnull().sum() != 0))
#
#
# #Using Pearson Correlation and ploting in the heat map
plt.figure(figsize=(5,5))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#
# # Printing the correlation with the target feature "quality"
print(cor['quality'].sort_values(ascending=False)[:4],'\n')
#
# ##Build a multiple linear regression model
Y = data['quality']
X = data[['alcohol', 'sulphates', 'citric acid']]

# print(X.shape)
#
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
                                    X, Y, random_state=42, test_size=.20)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, Y_train)
##Evaluate the performance by finding square error using R square method
print ("R^2 is: \n", model.score(X_test, Y_test))

##Evaluate the performance by finding square error using RMSE method
predictions = model.predict(X_test)
# print(predictions)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(Y_test, predictions))

# # ##visualize
# #
# actual_values = y_test
# plt.scatter(predictions, actual_values, alpha=.75,
#             color='b') #alpha helps to show overlapping data
# plt.xlabel('Predicted ')
# plt.ylabel('Actual')
# plt.title('Linear Regression Model')
# plt.show()