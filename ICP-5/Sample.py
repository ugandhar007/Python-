import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

train = pd.read_csv('winequality-red.csv')

# print(train.describe());

# Next, we'll check for skewness
print ("Skew is:", train.quality.skew())
plt.hist(train.quality, color='blue')
plt.show()

# Then, Logtransform the target
target = np.log(train.quality)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


## Null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
# print(nulls)


# #Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])

print(numeric_features)

corr = numeric_features.corr()
# print(corr)

# #representing Corr in heatmap
plt.figure(figsize=(5,5))
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()


# Selecting top 5 positive correlated features
print (corr['quality'].sort_values(ascending=False)[:4], '\n')

# Selecting top 5 negative correlated features
# print (corr['quality'].sort_values(ascending=False)[-5:])

y = data['quality']
X = data[['alcohol', 'sulphates', 'citric acid']]

quality_pivot = train.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)
print(quality_pivot)