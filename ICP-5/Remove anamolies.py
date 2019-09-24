import pandas as pd
import numpy as np
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('train.csv', sep=',', usecols=['GarageArea', 'SalePrice'])

# Create scatterplot of dataframe before removing Anamolies
sns.lmplot('GarageArea', # Horizontal axis
           'SalePrice', # Vertical axis
           data=df, # Data source
           fit_reg=False, # Don't fix a regression line
           scatter_kws={"marker": "o", # Set marker style
                        "s": 80}) # S marker size

# Set title
plt.title('plot between Garage Area and SalPrice')
plt.xlim(-200,1600)
# Set x-axis label
plt.xlabel('GarageArea')
# Set y-axis label
plt.ylabel('SalePrice')
plt.show()

# # Using box plot to identifty the outliers
# sns.boxplot(x=df['GarageArea'])
# plt.show()

# # Removing the Anamolies by directly specifying the limits of the outliers
train = df
train['GarageArea'] = train[train['GarageArea']>200]
train['GarageArea'] = train[train['GarageArea']<1200]


# Create Scatterplot of the dataframe after removing the anamolies in the data
sns.lmplot('GarageArea', # Horizontal axis
           'SalePrice', # Vertical axis
           data=train, # Data source
           fit_reg=False, # Don't fix a regression line
           scatter_kws={"marker": "o", # Set marker style
                        "s": 80}) # S marker size

# Set title
plt.title('plot between Garage Area and SalPrice')
plt.xlim(-200, 1600)
# Set x-axis label
plt.xlabel('GarageArea')
# Set y-axis label
plt.ylabel('SalePrice')
plt.show()
