import pandas as pd
import numpy as np
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
from scipy import stats


# Removing the Anamolies using z-score
# if the data is more than 3 standard deviations away then it is considered as outlier
# if the data is less than -3 standard deviations away then it is considered as outlier
df = pd.read_csv('train.csv', sep=',',usecols=(62,80))
z = np.abs(stats.zscore(df))
threshold = 3
print(np.where(z > 2))
modified_df = df[(z > -3).all(axis=1)]
print(df.shape)
print(modified_df.shape)

# Create Scatterplot of the dataframe after removing the anamolies in the data
sns.lmplot('GarageArea', # Horizontal axis
           'SalePrice', # Vertical axis
           data=modified_df, # Data source
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
