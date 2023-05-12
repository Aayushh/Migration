# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:13:06 2023

@author: aayus
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 12:00:36 2023

@author: aayus for Tulika data work
"""
import seaborn as sns
import numpy as np
import pandas as pd
from linearmodels import PanelOLS
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import os
import sys
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt


os.chdir("C:/Users/aayus/Downloads/Tulika-master")

pd.set_option('display.max_columns', 10)

#data = pd.read_excel('tulika_data.xlsx', sheet_name = 'Migration Details ').iloc[:, [0, 1, 3, 4, 6, 7, 9, 14, 21, 24]]
data = pd.read_excel('tulika_data.xlsx', sheet_name = 'Migration Details ')

#----
#Data Descreptive
data.head()
data.tail()
data.info()
data.describe()

data.iloc[:,12]
# drop columns 13 to 18
data = data.drop(data.columns[12:18], axis=1)

for col in data.columns:
    try:
        data[col] = pd.to_numeric(data[col])
    except ValueError:
        pass

# -------
# check for missing values in each column
missing_values = data.isnull().sum()

# select only the columns with missing values
missing_cols = missing_values[missing_values != 0]

# print the columns with missing values and the number of missing values in each column
print("Columns with missing values:")
for col, num_missing in missing_cols.iteritems():
    print(f"{col}: {num_missing} missing values")
# ---------------

# define the predictor variables and the response variable
X = data.iloc[:, 2:12].join(data.iloc[:, 13:]) # select all columns except the 5th column as predictor variables
# X = data.iloc[:, 13:]
y = data.iloc[:, 12] # select the 5th column as the response variable

X = sm.add_constant(X)
# fit the ordinal logistic regression model
model = sm.OLS(y, X).fit(method='pinv', maxiter=5000)

# print the model summary
print(model.summary())

# ---# Customize the summary table
table = model.summary2()
table.tables[1] = table.tables[1][table.tables[1]['P>|t|'] < 0.1]

# Print the modified summary table
print(table)







summary = model.summary2()
summary.tables[1]['P>|t|'] = summary.tables[1]['P>|t|'].apply(lambda x: '{:.3f}'.format(x))
summary.tables[1]['***'] = summary.tables[1]['P>|t|'].apply(lambda x: '*' if float(x) < 0.05 else '')
print(summary.tables[1])

# --export
fig, ax = plt.subplots(figsize=(10, 5))
summary.tables[1].plot(kind='bar', ax=ax)

# Set the title and axes labels
ax.set_title('Logistic Regression Summary Table')
ax.set_xlabel('Coefficients')
ax.set_ylabel('P-Values')

# Save the figure as a PNG image
plt.savefig('summary_table.png', dpi=300, bbox_inches='tight')

#table-export
table = summary.tables[1]
table = table.set_index(table.iloc[:,0])
table = table.iloc[:,1:]
plt.figure(figsize=(10, 5))
plt.axis('off')
plt.table(cellText=table.values,
          colLabels=table.columns,
          rowLabels=table.index,
          colWidths=[0.2]*len(table.columns),
          bbox=[0,0,1,1])
plt.savefig('summary.png')


# data.info()
# # select only the object columns
# object_cols = data.select_dtypes(include=['object'])

# # print the object columns
# print(object_cols)

# X = data.iloc[:, :-1]

# # check for missing values
# print(X.isnull().sum())

# # check for infinite values
# print(np.isinf(X).sum())



# -----
corr_matrix = data.corr()

# Visualize the correlation matrix using a heatmap
sns.heatmap(corr_matrix)

# Calculate the VIF for each predictor variable
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

# Print the VIF values
print(vif)





