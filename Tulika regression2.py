# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 00:20:55 2023

@author: aayus
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
from statsmodels.miscmodels.ordinal_model import OrderedModel

os.chdir("C:/Users/aayus/Downloads/Tulika-master")
pd.set_option('display.max_columns', 10)
data = pd.read_excel('tulika_data.xlsx', sheet_name = 'Migration Details ').iloc[:, [26,29,30,33,34,39,40]]
y = pd.read_excel('tulika_data.xlsx', sheet_name = 'Migration Details ').iloc[:, [44]]

data.dtypes

# define the predictor variables and the response variable
X = data
X = sm.add_constant(X)
# fit the ordinal logistic regression model
mod_prob = OrderedModel(y,X, distr='probit')
res_prob = mod_prob.fit(method='bfgs')
res_prob.summary()

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





