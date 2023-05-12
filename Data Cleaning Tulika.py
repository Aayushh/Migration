# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 12:00:36 2023

@author: aayus for Tulika data work
"""

import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import os
import sys

os.chdir("C:/Users/aayus/Downloads/Tulika-master")

#data = pd.read_excel('tulika_data.xlsx', sheet_name = 'Migration Details ').iloc[:, [0, 1, 3, 4, 6, 7, 9, 14, 21, 24]]
data = pd.read_excel('tulika_data.xlsx', sheet_name = 'Migration Details ')
data2 = pd.read_csv('Migration Details .csv')
##Constructing variables
#grouping by country
#data = data.sort_values(by = ['countrycode', 'year'])

#----
#Data Descreptive
data.head()
data.tail()
data.info()
stats = data.describe()
stats2 = data2.describe()
data.isnull().sum()

data.hist()
data.iloc[:,0:5].hist(label=None, stacked=True)
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.show()
plt.show()

plt.hist()
         
      # #    
      #     #creating variables of interest
      #     #gdp per worker
      #     data['rgdpew'] = data['rgdpe']/data['emp']
      #     data['const'] = 1

      #     #pop growth by country
      #     data['popgrowth'] = data.groupby('countrycode').pop.diff()/(data['pop'])

      #     #investment
      #     data['cn_1'] = data.groupby('countrycode').cn.shift(-1)
      #     data['investment'] = data['cn_1']- (data['const'] - data['delta'])*data['cn']
      #     data['i_y'] = data['investment']/data['rgdpe']

      #     #TFP
      #     data['g'] = data.groupby('countrycode').rtfpna.diff()/(data['rtfpna'])

      #     #dropping NaN's
      #     data.i_y = data.i_y.replace(0, np.nan)
      #     data = data.dropna(subset = ['popgrowth', 'g', 'delta', 'i_y'])

      #     #dropping years below 1971
      #     data = data.drop(data[data.year <= 1970].index)

      #     #Creating dummy for OECD countries
      #     oecd_countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 'Korea', 'Latvia', 'Lithuania', 'Luxembourg', 'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 'United States']
      #     data['oecd'] = np.where(data['country'].isin(oecd_countries), 1, 0)

      #     #Creating dummy for Developed countries
      #     developing_countries = ['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Armenia', 'Azerbaijan', 'Bangladesh', 'Belarus', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Cape Verde', 'Central African Republic', 'Chad', 'Colombia', 'Comoros', 'Congo', 'Costa Rica', "Cote d'Ivoire", 'Cuba', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Fiji', 'Gabon', 'Gambia', 'Georgia', 'Ghana', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'India', 'Indonesia', 'Iran', 'Iraq', 'Jamaica', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kosovo', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Romania', 'Russia', 'Rwanda', 'Samoa', 'Sao Tome and Principe', 'Senegal', 'Serbia', 'Sierra Leone', 'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan', 'Sri Lanka', 'St. Lucia', 'St. Vincent and the Grenadines', 'Sudan', 'Suriname', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']
      #     data['developing'] = np.where(data['country'].isin(developing_countries), 1, 0)

      #     #Constructing Logs
      #     data['ly'] = np.log(data['rgdpe'])
      #     data['lschool'] = np.log(data['hc'])
      #     data['ls'] = np.log(data['i_y']) #unable to take log
      #     data['lngd'] = np.log((data['popgrowth'] + data['g'] + data['delta']).replace(0, np.nan)) #unable to take log
      #     data['ls_lngd'] = data['ls']
      #     data['lsch_lngd'] = data['lschool'] - data['lngd']
      #     data['t'] = data['year'] - data.groupby('countrycode')['year'].transform('first')
      #     data['gt'] = data['g'] * data['t']
stats.to_csv ('Tulika_descreptive.csv')
stats2.to_csv ("Tulika descreptive non coded.csv")

#Plotting
# Create a grid of subplots for the first 10 columns
fig, axes = plt.subplots(nrows=54, ncols=1, figsize=(5, 84), sharex=False, sharey=False)

# Plot histograms for each column
for i, ax in enumerate(axes.flatten()):
    if i > 3 & i < 54:
        column_name = data.columns[i]
        ax.hist(data.iloc[:, i])
        ax.set_title(column_name + ': ' + str(data.iloc[0, i]))
        i+1
    else:
        ax.axis('off')

# Add a title to the figure
fig.suptitle('Histograms')

# Adjust spacing between subplots

fig.subplots_adjust(hspace=2.5, wspace=0.5)

# Show the plot
plt.show()

