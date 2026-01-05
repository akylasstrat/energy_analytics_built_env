# -*- coding: utf-8 -*-
"""
Predicting fuel poverty via classification

@author: a.stratigakos@ucl.ac.uk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Plotting figures default
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

#%%

data_path = 'C:\\Users\\ucbva19\\OneDrive - University College London\\ESDA\\BENV0092 Energy Data Analytics in the Built Environment\\data sets'
fuel_poverty_df = pd.read_csv(f'{data_path}\FuelPoverty_2019_processed.csv')


#%%
# Scatter plots of potential factors

plt.scatter(fuel_poverty_df['hh_income'], fuel_poverty_df['In_fuel_Poverty'])
plt.show()

plt.scatter(fuel_poverty_df['fuel_costs'], fuel_poverty_df['In_fuel_Poverty'])
plt.show()

plt.scatter(fuel_poverty_df['In_fuel_Poverty'], fuel_poverty_df['Head_Working_Status'])
plt.show()

#%% Create training/ test set

feat_col = ['fpvuln', 'Region', 'hh_income', 'fuel_costs', 
            'EPC', 'Tenure', 'hhcompx']

#Lets interpret a few significant variables:
#fpvuln: Being a non-vulnerable household decreases the log odds of being in fuel poverty (versus not being)
#A household is now counted as vulnerable in these statistics if it contains at least one household member who 
#is 65 or older, younger than 5 or living with a long-term health condition affecting mobility, breathing, heart or mental health condition. 

#Region: Living in West Midlands increases the log odds of being in fuel poverty, compared to
#living in the North East. 
#Living in London and the SE decreases the log odds of being in fuel poverty, compared to
#living in the North East.

#Tenure: Living in private rented, local authority or housing association homes increases the log odds of being in fuel poverty,
#compared to living in an owner occupied dwelling. So the ODDS increase by (e^Estimate)

#house composition: Over 60s Couples without children have increased log odds of being in fuel poverty,
#compared to Under 60s Couples without children. Same applies for most remaining categories.

#Working status: Houses where the head is unemployed or inactive, have increased log odds of being fuel poor,
#compared to houses where the head is working

#Ethnic group: Houses where the head belongs to an ethnic minority have increased log odds of being
#fuel poor compared to houses where the head is white

Y = fuel_poverty_df['In_fuel_Poverty']
X = fuel_poverty_df[feat_col]

# Check imbalance
print(f'Percentage of positive classes {(100*(Y==1).sum()/len(Y)).round(2)}%')
#%%

from sklearn.model_selection import train_test_split

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

#%%
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

et_model = ExtraTreesClassifier(n_estimators = 500)
et_model.fit(train_X, train_Y)

#%%
Y_pred = et_model.predict(test_X)

cm = confusion_matrix(test_Y.values, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

