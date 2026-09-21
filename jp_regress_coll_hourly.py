# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:00:18 2026

@author: princ
"""

import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns
from scipy import stats
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
#%%

#read in the cleaned and joined data
average = pd.read_csv('A:/SimpleAQ/Jackson_Pike/avg_hourly_JP.csv')
average.describe()
print(average.isnull().sum())
average_with_na = average
average = average.dropna()
 
 



# Example: assume df is your original DataFrame
# df = pd.DataFrame({...})  # already defined with mean_pm_pa, pm_mean_verde, humidity
# 1. Define features (X) and target (y)
X = average[['avg_saq_hourly','avg_temp_hourly']]  # Independent variables
y = average[['sample_measurement']]                    # Dependent variable
# 2. Split into train/test (if not already split)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
# 3. Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# 4. Get the coefficients and intercept
#a,b,c = model.coef_[2]
#d = model.intercept_
#print(f"Model: mean_pm_pa = {model.coef_[0]} x mean_pm_saq + {model.coef_[1]} x mean_rh_saq + {model.coef_[2]} x mean_temp_saq + {model.intercept_}")


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
#%%
residuals = y_test - y_pred
pyplot.scatter(y_pred, residuals, alpha=0.5)
pyplot.xlabel('predicted_saq Values')
pyplot.ylabel('Residuals')
pyplot.title('Residual Plot')
pyplot.axhline(y=0, color='red', linestyle='--')
pyplot.show()

# predicted_saq vs Actual Plot
pyplot.scatter(y_test, y_pred, alpha=0.5)
pyplot.xlabel('Actual Values')
pyplot.ylabel('predicted_saq Values')
pyplot.title('predicted_saq vs Actual Values')
pyplot.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=4)
pyplot.show()

#Model Stats

# Add a constant to the model
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())

# Q-Q Plot for residuals
sm.qqplot(model_sm.resid, line='s')
pyplot.title('Q-Q Plot of Residuals')
pyplot.show()

#%%
all_predictions = model.predict(X)
print(all_predictions)

#merged = y
average['predicted_pm_saq'] = all_predictions
average['predicted_pm_rh_saq'] = all_predictions
average['predicted_pm_temp_saq'] = all_predictions
print(average)

average.to_csv('A:/SimpleAQ/Jackson_Pike/JP_predict_hourly.csv', index=False)
avg_predict = pd.read_csv('A:/SimpleAQ/Jackson_Pike/JP_predict_hourly.csv')
#pd.set_option('display.max_columns', None)
#print(avg_predict.describe())

#%%
#PLAIN
slope, intercept, r_value, p_value, std_err = stats.linregress(avg_predict['sample_measurement'], avg_predict['predicted_pm_saq'])
r_squared = r_value**2
print(r_squared)
#%%
fig, ax = pyplot.subplots(figsize = (10,8))
scatter = ax.scatter(avg_predict['predicted_pm_saq'], avg_predict['sample_measurement'])
#legend = ax.legend(*scatter.legend_elements(), loc="lower right",
                  # title="EPA v Sen55 (SimpleAQ)")
#ax.add_artist(legend)
ax.set_xlabel('Predicted SimpleAQ Values')
ax.set_ylabel('EPA Values')
#pyplot.ylim(0)
#pyplot.xlim(0)
#ax.set_title('Plantower v Sen55', size = 14)
pyplot.show()

#pd.set_option('display.max_columns', None)
#avg_predict.describe(include= 'all')

sns.reset_defaults()
g=sns.relplot(data=avg_predict, x='predicted_pm_saq',y='sample_measurement', color='#ba0c2f', kind='scatter',
              height=3.5,aspect=2)
sns.regplot(x='predicted_pm_saq', y='sample_measurement', data=avg_predict, scatter=False, ax=g.ax, color="black")
pyplot.text(5.5, 21, s=r'$R^2$ = 0.85', fontsize=12, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=1))
pyplot.ylim(0)
#pyplot.xlim(3.5)
pyplot.ylabel('EPA Values ' r'$PM_{2.5} ug/m^3$', fontsize=12)
pyplot.xlabel('Predicted SimpleAQ Values ' r'$PM_{2.5} ug/m^3$', fontsize=12)
#pyplot.legend(title='%Humidity')
pyplot.title('EPA vs Predicted SimpleAQ', fontsize='18')
pyplot.show()
#g.figure.savefig('Fig6.jpg', dpi=300, bbox_inches='tight')

#boxplot
sns.set(style='whitegrid')
palette = ['#D9720D', 'green']
sns.boxplot(data=avg_predict[['sample_measurement', 'predicted_pm_saq']], palette=palette,
            width=0.7, linewidth=1, gap=0, medianprops={"color": "black", "linewidth": 2})
pyplot.ylabel('Particulate Matter Values')
pyplot.xlabel('Sensors')
pyplot.show()

fig, axes = pyplot.subplots(2, 1, figsize=(15, 15))

# Plot a histogram on each subplot
sns.histplot(data=avg_predict['predicted_pm_saq'], color='green', alpha=0.5, kde=True, ax=axes[0], bins=20)
axes[0].set_title('SimpleAQ', fontsize=16)
axes[0].set_ylabel('')
axes[0].set_xlabel('')

sns.histplot(data=avg_predict['sample_measurement'], color='#D9720D', alpha=0.5, kde=True, ax=axes[1], bins=20)
axes[1].set_title('EPA', fontsize=16)
axes[1].set_ylabel('')
axes[1].set_xlabel('')

#for ax in axes:
 #   ax.tick_params(axis='both', which='both', bottom=True, labelbottom=True)
    

fig.supxlabel('PM2.5 Measurements', fontsize=24, x=0.5, y=0)
fig.supylabel('Frequency', fontsize=24, x=(-0.01), y=0.5)
pyplot.tight_layout() # Adjust layout to prevent overlap
pyplot.show()

#subset = avg_predict[avg_predict['datetime'].between('2025-08-08 00:00', '2025-08-09 00:00')]
#subset['datetime'] = pd.to_datetime(subset['datetime'])

#%%
#RELATIVE HUMIDITY
slope, intercept, r_value, p_value, std_err = stats.linregress(avg_predict['sample_measurement'], avg_predict['predicted_pm_rh_saq'])
r_squared = r_value**2
print(r_squared)
#%%
fig, ax = pyplot.subplots(figsize = (10,8))
scatter = ax.scatter(avg_predict['predicted_pm_rh_saq'], avg_predict['sample_measurement'])
#legend = ax.legend(*scatter.legend_elements(), loc="lower right",
                  # title="EPA v Sen55 (SimpleAQ)")
#ax.add_artist(legend)
ax.set_xlabel('Predicted SimpleAQ Values w/ RH')
ax.set_ylabel('EPA Values')
#pyplot.ylim(0)
#pyplot.xlim(0)
#ax.set_title('Plantower v Sen55', size = 14)
pyplot.show()

#pd.set_option('display.max_columns', None)
#avg_predict.describe(include= 'all')

sns.reset_defaults()
g=sns.relplot(data=avg_predict, x='predicted_pm_rh_saq',y='sample_measurement', color='#ba0c2f', kind='scatter',
              height=3.5,aspect=2)
sns.regplot(x='predicted_pm_rh_saq', y='sample_measurement', data=avg_predict, scatter=False, ax=g.ax, color="black")
pyplot.text(5.5, 21, s=r'$R^2$ = 0.86', fontsize=12, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=1))
pyplot.ylim(0)
pyplot.xlim(3.5)
pyplot.ylabel('EPA Values ' r'$PM_{2.5} ug/m^3$', fontsize=12)
pyplot.xlabel('Predicted SimpleAQ Values w/ RH ' r'$PM_{2.5} ug/m^3$', fontsize=12)
#pyplot.legend(title='%Humidity')
pyplot.title('EPA vs Predicted SimpleAQ', fontsize='18')
pyplot.show()
#g.figure.savefig('Fig6.jpg', dpi=300, bbox_inches='tight')

#boxplot
sns.set(style='whitegrid')
palette = ['#D9720D', 'blue']
sns.boxplot(data=avg_predict[['sample_measurement', 'predicted_pm_rh_saq']], palette=palette,
            width=0.7, linewidth=1, gap=0, medianprops={"color": "black", "linewidth": 2})
pyplot.ylabel('Particulate Matter Values')
pyplot.xlabel('Sensors')
pyplot.show()

fig, axes = pyplot.subplots(2, 1, figsize=(15, 15))

# Plot a histogram on each subplot
sns.histplot(data=avg_predict['predicted_pm_rh_saq'], color='blue', alpha=0.5, kde=True, ax=axes[0], bins=20)
axes[0].set_title('SimpleAQ RH', fontsize=16)
axes[0].set_ylabel('')
axes[0].set_xlabel('')

sns.histplot(data=avg_predict['sample_measurement'], color='#D9720D', alpha=0.5, kde=True, ax=axes[1], bins=20)
axes[1].set_title('EPA', fontsize=16)
axes[1].set_ylabel('')
axes[1].set_xlabel('')

#for ax in axes:
 #   ax.tick_params(axis='both', which='both', bottom=True, labelbottom=True)
    

fig.supxlabel('PM2.5 Measurements', fontsize=24, x=0.5, y=0)
fig.supylabel('Frequency', fontsize=24, x=(-0.01), y=0.5)
pyplot.tight_layout() # Adjust layout to prevent overlap
pyplot.show()

#subset = avg_predict[avg_predict['datetime'].between('2025-08-08 00:00', '2025-08-09 00:00')]
#subset['datetime'] = pd.to_datetime(subset['datetime'])

#%%
#TEMPERATURE
slope, intercept, r_value, p_value, std_err = stats.linregress(avg_predict['sample_measurement'], avg_predict['predicted_pm_temp_saq'])
r_squared = r_value**2
print(r_squared)
#%%
fig, ax = pyplot.subplots(figsize = (10,8))
scatter = ax.scatter(avg_predict['predicted_pm_temp_saq'], avg_predict['sample_measurement'])
#legend = ax.legend(*scatter.legend_elements(), loc="lower right",
                  # title="EPA v Sen55 (SimpleAQ)")
#ax.add_artist(legend)
ax.set_xlabel('Predicted SimpleAQ Values w/ Temp')
ax.set_ylabel('EPA Values')
#pyplot.ylim(0)
#pyplot.xlim(0)
#ax.set_title('Plantower v Sen55', size = 14)
pyplot.show()

#pd.set_option('display.max_columns', None)
#avg_predict.describe(include= 'all')

sns.reset_defaults()
g=sns.relplot(data=avg_predict, x='predicted_pm_temp_saq',y='sample_measurement', color='#ba0c2f', kind='scatter',
              height=3.5,aspect=2)
sns.regplot(x='predicted_pm_temp_saq', y='sample_measurement', data=avg_predict, scatter=False, ax=g.ax, color="black")
pyplot.text(5.5, 21, s=r'$R^2$ = 0.86', fontsize=12, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=1))
pyplot.ylim(0)
pyplot.xlim(3.5)
pyplot.ylabel('EPA Values ' r'$PM_{2.5} ug/m^3$', fontsize=12)
pyplot.xlabel('Predicted SimpleAQ Values w/ Temp ' r'$PM_{2.5} ug/m^3$', fontsize=12)
#pyplot.legend(title='%Humidity')
pyplot.title('EPA vs Predicted SimpleAQ w/ Temp', fontsize='18')
pyplot.show()
#g.figure.savefig('Fig6.jpg', dpi=300, bbox_inches='tight')

#boxplot
sns.set(style='whitegrid')
palette = ['#D9720D', 'red']
sns.boxplot(data=avg_predict[['sample_measurement', 'predicted_pm_temp_saq']], palette=palette,
            width=0.7, linewidth=1, gap=0, medianprops={"color": "black", "linewidth": 2})
pyplot.ylabel('Particulate Matter Values')
pyplot.xlabel('Sensors')
pyplot.show()

fig, axes = pyplot.subplots(2, 1, figsize=(15, 15))

# Plot a histogram on each subplot
sns.histplot(data=avg_predict['predicted_pm_temp_saq'], color='red', alpha=0.5, kde=True, ax=axes[0], bins=20)
axes[0].set_title('SimpleAQ Temp', fontsize=16)
axes[0].set_ylabel('')
axes[0].set_xlabel('')

sns.histplot(data=avg_predict['sample_measurement'], color='#D9720D', alpha=0.5, kde=True, ax=axes[1], bins=20)
axes[1].set_title('EPA', fontsize=16)
axes[1].set_ylabel('')
axes[1].set_xlabel('')

#for ax in axes:
 #   ax.tick_params(axis='both', which='both', bottom=True, labelbottom=True)
    

fig.supxlabel('PM2.5 Measurements', fontsize=24, x=0.5, y=0)
fig.supylabel('Frequency', fontsize=24, x=(-0.01), y=0.5)
pyplot.tight_layout() # Adjust layout to prevent overlap
pyplot.show()
#%%
#JOINT PLOTS
sns.set(style='whitegrid')
palette = ['#D9720D','green','blue','red']
label = ['EPA','SAQ PLAIN','SAQ RH','SAQ TEMP']
ax = sns.boxplot(data=avg_predict[['sample_measurement','predicted_pm_saq','predicted_pm_rh_saq', 'predicted_pm_temp_saq']], palette=palette,
            width=0.7, linewidth=1, gap=0, medianprops={"color": "black", "linewidth": 2})
ax.set_xticklabels(label)
pyplot.ylabel('Particulate Matter Hourly Values')
pyplot.xlabel('Sensors')
pyplot.show()

palette = ['#D9720D','green','blue','red']
label = ['EPA','SAQ PLAIN','SAQ RH','SAQ TEMP']
ax = sns.violinplot(data=avg_predict[['sample_measurement','predicted_pm_saq','predicted_pm_rh_saq', 'predicted_pm_temp_saq']], palette=palette,
            width=0.75, linewidth=1.5, gap=0)
ax.set_xticklabels(label)
pyplot.ylabel('Particulate Matter Hourly Values')
pyplot.xlabel('Sensors')
pyplot.show()
'''
#joint scatter
sns.reset_defaults()
fig, ax = pyplot.subplots(figsize=(8,6))
#g=sns.relplot(data=avg_predict, x='predicted_pm_temp_saq',y='mean_pm_pa', color='red', kind='scatter', ax=ax, color='black')
sns.regplot(x='predicted_pm_temp_saq', y='mean_pm_pa', data=avg_predict, color='red', ax=ax, label='Temperature')
sns.regplot(x='predicted_pm_rh_saq', y='mean_pm_pa', data=avg_predict, color='blue', ax=ax, marker='x', label='Relative Humidity')
#pyplot.text(7, 18, s=r'$R^2$ = 0.89', fontsize=12, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=1))
pyplot.ylim(4)
pyplot.xlim(4)
pyplot.ylabel('Corrected Purple Air Values ' r'$PM_{2.5} ug/m^3$')
pyplot.xlabel('Predicted SimpleAQ Values ' r'$PM_{2.5} ug/m^3$')
pyplot.legend(title='Dependent Variables')
pyplot.title('Plantower v Sen55')
pyplot.show()
'''
#%%
#correlation matrix
correlate = avg_predict[['sample_measurement','predicted_pm_saq','predicted_pm_rh_saq', 'predicted_pm_temp_saq']]
correlate.columns = ['EPA','Plain','Rh','Temp']
matrix = correlate.corr()
pyplot.figure(figsize=(8,6))
sns.heatmap(matrix, annot=True, cmap="crest", fmt=".2f", linewidths=0.5, vmin=0.9, vmax=1)
pyplot.xticks(fontsize=9)
pyplot.yticks(fontsize=9)
pyplot.title("Hourly Means Correlation Heatmap")
pyplot.show()

#%%
#LINE PLOT
sns.reset_defaults()
avg_predict['time'] = pd.to_datetime(avg_predict['time'])
fig, axs = pyplot.subplots(3, figsize=(16, 12), constrained_layout=True, sharex=True)
fig.suptitle('EPA vs SimpleAQ Sensors', fontsize=24)
fig.supxlabel('Time (hours)', fontsize=20)
fig.supylabel(r'$PM_{2.5} ug/m^3$', fontsize=20)

date_format = mdates.DateFormatter("%b %d")   # e.g. "Jan 15"
locator = mdates.AutoDateLocator()

# Plot scatter plots for each dataframe
axs[0].plot(avg_predict['time'], avg_predict['predicted_pm_saq'], color='green', label='Plain SAQ')
axs[0].plot(avg_predict['time'], avg_predict['sample_measurement'], color='#D9720D', label='EPA')
#axs[0].set_ylim(bottom =0,top=25)
#axs[0].set_xticklabels(avg_predict['time'], rotation=45)
axs[0].set_ylim(bottom=0)
#axs[0].set_title('EPA', fontsize=16)
axs[0].legend()

axs[1].plot(avg_predict['time'], avg_predict['predicted_pm_rh_saq'], color='b', label= 'Rh SAQ')
axs[1].plot(avg_predict['time'], avg_predict['sample_measurement'], color='#D9720D', label= 'EPA')
#axs[1].set_ylim(bottom=0,top=25)
axs[1].set_ylim(bottom=0)
axs[1].legend()

axs[2].plot(avg_predict['time'], avg_predict['predicted_pm_temp_saq'], color='r', label='Temp SAQ')
axs[2].plot(avg_predict['time'], avg_predict['sample_measurement'], color='#D9720D', label= 'EPA')
#axs[0].set_ylim(bottom =0,top=25)
axs[2].set_ylim(bottom=0)
axs[2].legend()


for ax in axs.flat:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(date_format)
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
# Adjust layout and display the plot
#fig.text(0.5, 0.04, 'Time (Days)', ha='center', va='center', fontsize=16)
#fig.text(0.075, 0.5, (r'$PM_{2.5} ug/m^3$'), ha='center', va='center', rotation='vertical', fontsize=16)
#fig.tight_layout(pad=2.0)
#pyplot.tight_layout()
#pyplot.savefig('reviewfig.png')
pyplot.show()

