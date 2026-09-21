# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:38:59 2026

@author: princ
"""

import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
import seaborn as sns
from scipy import stats
from datetime import timedelta
import matplotlib.dates as mdates
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


SR = pd.read_csv('A:/SimpleAQ/Smoky_Row/avg_hourly_SR_corrected.csv')
JP = pd.read_csv('A:/SimpleAQ/Jackson_Pike/avg_hourly_JP_corrected.csv')
squared = "\u00b2"
#%%
from Spy.ccc import concordance_correlation_coefficient

ccc = concordance_correlation_coefficient(SR['epa_pm'], SR['corrected_SR'])
print(ccc['ccc'])
D = (f'CCC = {ccc["ccc"]:.3f}')
slope, intercept, r_value, p_value, std_err = stats.linregress(SR['epa_pm'], SR['corrected_SR_log'])
r_squared = r_value**2
print(f'y = {slope:.2f}x + {intercept:.2f}; R{squared} = {r_squared:.2f}')
D2 = (f'y = {slope:.2f}x + {intercept:.2f}; R{squared} = {r_squared:.2f}')

#fig, axs = pyplot.subplots(1, 3, figsize=(14,4))
#fig.suptitle(r'Agreement Measures: $R^2$, Slope, & Concordance Correlation Coefficient')
pyplot.scatter(SR['epa_pm'], SR['corrected_SR_log'], color='black')
pyplot.title('Corrected Agreement Measures for Site 2', size=18, pad=25)
pyplot.xlabel('Site 2 Corrected Hourly Average', size=12)
pyplot.ylabel('EPA Observed Hourly', size=12)
pyplot.xlim(0)
pyplot.ylim(0)
pyplot.text(2, 27, D, size=14)
pyplot.text(15, 27, D2, size=14)
#axs[0].text(28,0.5, '4',size=18)
pyplot.show()

print(f'y = {slope:.2f}x + {intercept:.2f}; R{squared} = {r_squared:.2f}')
D2 = (f'y = {slope:.2f}x + {intercept:.2f}; R{squared} = {r_squared:.2f}')

#%%
#histogram
sns.histplot(SR['epa_pm'], color='#60D2E0', kde=True, label='EPA')
sns.histplot(SR['corrected_SR_log'], color='blue', kde=True, label='SimpleAQ Log SR')
sns.histplot(SR['corrected_SR'], color='red', kde=True, label='SimpleAQ SR')
pyplot.xlabel('PM Readings')
pyplot.ylabel('Frequency')
pyplot.title('Hourly')
pyplot.legend()
pyplot.show()

#%%

SR['Datetime'] = pd.to_datetime(SR['Datetime'])
'''
fig, axs = pyplot.subplots(1, figsize=(15, 12), sharex=True)
fig.suptitle('Hourly SR SimpleAQ Values Over 10 Weeks', fontsize=22)
fig.supxlabel('Time (Days)', fontsize=18)
fig.supylabel('Particulate Matter'r'$_{2.5}$'r' $(ug/m^3)$', fontsize=18)
for ax in axs:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))   # e.g. "Jan 15"
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

# Plot scatter plots for each dataframe
pyplot.plot(SR['Datetime'], SR['epa_pm'], color='#60D2E0', label='EPA')
pyplot.plot(SR['Datetime'], SR['corrected_SR_log'], color='blue', label='Corrected SR')
pyplot.plot(SR['Datetime'], SR['avg_saq_hourly'], color='#82A6E0', label= 'SR')
pyplot.axhline(9, linestyle='--',color='black')

pyplot.title('Smoky Row Time Series')
pyplot.xlabel('Time (Days)', fontsize=18)
pyplot.ylabel('Particulate Matter'r'$_{2.5}$'r' $(ug/m^3)$', fontsize=18)
pyplot.legend()
#pyplot.tight_layout()
pyplot.show()
'''
#%%


SR['Datetime'] = pd.to_datetime(SR['Datetime'])
#SR = SR.sort_values('Datetime')
time_diff = SR['Datetime'].diff().dt.total_seconds() / 3600

# Mask large gaps (e.g., >2 hours)
SR.loc[time_diff > 2, ['epa_pm','corrected_SR_log','corrected_SR','avg_saq_hourly']] = None

fig, ax = pyplot.subplots(figsize=(15, 6))

# Plot lines
ax.plot(SR['Datetime'], SR['epa_pm'], color='black', label='EPA Observed')
#ax.plot(SR['Datetime'], SR['corrected_SR_log'], color='blue', label='Corrected Log SR')
#ax.plot(SR['Datetime'], SR['avg_saq_hourly'], color='#82A6E0', label='SR')
ax.plot(SR['Datetime'], SR['corrected_SR'], color='red', label='Site 2 Corrected')

ax.axhline(9, linestyle='--', color='black', label='Annual PM2.5 NAAQS')
ax.axhline(35, linestyle='--', color='black', label='Daily PM2.5 NAAQS')

# Titles and labels
ax.set_title('Site 2 Time Series', size=24)
ax.set_xlabel('Time (Days)', fontsize=18)
ax.set_ylabel('Particulate Matter' + r'$_{2.5}$' + r' $(ug/m^3)$', fontsize=18)

# ✅ Fix x-axis formatting
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# ✅ Rotate ticks properly
pyplot.setp(ax.get_xticklabels(), rotation=30, ha='right')

ax.legend()
pyplot.tight_layout()
pyplot.show()

#%%
#Jackson Pike

ccc = concordance_correlation_coefficient(JP['sample_measurement'], JP['corrected_JP'])
print(ccc['ccc'])
E = (f'CCC = {ccc["ccc"]:.3f}')
slope, intercept, r_value, p_value, std_err = stats.linregress(JP['sample_measurement'], JP['corrected_JP_log'])
r_squared = r_value**2
print(f'y = {slope:.2f}x + {intercept:.2f}; R{squared} = {r_squared:.2f}')
E2 = (f'y = {slope:.2f}x + {intercept:.2f}; R{squared} = {r_squared:.2f}')

#fig, axs = pyplot.subplots(1, 3, figsize=(14,4))
#fig.suptitle(r'Agreement Measures: $R^2$, Slope, & Concordance Correlation Coefficient')
pyplot.scatter(JP['sample_measurement'], JP['corrected_JP'], color='black')
pyplot.title('Corrected Agreement Measures for Site 1', size=18, pad=25)
pyplot.xlabel('Site 1 Corrected Hourly Average', size=12)
pyplot.ylabel('EPA Observed Hourly', size=12)
pyplot.xlim(0)
pyplot.ylim(0)
pyplot.text(1, 28.5, E, size=14)
pyplot.text(11, 28.5, E2, size=14)
#axs[0].text(28,0.5, '4',size=18)
pyplot.show()
#%%
#histogram
sns.histplot(JP['sample_measurement'], color='#60D2E0', kde=True, label='EPA')
sns.histplot(JP['corrected_JP_log'], color='red', kde=True, label='SimpleAQ Log JP')
sns.histplot(JP['corrected_JP'], color='green', kde=True, label='SimpleAQ JP')
pyplot.xlabel('PM Readings')
pyplot.ylabel('Frequency')
pyplot.title('Hourly')
pyplot.legend()
pyplot.show()


#%%


JP['datetime'] = pd.to_datetime(JP['datetime'])
#JP = JP.sort_values('datetime')
time_diff = JP['datetime'].diff().dt.total_seconds() / 3600

# Mask large gaps (e.g., >2 hours)
JP.loc[time_diff > 2, ['sample_measurement','corrected_JP_log','corrected_JP','avg_saq_hourly']] = None

fig, ax = pyplot.subplots(figsize=(15, 6))

# Plot lines
ax.plot(JP['datetime'], JP['sample_measurement'], color='black', label='EPA Observed')
#ax.plot(JP['datetime'], JP['corrected_JP_log'], color='red', label='Corrected Log JP')
#ax.plot(JP['datetime'], JP['avg_saq_hourly'], color='#DB4037', label='JP')
ax.plot(JP['datetime'], JP['corrected_JP'], color='blue', label='Site 1 Corrected')

ax.axhline(9, linestyle='--', color='black', label='Annual PM2.5 NAAQS')
ax.axhline(35, linestyle='--', color='black', label='Daily PM2.5 NAAQS')

# Titles and labels
ax.set_title('Site 1 Time Series', size=24)
ax.set_xlabel('Time (Days)', fontsize=18)
ax.set_ylabel('Particulate Matter' + r'$_{2.5}$' + r' $(ug/m^3)$', fontsize=18)

# ✅ Fix x-axis formatting
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# ✅ Rotate ticks properly
pyplot.setp(ax.get_xticklabels(), rotation=30, ha='right')

ax.legend(loc='upper right', framealpha=1)
pyplot.tight_layout()
pyplot.show()