# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 11:51:05 2025

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



#JP = pd.read_csv('A:/SimpleAQ/JP_1/JP_malfunction_2.csv')
#JP = pd.read_csv('A:/SimpleAQ/Jackson_Pike/OSUJP.csv')
#pd.set_option('display.max_columns', None)
#JP.describe()

#%%
JP = pd.read_csv('A:/SimpleAQ/Jackson_Pike/OSUJP.csv')
pd.set_option('display.max_columns', None)
JP.describe()
time_cols = [c for c in JP.columns if "time" in c.lower()]
JP[time_cols] = JP[time_cols].apply(pd.to_datetime)

def hourly_mean(JP, value_cols, time_col, tz_local="US/Eastern"):
    return (
        JP[value_cols + [time_col]]
        .dropna(subset=[time_col])
        .rename(columns={time_col: "time"})
        .assign(
            time=lambda x: pd.to_datetime(x["time"], utc=True)
                             .dt.tz_convert(tz_local)
                             .dt.tz_localize(None)
        )
        .set_index("time")
        .resample("1H")
        .mean()
    )


OSUA = hourly_mean(JP, ["rh_A"], "rhAtime") \
       .join(hourly_mean(JP, ["temp_A"], "tempAtime")) \
       .join(hourly_mean(JP, ["pm_A"], "timestamp_A"))
OSUB = hourly_mean(JP, ["rh_B"], "rhBtime") \
       .join(hourly_mean(JP, ["temp_B"], "tempBtime")) \
       .join(hourly_mean(JP, ["pm_B"], "timestamp_B"))       
OSUD = hourly_mean(JP, ["rh_D"], "rhDtime") \
       .join(hourly_mean(JP, ["temp_D"], "tempDtime")) \
       .join(hourly_mean(JP, ["pm_D"], "timestamp_D"))

lowcost_hourly = (
    OSUA
    .join(OSUB, how="outer")
    .join(OSUD, how="outer")
)
lowcost_hourly.to_csv('A:/SimpleAQ/Jackson_Pike/Jan_SAQ.csv')

epa_hourly = pd.read_csv("A:/SimpleAQ/Jackson_Pike/0040_hourly.csv")
epa_hourly['datetime'] = pd.to_datetime(epa_hourly['datetime'])
epa_hourly = epa_hourly.set_index('datetime').sort_index()
lowcost_hourly = pd.read_csv('A:/SimpleAQ/Jackson_Pike/Jan_SAQ.csv')
lowcost_hourly['time'] = pd.to_datetime(lowcost_hourly['time'])
lowcost_hourly = lowcost_hourly.set_index('time').sort_index()

comparison_hourly = lowcost_hourly.join(epa_hourly, how="inner")
comparison_hourly.to_csv('A:/SimpleAQ/Jackson_Pike/JP_comparison_hourly.csv')

epa_daily = pd.read_csv("A:/SimpleAQ/Jackson_Pike/0040_daily.csv")
epa_daily["date"] = pd.to_datetime(epa_daily["date"])
epa_daily = epa_daily.set_index("date").sort_index()

lowcost_daily = lowcost_hourly.resample("24h").mean()
lowcost_daily.to_csv('A:/SimpleAQ/Jackson_Pike/Jan_SAQ_daily.csv')


comparison_daily = lowcost_daily.join(epa_daily, how="inner")
comparison_daily.to_csv('A:/SimpleAQ/Jackson_Pike/JP_comparison_daily.csv')
#%%
from Spy.ccc import concordance_correlation_coefficient


#JPB = JP_BD['reading_B']
#JPD = JP_BD['reading_D']
lowcost_hourly = pd.read_csv('A:/SimpleAQ/Jackson_Pike/Jan_SAQ.csv')
lowcost_hourly_valid = lowcost_hourly.dropna()
lowcost_daily = pd.read_csv('A:/SimpleAQ/Jackson_Pike/Jan_SAQ_daily.csv')
comparison_daily = pd.read_csv('A:/SimpleAQ/Jackson_Pike/JP_comparison_daily.csv')

cut_off = ('2025-11-20')
JP_cut = lowcost_hourly[lowcost_hourly['time'] > cut_off]

JPA_nan = JP_cut['pm_A']
JPA = lowcost_hourly_valid['pm_A'] 
JPB_nan = JP_cut['pm_B']
JPB = lowcost_hourly_valid['pm_B'] 
JPD_nan = JP_cut['pm_D']
JPD = lowcost_hourly_valid['pm_D'] 

#correlate
corr = lowcost_hourly[["pm_A","pm_B","pm_D"]].dropna().corr()
print(corr)
#correlate where they match 
match_corr = lowcost_hourly[["pm_A","pm_B","pm_D"]].corr()
print(match_corr)

pyplot.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="crest", fmt=".3f", linewidths=0.5, vmin=0.9, vmax=1)
pyplot.xticks(fontsize=9)
pyplot.yticks(fontsize=9)
pyplot.title("JP SimpleAQ Hourly Correlation Heatmap")
pyplot.show()

pyplot.figure(figsize=(8,6))
sns.heatmap(match_corr, annot=True, cmap="crest", fmt=".3f", linewidths=0.5, vmin=0.9, vmax=1)
pyplot.xticks(fontsize=9)
pyplot.yticks(fontsize=9)
pyplot.title("JP SimpleAQ Hourly Correlation Heatmap (All Matches)")
pyplot.show()
#%%
ccc = concordance_correlation_coefficient(JPA_nan, JPB_nan)
print(ccc['ccc'])
A = (f'CCC = {ccc["ccc"]:.3f}')
slope, intercept, r_value, p_value, std_err = stats.linregress(JPA, JPB)
r_squared = r_value**2
print(f'y = {slope:.2f}x + {intercept:.2f}; R2 = {r_squared:.2f}')
A2 = (f'y = {slope:.2f}x + {intercept:.2f}; R2 = {r_squared:.2f}')

#%%
ccc = concordance_correlation_coefficient(JPA_nan, JPD_nan)
print(ccc['ccc'])
B = (f'CCC = {ccc["ccc"]:.3f}')
slope, intercept, r_value, p_value, std_err = stats.linregress(JPA, JPB)
r_squared = r_value**2
print(f'y = {slope:.2f}x + {intercept:.2f}; R2 = {r_squared:.2f}')
B2 = (f'y = {slope:.2f}x + {intercept:.2f}; R2 = {r_squared:.2f}')

#%%
ccc = concordance_correlation_coefficient(JPB_nan, JPD_nan)
print(ccc['ccc'])
C = (f'CCC = {ccc["ccc"]:.3f}')
slope, intercept, r_value, p_value, std_err = stats.linregress(JPB, JPD)
r_squared = r_value**2
print(f'y = {slope:.2f}x + {intercept:.2f}; R2 = {r_squared:.2f}')
C2 = (f'y = {slope:.2f}x + {intercept:.2f}; R2 = {r_squared:.2f}')

#%%
fig, axs = pyplot.subplots(1, 3, figsize=(14,4))
#fig.suptitle('Site A', size=20)
axs[0].scatter(JPA, JPB, color='black')
axs[0].set_xlabel('Sensor A', size=12)
axs[0].set_ylabel('Sensor B', size=12)
axs[0].set_xlim(0, 30)
axs[0].set_ylim(0, 30)
axs[0].text(0, 1.12, A, transform=axs[0].transAxes, size=16)
axs[0].text(0, 1.05, A2, transform=axs[0].transAxes, size=16)
axs[0].text(28,0.5, '1',size=18)

axs[1].scatter(JPA, JPD, color='black')
axs[1].set_xlabel('Sensor A', size=12)
axs[1].set_ylabel('Sensor D', size=12)
axs[1].set_xlim(0, 30)
axs[1].set_ylim(0, 30)
axs[1].text(0, 1.12, B, transform=axs[1].transAxes, size=16)
axs[1].text(0, 1.05, B2, transform=axs[1].transAxes, size=16)
axs[1].text(28,0.5, '2',size=18)

axs[2].scatter(JPB, JPD, color='black')
axs[2].set_xlabel('Sensor B', size=12)
axs[2].set_ylabel('Sensor D', size=12)
axs[2].set_xlim(0, 30)
axs[2].set_ylim(0, 30)
axs[2].text(0, 1.12, C, transform=axs[2].transAxes, size=16)
axs[2].text(0, 1.05, C2, transform=axs[2].transAxes, size=16)
axs[2].text(28,0.5, '3',size=18)

pyplot.show()
#%%
'''
pyplot.plot(lowcost_hourly_valid['time'], lowcost_hourly_valid['pm_A'], color='green', label='Sensor A')
pyplot.plot(lowcost_hourly_valid['time'], lowcost_hourly_valid['pm_B'], color='blue', label='Sensor B')
pyplot.plot(lowcost_hourly_valid['time'], lowcost_hourly_valid['pm_D'], color='red', label='Sensor D')
pyplot.xlabel('Time')
pyplot.ylabel('PM2.5')
pyplot.title('Sensor Data from Jackson Pike Nans Removed')
pyplot.legend()
pyplot.show()

sns.histplot(data=four['pm_1'], color='#26AB68', kde=True, label='SimpleAQ 1')
sns.histplot(data=four['pm_2'], color='#AB6826', kde=True, label='SimpleAQ 2')
sns.histplot(data=four['pm_3'], color='#6926AB', kde=True, label='SimpleAQ 3')
sns.histplot(data=four['pm_4'], color='#E036A6', kde=True, label='SimpleAQ 4')
sns.histplot(data=combo['pm_saq'], color='black', kde=True, label='SimpleAQ Average')
pyplot.xlabel('PM Readings')
pyplot.ylabel('Frequency')
pyplot.legend()
pyplot.show()

cut_off = pd.to_datetime('2025-11-27')
JP_BD = JP[JP['est_D'] < cut_off]

JPB = JP_BD['reading_B']
JPD = JP_BD['reading_D']
'''
#%%
JP_cut['time'] = pd.to_datetime(JP_cut['time'])

fig, axs = pyplot.subplots(3, figsize=(15, 12),sharex=True)
fig.suptitle('Site 1 Hourly SimpleAQ Values Over 6 Weeks', fontsize=22)
fig.supxlabel('Time (Days)', fontsize=18)
fig.supylabel('Particulate Matter'r'$_{2.5}$'r' $(ug/m^3)$', fontsize=18)
for ax in axs:
    #date_format = mdates.DateFormatter("%b %d")   # e.g. "Jan 15"
    #ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))   # e.g. "Jan 15"
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    
# Plot scatter plots for each dataframe
axs[0].plot(JP_cut['time'], JP_cut['pm_A'], color='red', label='Sensor OSU_A')
axs[0].legend()

axs[1].plot(JP_cut['time'], JP_cut['pm_B'], color='#FF6600', label='Sensor OSU_B')
axs[1].legend()

axs[2].plot(JP_cut['time'], JP_cut['pm_D'], color='green', label= 'Sensor OSU_D')
axs[2].legend(loc='upper left')

pyplot.tight_layout()
pyplot.show()
#%%
#histogram
sns.histplot(lowcost_hourly_valid['pm_A'], color='red', kde=True, label='SimpleAQ A')
sns.histplot(lowcost_hourly_valid['pm_B'], color='#FF6600', kde=True, label='SimpleAQ B')
sns.histplot(lowcost_hourly_valid['pm_D'], color='green', kde=True, label='SimpleAQ D')
#sns.histplot(data=combo['pm_saq'], color='black', kde=True, label='SimpleAQ Average')
pyplot.xlabel('PM'r'$_{2.5}$' ' Readings' r' $(ug/m^3)$')
pyplot.ylabel('Frequency')
pyplot.title('Site 1 Hourly')
pyplot.legend()
pyplot.show()

#%%

average_hourly = comparison_hourly
average_hourly['avg_saq_hourly'] = average_hourly[['pm_A','pm_B','pm_D']].mean(axis=1)
average_hourly['avg_rh_hourly'] = average_hourly[['rh_A','rh_B','rh_D']].mean(axis=1)
average_hourly['avg_temp_hourly'] = average_hourly[['temp_A','temp_B','temp_D']].mean(axis=1)
avg_hour = average_hourly[['sample_measurement','avg_saq_hourly', 'avg_rh_hourly', 'avg_temp_hourly']].copy()
avg_hour.to_csv('A:/SimpleAQ/Jackson_Pike/avg_hourly_JP.csv')

average_daily = comparison_daily
average_daily['avg_saq_daily'] = average_daily[['pm_A','pm_B','pm_D']].mean(axis=1)
average_daily['avg_rh_daily'] = average_daily[['rh_A','rh_B','rh_D']].mean(axis=1)
average_daily['avg_temp_daily'] = average_daily[['temp_A','temp_B','temp_D']].mean(axis=1)
avg_daily = average_daily[['arithmetic_mean','avg_saq_daily', 'avg_rh_daily', 'avg_temp_daily']].copy()
avg_daily.to_csv('A:/SimpleAQ/Jackson_Pike/avg_daily_JP.csv')
#%%
avg_hour = pd.read_csv('A:/SimpleAQ/Jackson_Pike/avg_hourly_JP.csv')
avg_hour = avg_hour.dropna()
ccc = concordance_correlation_coefficient(avg_hour['sample_measurement'], avg_hour['avg_saq_hourly'])
print(ccc['ccc'])
J = (f'CCC = {ccc["ccc"]:.3f}')
slope, intercept, r_value, p_value, std_err = stats.linregress(avg_hour['sample_measurement'], avg_hour['avg_saq_hourly'])
r_squared = r_value**2
squared = "\u00b2"
print(f'y = {slope:.2f}x + {intercept:.2f}; R{squared} = {r_squared:.2f}')
J2 = (f'y = {slope:.2f}x + {intercept:.2f}; R{squared} = {r_squared:.2f}')

#fig, axs = pyplot.subplots(1, 3, figsize=(14,4))
#fig.suptitle(r'Agreement Measures: $R^2$, Slope, & Concordance Correlation Coefficient')
pyplot.scatter(avg_hour['sample_measurement'], avg_hour['avg_saq_hourly'], color='black')
pyplot.xlabel('Site 1 Hourly Average', size=12)
pyplot.ylabel('EPA Observed Hourly', size=12)
#pyplot.title(r'Agreement Measures: $R^2$, Slope, & Concordance Correlation Coefficient', pad=20)
pyplot.title('Agreement Measures for Site 1', size=18, pad=25)
pyplot.xlim(0)
pyplot.ylim(0)
#pyplot.text(0, 44.5, J, size=16)
#pyplot.text(0, 42, J2, size=16)
pyplot.text(1,42, J, size =14)
pyplot.text(11, 42, J2, size =14)
#axs[0].text(28,0.5, '4',size=18)
pyplot.show()
#%%
avg_daily = pd.read_csv('A:/SimpleAQ/Jackson_Pike/avg_daily_JP.csv')
ccc = concordance_correlation_coefficient(avg_daily['arithmetic_mean'], avg_daily['avg_saq_daily'])
print(ccc['ccc'])
K = (f'CCC = {ccc["ccc"]:.3f}')
slope, intercept, r_value, p_value, std_err = stats.linregress(avg_daily['arithmetic_mean'], avg_daily['avg_saq_daily'])
r_squared = r_value**2
print(f'y = {slope:.2f}x + {intercept:.2f}; R2 = {r_squared:.2f}')
K2 = (f'y = {slope:.2f}x + {intercept:.2f}; R2 = {r_squared:.2f}')

#fig, axs = pyplot.subplots(1, 3, figsize=(14,4))
#fig.suptitle(r'Agreement Measures: $R^2$, Slope, & Concordance Correlation Coefficient')
pyplot.scatter(avg_daily['arithmetic_mean'], avg_daily['avg_saq_daily'], color='black')
pyplot.xlabel('SAQ Site 1 Daily Average', size=12)
pyplot.ylabel('EPA', size=12)
pyplot.xlim(0)
pyplot.ylim(0)
pyplot.text(0, 34, K, size=16)
pyplot.text(0, 32, K2, size=16)
#axs[0].text(28,0.5, '4',size=18)
pyplot.show()
