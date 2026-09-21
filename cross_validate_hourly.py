# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 15:04:55 2026

@author: princ
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf

original_jp = pd.read_csv('A:/SimpleAQ/Jackson_Pike/avg_hourly_JP.csv')
original_sr = pd.read_csv('A:/SimpleAQ/Smoky_Row/avg_hourly_SR.csv')

average_jp = original_jp.dropna()
average_sr = original_sr.dropna()

average_jp.to_csv('A:/SimpleAQ/Jackson_Pike/avg_hourly_JP_clean.csv')
average_sr.to_csv('A:/SimpleAQ/Smoky_Row/avg_hourly_SR_clean.csv')


#%%

def calibration_analysis(df, predictors, target, site_name):
    
    print(f"\n===== {site_name} =====")
    
    X = df[predictors]
    y = df[target]
    
    # ----- OLS (Full Data) -----
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    print(model.summary())
    
    # ----- Cross-Validated RMSE -----
    kf = KFold(n_splits=5, shuffle=True, random_state=16)
    r2_scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2')
    rmse = np.sqrt(
        -cross_val_score(
            LinearRegression(), 
            X, y, 
            cv=kf,
            scoring='neg_mean_squared_error'
        )
    )
    print("Mean R²:", r2_scores.mean())
    print("Std R²:", r2_scores.std())
    print("Mean CV RMSE:", rmse.mean())
    print("Std CV RMSE:", rmse.std())
    
    # ----- VIF -----
    vif_df = pd.DataFrame()
    vif_df["Variable"] = X_sm.columns
    vif_df["VIF"] = [
        variance_inflation_factor(X_sm.values, i)
        for i in range(X_sm.shape[1])
    ]
    print("\nVIF:")
    print(vif_df)
    
    return model
#%%
def cv_rmse(df, predictors, target):
    
    if isinstance(predictors, str):
        predictors = [predictors]
        
    X = df[predictors]
    y = df[target]
    
    model = LinearRegression()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rmse = np.sqrt(
        -cross_val_score(model, X, y,
                         cv=kf,
                         scoring='neg_mean_squared_error')
    )
    
    return rmse.mean()
#%% SITE 1
print("Sensor only:", cv_rmse(average_jp, ['avg_saq_hourly'], 'sample_measurement'))
print("Sensor + RH:", cv_rmse(average_jp, ['avg_saq_hourly', 'avg_rh_hourly'], 'sample_measurement'))
print("Sensor + Temp:", cv_rmse(average_jp, ['avg_saq_hourly', 'avg_temp_hourly'], 'sample_measurement'))
print("All three:", cv_rmse(average_jp, ['avg_saq_hourly', 'avg_rh_hourly', 'avg_temp_hourly'], 'sample_measurement'))
#%%
model_sensor_1 = calibration_analysis(
    average_jp,
    predictors=['avg_saq_hourly'],
    target='sample_measurement',
    site_name='Site 1 - Sensor Only'
)

model_rh_1 = calibration_analysis(
    average_jp,
    predictors=['avg_saq_hourly', 'avg_rh_hourly'],
    target='sample_measurement',
    site_name='Site 1 - Sensor + RH'
)

model_temp_1 = calibration_analysis(
    average_jp,
    predictors=['avg_saq_hourly', 'avg_temp_hourly'],
    target='sample_measurement',
    site_name='Site 1 - Sensor + Temp'
)

model_all_1 = calibration_analysis(
    average_jp,
    predictors=['avg_saq_hourly', 'avg_rh_hourly', 'avg_temp_hourly'],
    target='sample_measurement',
    site_name='Site 1 - Sensor + RH + Temp'
)
#%%
#anova_results = sm.stats.anova_lm(model_sensor_1, model_rh_1)
#print(anova_results)

print("Sensor only AIC:", model_sensor_1.aic)
print("Sensor + RH AIC:", model_rh_1.aic)
print("Sensor + Temp AIC:", model_temp_1.aic)
print("Sensor + RH + Temp AIC:", model_all_1.aic)
#%%
residuals = model_rh_1.resid
fitted = model_rh_1.fittedvalues

pyplot.scatter(fitted, residuals, color='#A2DDF2')
pyplot.axhline(0, linestyle='--', color='black')
pyplot.xlabel("Fitted")
pyplot.ylabel("Residuals")
pyplot.title("Residual vs Fitted Rh")
pyplot.show()
#%%
residuals = model_sensor_1.resid
fitted = model_sensor_1.fittedvalues

pyplot.scatter(fitted, residuals,color='#0090D3')
pyplot.axhline(0, linestyle='--', color='black')
pyplot.xlabel("Fitted")
pyplot.ylabel("Residuals")
pyplot.title("Residual vs Fitted Solo")
pyplot.show()
#%%
residuals = model_temp_1.resid
fitted = model_temp_1.fittedvalues

pyplot.scatter(fitted, residuals,color='#0027D3')
pyplot.axhline(0, linestyle='--',color='black')
pyplot.xlabel("Fitted")
pyplot.ylabel("Residuals")
pyplot.title("Residual vs Fitted Temp")
pyplot.show()
#%%
residuals = model_all_1.resid
fitted = model_all_1.fittedvalues

pyplot.scatter(fitted, residuals, color='#4300D3')
pyplot.axhline(0, linestyle='--',color='black')
pyplot.xlabel("Fitted")
pyplot.ylabel("Residuals")
pyplot.title("Residual vs Fitted All")
pyplot.show()
#%% SITE 2

print("Sensor only:", cv_rmse(average_sr, ['avg_saq_hourly'], 'epa_pm'))
print("Sensor + RH:", cv_rmse(average_sr, ['avg_saq_hourly', 'avg_rh_hourly'], 'epa_pm'))
print("Sensor + Temp:", cv_rmse(average_sr, ['avg_saq_hourly', 'avg_temp_hourly'], 'epa_pm'))
print("All three:", cv_rmse(average_sr, ['avg_saq_hourly', 'avg_rh_hourly', 'avg_temp_hourly'], 'epa_pm'))
#%%
model_sensor_2 = calibration_analysis(
    average_sr,
    predictors=['avg_saq_hourly'],
    target='epa_pm',
    site_name='Site 2 - Sensor Only'
)

model_rh_2 = calibration_analysis(
    average_sr,
    predictors=['avg_saq_hourly', 'avg_rh_hourly'],
    target='epa_pm',
    site_name='Site 2 - Sensor + RH'
)

model_temp_2 = calibration_analysis(
    average_sr,
    predictors=['avg_saq_hourly', 'avg_temp_hourly'],
    target='epa_pm',
    site_name='Site 2 - Sensor + Temp'
)

model_all_2 = calibration_analysis(
    average_sr,
    predictors=['avg_saq_hourly', 'avg_rh_hourly', 'avg_temp_hourly'],
    target='epa_pm',
    site_name='Site 2 - Sensor + RH + Temp'
)
#%%
#anova_results = sm.stats.anova_lm(model_sensor_2, model_rh_2)
#print(anova_results)

print("Sensor only AIC:", model_sensor_2.aic)
print("Sensor + RH AIC:", model_rh_2.aic)
print("Sensor + Temp AIC:", model_temp_2.aic)
print("Sensor + RH + Temp AIC:", model_all_2.aic)
#%%
residuals = model_rh_2.resid
fitted = model_rh_2.fittedvalues

pyplot.scatter(fitted, residuals, color='#FB3C32')
pyplot.axhline(0, linestyle='--', color='black')
pyplot.xlabel("Fitted")
pyplot.ylabel("Residuals")
pyplot.title("Residual vs Fitted Rh")
pyplot.show()
#%%
residuals = model_sensor_2.resid
fitted = model_sensor_2.fittedvalues

pyplot.scatter(fitted, residuals,color='#FA1105')
pyplot.axhline(0, linestyle='--', color='black')
pyplot.xlabel("Fitted")
pyplot.ylabel("Residuals")
pyplot.title("Residual vs Fitted Solo")
pyplot.show()
#%%
residuals = model_temp_2.resid
fitted = model_temp_2.fittedvalues

pyplot.scatter(fitted, residuals,color='#CD0E04')
pyplot.axhline(0, linestyle='--',color='black')
pyplot.xlabel("Fitted")
pyplot.ylabel("Residuals")
pyplot.title("Residual vs Fitted Temp")
pyplot.show()
#%%
residuals = model_all_2.resid
fitted = model_all_2.fittedvalues

pyplot.scatter(fitted, residuals, color='#730802')
pyplot.axhline(0, linestyle='--',color='black')
pyplot.xlabel("Fitted")
pyplot.ylabel("Residuals")
pyplot.title("Residual vs Fitted All")
pyplot.show()

#%%

def get_kfold_predictions(df, predictors, target, k=5):

    X = df[predictors]
    y = df[target]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    model = LinearRegression()

    y_pred = cross_val_predict(model, X, y, cv=kf)

    results = pd.DataFrame({
        "Observed": y,
        "Predicted": y_pred
    })

    return results

#%%

JP_sensor = get_kfold_predictions(average_jp,
    predictors=['avg_saq_hourly'], target='sample_measurement')
JP_rh = get_kfold_predictions(average_jp,
    predictors=['avg_saq_hourly','avg_rh_hourly'],
    target='sample_measurement')
JP_temp = get_kfold_predictions(average_jp,
    predictors=['avg_saq_hourly','avg_temp_hourly'],
    target='sample_measurement')
JP_all = get_kfold_predictions(average_jp,
    predictors=['avg_saq_hourly','avg_rh_hourly','avg_temp_hourly'],
    target='sample_measurement')

SR_sensor = get_kfold_predictions(average_sr,
    predictors=['avg_saq_hourly'], target='epa_pm')
SR_rh = get_kfold_predictions(average_sr,
    predictors=['avg_saq_hourly','avg_rh_hourly'], target='epa_pm')
SR_temp = get_kfold_predictions(average_sr,
    predictors=['avg_saq_hourly','avg_temp_hourly'], target='epa_pm')
SR_all = get_kfold_predictions(average_sr,
    predictors=['avg_saq_hourly','avg_rh_hourly','avg_temp_hourly'],
    target='epa_pm')

#%%

def plot_cv_results(df, color, title):

    r2 = r2_score(df["Observed"], df["Predicted"])

    pyplot.figure(figsize=(4,4))
    pyplot.scatter(df["Predicted"], df["Observed"], color=color, alpha=0.7)

    # 1:1 line
    min_val = min(df.min())
    max_val = max(df.max())
    pyplot.plot([min_val, max_val],
             [min_val, max_val],
             'k--')

    pyplot.xlabel("SAQ Predicted PM2.5")
    pyplot.ylabel("EPA Observed PM2.5")
    pyplot.title(title)
    pyplot.text(min_val, max_val*0.9, f"$R^2$ = {r2:.3f}")
    pyplot.tight_layout()
    pyplot.show()
    
#%%

plot_cv_results(JP_sensor, '#A2DDF2', 'JP EPA vs Predicted SimpleAQ')
plot_cv_results(JP_rh,'#0090D3', 'JP EPA vs Predicted SimpleAQ w/ RH')
plot_cv_results(JP_temp,'#0027D3', ' JP EPA vs Predicted SimpleAQ w/ Temperature')
plot_cv_results(JP_all,'#4300D3', 'JP EPA vs Predicted SimpleAQ w/ All Terms')

plot_cv_results(SR_sensor,'#FB3C32', 'SR EPA vs Predicted SimpleAQ')
plot_cv_results(SR_rh,'#FA1105', 'SR EPA vs Predicted SimpleAQ w/ RH')
plot_cv_results(SR_temp,'#CD0E04', ' SR EPA vs Predicted SimpleAQ w/ Temperature')
plot_cv_results(SR_all,'#730802', 'SR EPA vs Predicted SimpleAQ w/ All Terms')

#%%

JP1 = average_jp.rename(columns={'sample_measurement':'epa_pm'})
SR2 = average_sr
JP1['site'] = 'JP'
SR2['site'] = 'SR'
combined = pd.concat([JP1, SR2], ignore_index=True)
combined.to_csv('A:/SimpleAQ/combined_avg.csv')

combined = pd.read_csv('A:/SimpleAQ/combined_avg.csv')

#%% COMBINED SITES
print("Sensor only:", cv_rmse(combined, ['avg_saq_hourly'], 'epa_pm'))
print("Sensor + RH:", cv_rmse(combined, ['avg_saq_hourly', 'avg_rh_hourly'], 'epa_pm'))
print("Sensor + Temp:", cv_rmse(combined, ['avg_saq_hourly', 'avg_temp_hourly'], 'epa_pm'))
print("All three:", cv_rmse(combined, ['avg_saq_hourly', 'avg_rh_hourly', 'avg_temp_hourly'], 'epa_pm'))
#%%
model_sensor_combo = calibration_analysis(
    combined,
    predictors=['avg_saq_hourly'],
    target='epa_pm',
    site_name='Combined Sites - Sensor Only'
)

model_rh_combo = calibration_analysis(
    combined,
    predictors=['avg_saq_hourly', 'avg_rh_hourly'],
    target='epa_pm',
    site_name='Combined Sites - Sensor + RH'
)

model_temp_combo = calibration_analysis(
    combined,
    predictors=['avg_saq_hourly', 'avg_temp_hourly'],
    target='epa_pm',
    site_name='Combined Sites - Sensor + Temp'
)

model_all_combo = calibration_analysis(
    combined,
    predictors=['avg_saq_hourly', 'avg_rh_hourly', 'avg_temp_hourly'],
    target='epa_pm',
    site_name='Combined Sites - Sensor + RH + Temp'
)

#%%

anova_results = sm.stats.anova_lm(model_rh_combo, model_all_combo)
print(anova_results)



#%%

combined_sensor = get_kfold_predictions(combined,
    predictors=['avg_saq_hourly'], target='epa_pm')

combined_rh = get_kfold_predictions(combined,
    predictors=['avg_saq_hourly','avg_rh_hourly'],
    target='epa_pm')

combined_temp = get_kfold_predictions(combined,
    predictors=['avg_saq_hourly','avg_temp_hourly'],
    target='epa_pm')

combined_all = get_kfold_predictions(combined,
    predictors=['avg_saq_hourly','avg_rh_hourly','avg_temp_hourly'],
    target='epa_pm')
#%%

plot_cv_results(combined_sensor, '#C25FFC', 'Combined EPA vs Predicted SimpleAQ')
plot_cv_results(combined_rh,'#B132FB', 'Combined EPA vs Predicted SimpleAQ w/ RH')
plot_cv_results(combined_temp,'#8304CD', ' Combined EPA vs Predicted SimpleAQ w/ Temperature')
plot_cv_results(combined_all,'#4A0273', 'Combined EPA vs Predicted SimpleAQ w/ All Terms')

