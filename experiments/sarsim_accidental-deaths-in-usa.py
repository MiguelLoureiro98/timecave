import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pandas as pd
length=700
lags=1
s=12
seed=2
np.random.seed(seed)

#Data
data = pd.read_csv('accidental-deaths-in-usa-monthly.csv', usecols=[1], names = ['accidental_deaths'], skiprows=1)
ts = data.accidental_deaths.to_numpy()
          
#Model
sar = SARIMAX(ts, order=(1, 0, 0), seasonal_order = (1, 0, 0, 12))
sar_fit = sar.fit()

#Simulation
ts_sar = sar_fit.simulate(length)

#Plots
model = 'sarusa'

plt.plot(ts_sar, c='y', label ='SAR TS')
plt.title('Synthetic Time Series')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(data.index, data['accidental_deaths'], label='Observed')
plt.plot(data.index, sar_fit.fittedvalues, label='Fitted', color='red')
plt.title('Seasonal AR Model - Fitted vs Observed')
plt.legend()
plt.show()

