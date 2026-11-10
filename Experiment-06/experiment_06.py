import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing, Holt, ExponentialSmoothing
)
sales = [100,120,110,115,105,125,118,122,110,119]
data = pd.Series(sales)
model1 = SimpleExpSmoothing(
    data, initialization_method="estimated"
)
fit1 = model1.fit(smoothing_level=0.3, optimized=False)
forecast1 = fit1.forecast(3)
sales_trend = [
    100,105,112,118,125,
    130,138,145,152,160
]
data_trend = pd.Series(sales_trend)
model2 = Holt(
    data_trend, initialization_method="estimated"
)
fit2 = model2.fit()
forecast2 = fit2.forecast(3)
seasonal_sales = [
    100,130,160,120,110,145,175,130,
    120,155,190,140,130,165,205,150
]
data_seasonal = pd.Series(seasonal_sales)
model3 = ExponentialSmoothing(
    data_seasonal,
    trend="add",
    seasonal="add",
    seasonal_periods=4,
    initialization_method="estimated"
)
fit3 = model3.fit()
forecast3 = fit3.forecast(4)
print("Simple Exponential Smoothing")
print(forecast1)
print("\nHolt's Method")
print(forecast2)
print("\nHolt-Winters Method")
print(forecast3)
plt.plot(data, marker='o', label='Actual')
plt.plot(fit1.fittedvalues, marker='o', label='Forecast')
plt.title("Simple Exponential Smoothing")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
plt.plot(data_trend, marker='o', label='Actual')
plt.plot(fit2.fittedvalues, marker='o', label='Holt Forecast')
plt.title("Holt's Method")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
plt.plot(data_seasonal, marker='o', label='Actual')
plt.plot(fit3.fittedvalues, marker='o',
         label='Holt-Winters Forecast')
plt.title("Holt-Winters Method")
plt.xlabel("Quarter")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
