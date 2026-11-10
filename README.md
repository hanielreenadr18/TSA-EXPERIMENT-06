# Experiment 06 – Exponential Smoothing Methods

### Name: HANIEL REENA D R
### Register No: 2305001008

## Aim
To implement and compare three exponential smoothing techniques for time-series forecasting:
Simple Exponential Smoothing, Holt's Linear Trend Method, and Holt-Winters Seasonal Method.

## Algorithm
1. Start the program.
2. Import Pandas, Matplotlib, and the required exponential smoothing models.
3. Apply Simple Exponential Smoothing to sales data without a clear trend or seasonal pattern.
4. Forecast the next three values.
5. Apply Holt's method to sales data containing an increasing trend.
6. Forecast the next three values using Holt's method.
7. Apply Holt-Winters to seasonal sales data using additive trend and seasonality.
8. Forecast the next four values using Holt-Winters.
9. Plot the actual and fitted values for all three methods.
10. Compare the results and stop the program.


## Program
File: `experiment_06.py`

```python
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
```

## Output

<img width="930" height="500" alt="image" src="https://github.com/user-attachments/assets/467d3221-c38e-446f-a72f-4339775db44e" />

<img width="940" height="510" alt="image" src="https://github.com/user-attachments/assets/995aa8c6-c07a-441a-925e-9581d1f23d5a" />

<img width="940" height="540" alt="image" src="https://github.com/user-attachments/assets/dcc873c2-a7e4-4fe7-a8cc-ea1e65b2d071" />


## Observation
Simple Exponential Smoothing follows the level of the data, Holt's method follows the increasing
trend, and Holt-Winters captures both the trend and the repeating seasonal pattern.

## Result
Thus, Simple Exponential Smoothing, Holt's Method, and Holt-Winters Method were successfully
implemented and compared for time-series forecasting.
