import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets['Y1']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Коефіцієнти регресії:", model.coef_)
print("Вільний член (intercept):", model.intercept_)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='orange', label='Прогноз')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Ідеальна лінія')
plt.title("Реальне vs Прогнозоване значення Heating Load (Y1)")
plt.xlabel("Реальне значення Y1")
plt.ylabel("Прогнозоване значення Y1")
plt.legend()
plt.grid(True)
plt.show()
