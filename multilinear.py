import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

y = df['alcohol']
X = df.drop(columns=['alcohol'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='orange', label='Прогнозовані значення')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Ідеальна лінія (y=x)')

plt.title("Реальне vs Прогнозоване значення Alcohol")
plt.xlabel("Реальне значення Alcohol")
plt.ylabel("Прогнозоване значення Alcohol")
plt.legend()
plt.grid(True)
plt.show()

print("\nКоефіцієнти регресії (вплив ознак):")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.4f}")

print("\nВільний член (intercept):", model.intercept_)
print("\nR² (коефіцієнт детермінації):", r2_score(y_test, y_pred))
print("RMSE (корінь середньоквадратичної помилки):", np.sqrt(mean_squared_error(y_test, y_pred)))
