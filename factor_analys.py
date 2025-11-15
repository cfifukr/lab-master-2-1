import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis

from sklearn.preprocessing import StandardScaler

wine = fetch_ucirepo(id=109)
df = wine.data.features  
df['class'] = wine.data.targets

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('class', axis=1))

fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(factors[:, 0], factors[:, 1], c=df['class'], cmap='viridis', alpha=0.7)
plt.title("Факторний аналіз")
plt.xlabel("Фактор 1")
plt.ylabel("Фактор 2")
plt.colorbar(label='Wine Class')
plt.grid(True)
plt.show()
