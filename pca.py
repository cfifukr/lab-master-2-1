
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets.values.flatten()  

df = pd.DataFrame(X, columns=wine.data.feature_names)
df['class'] = y

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = pd.Categorical(df[col]).codes

print("Інформація про датасет після перетворення:")
df.info()

mean_values = df.groupby('class').mean()

mean_values.plot(kind='bar', figsize=(12, 6))

plt.title('Середні значення характеристик вин за класами')
plt.xlabel('Клас вина')
plt.ylabel('Середнє значення')
plt.xticks(rotation=0) 
plt.legend(title='Характеристики', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


pca = PCA()
pca.fit(scaled_data)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('PCA of Wine Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA()
pca.fit(scaled_data)

explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))

plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='red', alpha=0.7, label='Пояснена дисперсія')

plt.bar(range(1, len(cumulative_variance) + 1), cumulative_variance, color='orange', alpha=0.5, label='Кумулятивна пояснена дисперсія')

for i, v in enumerate(explained_variance):
    plt.text(i + 1, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

for i, v in enumerate(cumulative_variance):
    plt.text(i + 1, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

plt.title('Пояснена та Кумулятивна пояснена дисперсія за головними компонентами')
plt.xlabel('Головна компонента')
plt.ylabel('Дисперсія')
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(axis='y')
plt.legend()
plt.show()


pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

pca_result_df = pd.DataFrame({'pca_1': pca_result[:,0], 'pca_2': pca_result[:,1], 'label': y})

plt.figure(figsize=(12, 8))
sns.scatterplot(x='pca_1', y='pca_2', 
                hue='label', 
                data=pca_result_df, 
                s=50, 
                alpha=1)  

plt.title('PCA Visualization', fontsize=14, pad=15)
plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)

plt.legend(title='Wine', 
          bbox_to_anchor=(1.05, 1), 
          loc='upper left',
          fontsize=10)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()