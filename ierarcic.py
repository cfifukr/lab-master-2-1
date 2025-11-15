import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

 

df = pd.read_csv('cybersecurity_intrusion_data.csv')
print("Кількість рядків і стовпців:", df.shape)
print("Назви колонок:", df.columns.tolist())

features = [
    'network_packet_size',
    'login_attempts',
    'session_duration',
    'ip_reputation_score',
    'failed_logins'
]

X = df[features].select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method='ward')  # метод Уорда — мінімізує суму квадратів відстаней

plt.figure(figsize=(10, 6))
plt.title("Дендрограма ієрархічної кластеризації ")
plt.xlabel("Об’єкти (зразки)")
plt.ylabel("Відстань")
dendrogram(Z, leaf_rotation=90)
plt.tight_layout()
plt.show()

clusters = fcluster(Z, t=3, criterion='maxclust')
df['Cluster'] = clusters

print("\nРезультати кластеризації:")
print(df[['network_packet_size', 'login_attempts', 'session_duration', 'ip_reputation_score', 'failed_logins', 'Cluster']].head(15))

cluster_summary = df.groupby('Cluster')[features].mean()
print("\nСередні значення ознак у кожному кластері:")
print(cluster_summary)