import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
target = 'attack_detected'

df = df[features + [target]].dropna()

X = df[features].values
y = df[target].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 5  # кількість сусідів
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\nТочність моделі (k={k}): {acc:.3f}")

print("\nМатриця плутанини:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.title(f'Матриця плутанини для KNN (k={k})')
plt.xlabel('Передбачено')
plt.ylabel('Реальне значення')
plt.tight_layout()
plt.show()

print("\nЗвіт класифікації:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# Перевірка впливу k на точність
k_values = range(1, 21)
scores = []
for i in k_values:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

plt.figure(figsize=(8,4))
plt.plot(k_values, scores, marker='o', color='orange')
plt.title('Залежність точності від кількості сусідів (k)')
plt.xlabel('k')
plt.ylabel('Точність')
plt.grid(True)
plt.show()
