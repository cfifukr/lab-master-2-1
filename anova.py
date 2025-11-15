import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("cybersecurity_intrusion_data.csv")

print("Перші рядки датасету:")
print(data.head(), "\n")

print("Пропуски у даних:\n", data.isnull().sum(), "\n")

groups = [group["network_packet_size"].values for _, group in data.groupby("protocol_type")]

f_stat, p_value = stats.f_oneway(*groups)
print(f"F-статистика: {f_stat:.4f}")
print(f"p-значення: {p_value:.4f}")

if p_value < 0.05:
    print("Є статистично значуща різниця між групами.\n")
else:
    print("Різниця між групами не є статистично значущою (p ≥ 0.05).\n")

plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x="protocol_type", y="network_packet_size", hue="protocol_type", palette="Set2", legend=False)
plt.title("Порівняння розміру мережевих пакетів між протоколами (ANOVA)")
plt.xlabel("Тип протоколу")
plt.ylabel("Розмір мережевого пакету")
plt.grid(alpha=0.3)
plt.show()

print("Перевірка нормальності (тест Шапіро-Вілка):")
for proto, group in data.groupby("protocol_type"):
    stat, p = stats.shapiro(group["network_packet_size"])
    print(f"  {proto}: p={p:.4f}")

levene_stat, levene_p = stats.levene(*groups)
print(f"\n Тест Левена (гомогенність дисперсій): p={levene_p:.4f}\n")

if p_value < 0.05:
    tukey = pairwise_tukeyhsd(
        endog=data['network_packet_size'],
        groups=data['protocol_type'],
        alpha=0.05
    )
    print("Результати тесту Tukey HSD:\n")
    print(tukey)
else:
    print("Post-hoc тест не виконується, бо різниця між групами незначуща.\n")
