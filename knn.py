from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd 
import numpy as np 
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.preprocessing import StandardScaler 
import seaborn as sns 
import matplotlib.pyplot as plt 

 

wine = fetch_ucirepo(id=109) 
X = pd.DataFrame(wine.data.features, columns=wine.data.feature_names) 
y = wine.data.targets 

X_cleaned = X.dropna() 
y_cleaned = y[:len(X_cleaned)] 

scaler = StandardScaler() 
X_normalized = scaler.fit_transform(X_cleaned) 
X_normalized_df = pd.DataFrame(X_normalized, columns=X_cleaned.columns) 
X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y_cleaned, test_size=0.3, random_state=42) 

knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train) 

y_pred = knn.predict(X_test) 
conf_matrix = confusion_matrix(y_test, y_pred) 

print(classification_report(y_test, y_pred)) 

 

plt.figure(figsize=(10, 7)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
xticklabels=np.unique(y), yticklabels=np.unique(y)) 
plt.title('Confusion Matrix') 
plt.xlabel('Predicted Class') 
plt.ylabel('True Class') 
plt.show() 