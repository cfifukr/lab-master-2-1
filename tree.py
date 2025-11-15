import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('diabetes_prediction_dataset.csv')

df.head()


categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = pd.Categorical(df[col]).codes

numeric_features = df.columns.drop('diabetes')  

df.info()


print("\nКількість пропусків у кожній колонці:")
print(df.isnull().sum())

df.dropna(inplace=True)

print("\nКількість пропусків після видалення:")
print(df.isnull().sum())
print("\nРозмір датасету після обробки пропусків:", df.shape)


df_diabetes = df[df['diabetes'] == 1]
df_no_diabetes = df[df['diabetes'] == 0]

min_size = min(len(df_diabetes), len(df_no_diabetes))

df_diabetes_sampled = df_diabetes.sample(n=min_size, random_state=42)
df_no_diabetes_sampled = df_no_diabetes.sample(n=min_size, random_state=42)

df_balanced = pd.concat([df_diabetes_sampled, df_no_diabetes_sampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_balanced.drop('diabetes', axis=1)
y = df_balanced['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Розподіл класів у тренувальному наборі:")
print(y_train.value_counts(normalize=True))
print("\nРозподіл класів у тестовому наборі:")
print(y_test.value_counts(normalize=True))

scaler = StandardScaler()
scaler.fit(X_train)

scaled_X_train = scaler.transform(X_train) 
scaled_X_test = scaler.transform(X_test) 

model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=numeric_features, 
          class_names=['0', '1'], filled=True,
          max_depth=5)  
plt.show()

