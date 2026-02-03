

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


california = fetch_california_housing()

df = pd.DataFrame(california.data, columns=california.feature_names)
df["Preco_Casa"] = california.target


df = df[
    [
        "HouseAge",
        "Latitude",
        "Longitude",
        "AveBedrms",
        "AveRooms",
        "Preco_Casa"
    ]
]


print("Informações do Dataset:\n")
print(df.info())

print("\nEstatísticas Descritivas:\n")
print(df.describe())

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlação entre as Variáveis")
plt.show()


X = df.drop("Preco_Casa", axis=1)
y = df["Preco_Casa"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


modelo = LinearRegression()
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nAvaliação do Modelo:")
print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


importancia = pd.DataFrame({
    "Variavel": X.columns,
    "Coeficiente": modelo.coef_
}).sort_values(by="Coeficiente", ascending=False)

print("\nImportância das Variáveis:")
print(importancia)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Preço Real")
plt.ylabel("Preço Previsto")
plt.title("Preço Real vs Preço Previsto")
plt.show()
