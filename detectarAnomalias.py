import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregar o dataset
df = pd.read_csv("./dataset/creditcard.csv") 

# 2. Selecionar colunas numéricas (features) e remover valores nulos
colunas = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']
X = df[colunas].dropna()
y_true = df.loc[X.index, 'Class']  # Classe real: 1 = fraude, 0 = normal

# 3. Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Isolation Forest
clf_if = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
pred_if = clf_if.fit_predict(X_scaled)  # -1 = anomalia, 1 = normal
y_pred_if = (pred_if == -1).astype(int)

# 5. Local Outlier Factor
clf_lof = LocalOutlierFactor(n_neighbors=50, contamination=0.01)
pred_lof = clf_lof.fit_predict(X_scaled)  # -1 = anomalia, 1 = normal
y_pred_lof = (pred_lof == -1).astype(int)

# 6. Avaliação dos modelos
print("📊 Avaliação - Isolation Forest")
print(classification_report(y_true, y_pred_if, target_names=["Normal", "Fraude"]))
print(confusion_matrix(y_true, y_pred_if))

print("\n📊 Avaliação - Local Outlier Factor")
print(classification_report(y_true, y_pred_lof, target_names=["Normal", "Fraude"]))
print(confusion_matrix(y_true, y_pred_lof))

# 7. Visualização dos resultados (usando as duas primeiras features)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred_if, cmap='coolwarm', s=2)
plt.title("Isolation Forest")

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred_lof, cmap='coolwarm', s=2)
plt.title("Local Outlier Factor")

plt.tight_layout()
plt.savefig("anomalias_resultado.png", dpi=300)
print("\n✅ Gráfico salvo como anomalias_resultado.png")