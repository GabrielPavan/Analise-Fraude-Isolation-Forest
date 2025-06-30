import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- 1. Carregar o dataset ---
try:
    df = pd.read_csv("./dataset/creditcard.csv") 
except FileNotFoundError:
    print("Erro: Arquivo 'creditcard.csv' não encontrado. Verifique o caminho.")
    exit()

# --- 2. Preparação dos Dados ---
# Utilizando todas as features 'V' e a coluna 'Amount'.
colunas = [f'V{i}' for i in range(1, 29)] + ['Amount']
X = df[colunas]
y_true = df['Class']

# --- 3. Normalizar os Dados ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Definir Parâmetros dos Modelos ---
# Calculando a contaminação real do dataset.
fraude_ratio = y_true.value_counts(normalize=True)[1]
print(f"Proporção real de fraudes (contaminação): {fraude_ratio:.6f}\n")

# --- 5. Treinar e Prever com Isolation Forest ---
clf_if = IsolationForest(
    n_estimators=100, 
    contamination=fraude_ratio, 
    random_state=42,
    n_jobs=-1
)
y_pred_if_raw = clf_if.fit_predict(X_scaled)
y_pred_if = (y_pred_if_raw == -1).astype(int)

# --- 6. Treinar e Prever com Local Outlier Factor ---
clf_lof = LocalOutlierFactor(
    n_neighbors=200, 
    contamination=fraude_ratio,
    n_jobs=-1
)
y_pred_lof_raw = clf_lof.fit_predict(X_scaled)
y_pred_lof = (y_pred_lof_raw == -1).astype(int)

# --- 7. Avaliação dos Modelos ---
print("📊 Avaliação - Isolation Forest")
print(classification_report(y_true, y_pred_if, target_names=["Normal", "Fraude"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_true, y_pred_if))
# Cálculo da AUC-ROC para Isolation Forest
if_scores = clf_if.decision_function(X_scaled)
roc_auc_if = roc_auc_score(y_true, -if_scores)
print(f"AUC-ROC Score: {roc_auc_if:.4f}")


print("\n" + "="*40 + "\n")

print("📊 Avaliação - Local Outlier Factor")
print(classification_report(y_true, y_pred_lof, target_names=["Normal", "Fraude"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_true, y_pred_lof))
# Cálculo da AUC-ROC para Local Outlier Factor
lof_scores = clf_lof.negative_outlier_factor_
roc_auc_lof = roc_auc_score(y_true, -lof_scores)
print(f"AUC-ROC Score: {roc_auc_lof:.4f}")


# --- 8. Visualização dos Resultados ---
plt.figure(figsize=(14, 6))

# Gráfico para Isolation Forest
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred_if, cmap='coolwarm', s=4, alpha=0.6)
plt.title("Isolation Forest - Predições")
plt.xlabel("V1 (Normalizado)")
plt.ylabel("V2 (Normalizado)")

# Gráfico para Local Outlier Factor
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred_lof, cmap='coolwarm', s=4, alpha=0.6)
plt.title("Local Outlier Factor - Predições")
plt.xlabel("V1 (Normalizado)")

plt.suptitle("Visualização das Anomalias Detectadas (Projeção 2D)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("anomalias_resultado_otimizado.png", dpi=300)
print("\n✅ Gráfico otimizado salvo como anomalias_resultado_otimizado.png")
