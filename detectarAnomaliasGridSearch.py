import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score, roc_auc_score

# --- 1. Carregar o dataset ---
try:
    df = pd.read_csv("./dataset/creditcard.csv") 
except FileNotFoundError:
    print("Erro: Arquivo 'creditcard.csv' não encontrado. Verifique o caminho.")
    exit()

# --- 2. Preparação dos Dados ---
colunas = [f'V{i}' for i in range(1, 29)] + ['Amount']
X = df[colunas]
y_true = df['Class']

# --- 3. Normalizar as Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Otimização com GridSearch para Isolation Forest ---
print("Iniciando GridSearch para Isolation Forest... (Isso pode demorar vários minutos)")

# Definindo os parâmetros que queremos testar
param_grid_if = {
    'n_estimators': [100, 200, 300],
    'max_samples': ['auto', 0.8],
    'contamination': [y_true.value_counts(normalize=True)[1]]
}

# Função de pontuação customizada para corrigir o erro anterior
def f1_scorer_iforest(y_true, y_pred):
    y_pred_binary = (np.array(y_pred) == -1).astype(int)
    return f1_score(y_true, y_pred_binary, pos_label=1)

scorer = make_scorer(f1_scorer_iforest)

# Configurando o GridSearch
grid_search_if = GridSearchCV(
    estimator=IsolationForest(random_state=42, bootstrap=True),
    param_grid=param_grid_if,
    scoring=scorer,
    cv=3,
    n_jobs=-1
)

# Executando a busca
grid_search_if.fit(X_scaled, y_true)

# Exibindo os melhores parâmetros encontrados
print(f"Melhores parâmetros para Isolation Forest: {grid_search_if.best_params_}")
best_if = grid_search_if.best_estimator_

# --- 5. Avaliação do Modelo Otimizado (Isolation Forest) ---
y_pred_if_raw = best_if.predict(X_scaled)
y_pred_if = (y_pred_if_raw == -1).astype(int)

print("\n" + "="*40)
print("📊 Avaliação - Isolation Forest (Otimizado com GridSearch)")
print("="*40)
print(classification_report(y_true, y_pred_if, target_names=["Normal", "Fraude"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_true, y_pred_if))

# --- Cálculo da AUC-ROC para Isolation Forest ---
# O decision_function retorna scores onde valores mais baixos são mais anômalos.
# Invertemos o sinal para que valores mais altos indiquem maior probabilidade de ser anomalia.
if_scores = best_if.decision_function(X_scaled)
roc_auc_if = roc_auc_score(y_true, -if_scores)
print(f"AUC-ROC Score: {roc_auc_if:.4f}")


# --- 6. Avaliação do Local Outlier Factor (sem GridSearch) ---
print("\n" + "="*40)
print("📊 Avaliação - Local Outlier Factor (Implementação Padrão)")
print("="*40)

contamination_real = y_true.value_counts(normalize=True)[1]
clf_lof = LocalOutlierFactor(n_neighbors=200, contamination=contamination_real, n_jobs=-1)
y_pred_lof_raw = clf_lof.fit_predict(X_scaled)
y_pred_lof = (y_pred_lof_raw == -1).astype(int)

print(classification_report(y_true, y_pred_lof, target_names=["Normal", "Fraude"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_true, y_pred_lof))

# --- Cálculo da AUC-ROC para Local Outlier Factor ---
# O negative_outlier_factor_ também retorna scores onde valores mais baixos são mais anômalos.
# Invertemos o sinal para o cálculo da AUC.
lof_scores = clf_lof.negative_outlier_factor_
roc_auc_lof = roc_auc_score(y_true, -lof_scores)
print(f"AUC-ROC Score: {roc_auc_lof:.4f}")


# --- 7. Geração da Imagem com os Resultados Finais ---
print("\nGerando imagem com os resultados...")
plt.figure(figsize=(14, 6))

# Gráfico para Isolation Forest (Otimizado)
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred_if, cmap='coolwarm', s=4, alpha=0.6)
plt.title("Isolation Forest (com GridSearch)")
plt.xlabel("V1 (Normalizado)")
plt.ylabel("V2 (Normalizado)")

# Gráfico para Local Outlier Factor
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred_lof, cmap='coolwarm', s=4, alpha=0.6)
plt.title("Local Outlier Factor")
plt.xlabel("V1 (Normalizado)")

plt.suptitle("Visualização das Anomalias Detectadas (Resultado Final)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("anomalias_resultado_gridsearch.png", dpi=300)
print("\n✅ Gráfico final salvo como anomalias_resultado_gridsearch.png")
