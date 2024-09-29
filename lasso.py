import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dados simulados
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 amostras, 5 características
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)  # Função linear com ruído

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicando Lasso (Regularização L1)
lasso_model = Lasso(alpha=0.1)  # alpha é o parâmetro lambda
lasso_model.fit(X_train, y_train)

# Prever nos dados de teste
y_pred = lasso_model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse}")
print("Coeficientes do modelo Lasso:", lasso_model.coef_)