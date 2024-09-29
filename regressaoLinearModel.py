import numpy as np

def getData():
    dataFinal=[
        [19,900],
        [20,920],
        [22,950],
        [25,932],
        [26,935],
        [28,950],
        [29,1000],
        [30,1250],
        [39,2500],
        [40,2750],
        [42,2800],
        [55,3200],
        [60,4000]
    ]
    return np.array(dataFinal)

# Função para calcular o RMSE
def compute_rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))

# Função para calcular as previsões baseadas nos parâmetros (theta_0 e theta_1)
def predict(X, theta_0, theta_1):
    return theta_0 + theta_1 * X

# Função de gradiente descendente
def gradient_descent(X, y, theta_0, theta_1, learning_rate, n_iterations):
    m = len(y)  # Número de exemplos
    for i in range(n_iterations):
        y_pred = predict(X, theta_0, theta_1)  # Previsões com os parâmetros atuais
        
        # Gradientes parciais em relação a theta_0 e theta_1
        d_theta_0 = -2 * np.sum(y - y_pred) / m
        d_theta_1 = -2 * np.sum((y - y_pred) * X) / m
        
        # Atualização dos parâmetros
        theta_0 -= learning_rate * d_theta_0
        theta_1 -= learning_rate * d_theta_1
        
        # Calcular e imprimir o RMSE a cada 100 iterações
        if i % 10 == 0:
            rmse = compute_rmse(y, y_pred)
            print(f"Iteração {i}: RMSE = {rmse:.4f}, theta_0 = {theta_0:.4f}, theta_1 = {theta_1:.4f}")
    
    return theta_0, theta_1

# Dados simulados (X e y)
X = getData()[:,0]
y = getData()[:,1]

# Parâmetros iniciais (theta_0 e theta_1)
theta_0_init = 0
theta_1_init = 0

# Hiperparâmetros
learning_rate = 0.1
n_iterations = 1000

# Executar o gradiente descendente
theta_0_final, theta_1_final = gradient_descent(X, y, theta_0_init, theta_1_init, learning_rate, n_iterations)

print(f"Parâmetros finais: theta_0 = {theta_0_final:.4f}, theta_1 = {theta_1_final:.4f}")
