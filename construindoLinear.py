import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def mse(y,yPred):
    return np.mean((y-yPred) ** 2)

def rmse(y,yPred):
    return np.sqrt(mse(y,yPred))

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

def main():
    dados = getData()
    mediaX = np.mean(dados[:,0])
    mediaY = np.mean(dados[:,1])
    sxy = np.sum(dados[:,0] * dados[:,1]) - dados.shape[0] * mediaX * mediaY
    sxx = np.sum(dados[:,0]**2) - dados.shape[0]*(mediaX**2)
    b1 = sxy / sxx
    b0 = mediaY - b1 * mediaX
    print("y = %fx + %f" % (b0,b1))
    yls = b0 + b1 * dados[:,0]
    plt.plot(dados[:,0],yls,c="red")
    plt.scatter(dados[:,0],dados[:,1])
    plt.show()

    print("MSE = %f" % (mse(dados[:,1],yls)))
    print("RMSE = %f" % (rmse(dados[:,1],yls)))

if __name__ == '__main__':
    main()