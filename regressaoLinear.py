import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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
    sxy = np.sum(dados[:,1] * dados[:,0]) - dados.shape[0]*np.mean(dados[:,1])*np.mean(dados[:,0])
    sxx = np.sum(dados[:,0]**2) - dados.shape[0]*(np.mean(dados[:,0])**2)
    b1 = sxy / sxx
    b0 = np.mean(dados[:,1]) - (b1 * np.mean(dados[:,0]))
    print(b0)
    print(b1)
    yls = b0 + (b1 * dados[:,0])
    plt.plot(dados[:,0],yls)
    plt.scatter(dados[:,0],dados[:,1])
    plt.show()
    erro = mean_squared_error(dados[:,1],yls,squared=False)
    print(erro)
    print('oi')

if __name__ == '__main__':
    main()