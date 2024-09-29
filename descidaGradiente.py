import numpy as np, matplotlib.pyplot as plt

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

def mse(y,yPred):
    return np.mean((y-yPred) ** 2)

def rmse(y,yPred):
    return np.sqrt(mse(y,yPred))

def predicao(x, b0, b1):
    return b1 + b0 * x

def main():
    X = getData()[:,0]
    y = getData()[:,1]

    b0 = 0
    b1 = 0

    lr = 0.0005
    nIteracoes = 1000

    for i in range(nIteracoes):
        yPred = predicao(X,b0,b1)

        b0grad = -2 * np.sum((y-yPred) * X) / y.shape[0]
        b1grad = -2 * np.sum(y-yPred) / y.shape[0]

        b0 = b0 - lr * b0grad
        b1 = b1 - lr * b1grad
        if (i % 5 == 0):
            rmseCalc = rmse(y,yPred)
            print("Iteração %d RMSE %f b0 %f b1 %f" % (i,rmseCalc,b0,b1))
    plt.plot(X,yPred,c="red")
    plt.scatter(X,y)
    plt.show()


if __name__ == '__main__':
    main()