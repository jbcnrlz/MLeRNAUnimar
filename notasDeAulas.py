import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
def gauss(x,mu,sigma):
    ptermo = (2*np.pi*sigma)**.5 # 1 / sqrt(2 pi sigma)
    stermo = np.exp(-(x - mu)**2 / (2*sigma)) #exp
    return  stermo / ptermo

def main():
    notas = pd.read_csv('dados/notas.csv') #carrega arquivo csv
    notas = np.array(notas).flatten() #transforma em matriz e achata ela transformando em um vetor
    notas = np.array([n for n in notas if n >= 0.5])
    
    gmm = GaussianMixture(n_components=3,tol=0.00001) #crio objeto do GMM com 3 Gaussianas
    gmm.fit(np.expand_dims(notas,1)) #Treino o modelo

    grupos = gmm.predict(np.expand_dims(notas,1)).flatten()
    cores = ['red','green','blue','purple','yellow']
    ggmm = np.exp([gmm.score_samples(xline.reshape(-1,1)) for xline in notas]).flatten()

    #plt.scatter(notas,grupos + 1)

    md = np.mean(notas)
    std = np.std(notas)

    gx = np.array([gauss(vNota,md,std**2) for vNota in notas])

    y, x = np.histogram(notas) #Gerando histograma dos meus dados, em Y estão as quantidades e em X os extratos
    x = x[:-1] #retirando o último extrato

    plt.bar(x,y / np.sum(y)) #gerando gráfico de barras
    #plt.scatter(notas,gx) #gerando a gaussiana do conjunto de notas
    plt.scatter(notas,ggmm,c=np.array(cores)[grupos]) #distribuição a partir do modelo de mistura
    plt.show() #mostrando na tela

if __name__ == '__main__':
    main()