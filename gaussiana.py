import pandas as pd, numpy as np, matplotlib.pyplot as plt, math
import scipy.stats
from sklearn.mixture import GaussianMixture

def gauss(mu,sigma,x):
    denomi = (2*math.pi*sigma)**.5
    num = math.exp(-(float(x)-float(mu))**2/(2*sigma))
    return num/denomi

def main():
    nts = pd.read_csv('dados/notas.csv')
    nts = np.array(nts).flatten()

    gmm = GaussianMixture(n_components=3,tol=0.000001)
    gmm.fit(np.expand_dims(nts,1))

    gmmgs = np.exp([gmm.score_samples(xline.reshape(-1,1)) for xline in nts]).flatten()

    y, x = np.histogram(nts)

    med = np.mean(nts)
    std = np.std(nts)
    plt.bar(x[:-1],y/np.max(y))
    gx = np.array([gauss(med,std**2,xline) for xline in nts])
    norm = scipy.stats.norm(med,std)
    gxs = np.array([norm.pdf(xline) for xline in nts])
    for mu, sd, p in zip(gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_):        
        g_s = scipy.stats.norm(mu, sd).pdf(nts) * p
        plt.scatter(nts, g_s)
    
    
    #plt.scatter(nts,gx)
    #plt.scatter(nts,gxs,c='red')
    plt.scatter(nts,gmmgs,c='blue')    
    plt.show()


if __name__ == '__main__':
    main()