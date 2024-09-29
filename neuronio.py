class Neuronio:

    limiar = 0
    pesos = None
    taxaAprendizado = 0.1

    def __init__(self,pesos,limiar=0):
        self.pesos = pesos
        self.limiar = limiar

    def soma(self,entrada):
        resultSoma = 0
        for i in range(len(entrada)):
            resultSoma += entrada[i] * self.pesos[i]

        return resultSoma

    def ativacao(self,entrada):
        if (self.soma(entrada) > self.limiar):
            return 1
        else:
            return 0
        
    def delta(self,desejado,ativacao):
        return desejado - ativacao

    def treino(self,entradas,rotulos,limiteEpocas=1000):     
        epoca = 0   
        while(epoca < limiteEpocas):
            acertos = 0
            for idx, ent in enumerate(entradas):
                if (self.ativacao(ent) != rotulos[idx]):
                    delta = self.delta(rotulos[idx],self.ativacao(ent))
                    for j in range(len(self.pesos)):
                        self.pesos[j] = self.pesos[j] + self.taxaAprendizado * delta * ent[j]
                else:
                    acertos += 1
            print("Epoca %d - Erros %d - Pesos [%f,%f,%f]" % (epoca,len(rotulos) - acertos,self.pesos[0],self.pesos[1],self.pesos[2]))
            epoca += 1
            if (acertos == (len(rotulos))):
                break

if __name__ == '__main__':

    andNeuronio = Neuronio([0.1, -0.1, -0.1])
    dadosEntradas = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    rotulos = [0, 0, 0, 1]
    andNeuronio.treino(dadosEntradas, rotulos)

    orNeuronio = Neuronio([0.1, -0.1, -0.1])
    rotulos = [0, 1, 1, 1]
    orNeuronio.treino(dadosEntradas,rotulos)

    nandNeuronio = Neuronio([0.1, -0.1, -0.1])
    rotulos = [1, 1, 1, 0]
    nandNeuronio.treino(dadosEntradas, rotulos)

    for ent in dadosEntradas:
        xor = andNeuronio.ativacao([1,orNeuronio.ativacao(ent),nandNeuronio.ativacao(ent)])
        print("XOR(%d,%d) = %d" % (ent[1],ent[2],xor))
