
import numpy as np
import matplotlib.pyplot as plt


class PopulationNode:

    def __init__(self, nodePopulation, GraphPopulation, name, beta=0.2, gamma=0.05):

        self.Population = nodePopulation
        self.GraphPopulation = GraphPopulation
        self.S = np.divide(nodePopulation, GraphPopulation)
        self.I = 0.0
        self.R = 0.0

        self.b = beta
        self.g = gamma
        self.name = name
        self.history = []
        self.PopulationHistory = []

    def advanceByDays(self, bias_S=0, bias_I=0, bias_R=0, days=1):

        z = []
        S_old = self.S
        I_old = self.I
        R_old = self.R
        for day in range(0, days):

            A=np.multiply(np.multiply(self.b, S_old) , np.multiply(I_old , self.GraphPopulation))
            B=np.multiply(self.g,I_old)

            self.S += - A + np.divide(bias_S,self.GraphPopulation)
            self.I += A - B + np.divide(bias_I,self.GraphPopulation)
            self.R += B+ np.divide(bias_R,self.GraphPopulation)

            S_old = self.S
            I_old = self.I
            R_old = self.R

            z.append([S_old,I_old,R_old])

        self.history = self.history + z

        return z

    def TestInfect(self, dI=1e-4):
        self.S=self.S-dI
        self.I=dI

    def getName(self):
        return self.name

    def plotNodeHistory(self):

        history = self.getHistory()
        t = range(0, np.size(history, 0))

        S = []
        I = []
        R = []
        for i in range(0, len(history)):
            S.append((history[i])[0])
            I.append((history[i])[1])
            R.append((history[i])[2])

        plt.plot(t, S, 'r-', t, I, 'g-', t, R, 'b-')

        plt.show()


    def getHistory(self):
        return self.history

    def updatePopulation(self):
        self.PopulationHistory.append(self.Population)


def Test0():
    node = PopulationNode(nodePopulation=1e06, GraphPopulation=1e06, name="nm", beta=0.2, gamma=0.05)
    node.TestInfect()
    node.advanceByDays(days=100)
    node.plotNodeHistory()

Test0()