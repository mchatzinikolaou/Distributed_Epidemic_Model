from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

class PopulationNode:


    def __init__(self, total_Population, name, beta=0.2, gamma=0.05):

        # Demographics
        self.Population = total_Population
        self.InitialPopulation=total_Population
        self.S = 1.0
        self.I = 0.0
        self.R = 0.0

        self.b = beta
        self.g = gamma
        self.name = name
        self.history = []
        self.PopulationHistory = []
        self.gotInfected=False


    def advanceByDays(self, days=1):


        if not self.gotInfected and self.I>0:
            self.gotInfected = True
        z = []
        S_old = self.S
        I_old = self.I
        R_old = self.R
        for day in range(0, days):
            # progress according to the model
            self.S += -self.b *(self.Population/self.InitialPopulation) * S_old * I_old
            self.I += self.b *(self.Population/self.InitialPopulation) * S_old * I_old - self.g * I_old
            self.R += self.g * I_old

            # self.S += -self.b * S_old * I_old
            # self.I += self.b  * S_old * I_old - self.g * I_old
            # self.R += self.g * I_old

            # update history.
            self.S = max(self.S, 0)
            self.I = max(self.I, 0)
            z.append([int(self.S * self.Population + 0.5), int(self.I * self.Population + 0.5),int(self.R * self.Population + 0.5)])

            # renew percentages for the next day.
            S_old = self.S
            I_old = self.I
            R_old = self.R

        self.history = self.history + z

        [self.S, self.I, self.R] = self.normalize([self.S, self.I, self.R])
        return z

    def plotNodeHistory(self):

        """
        Plot the whole history of the "nodeName" named Node.

        """

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

    def TravelTo(self, targetNode, groupSize):
        """
        :param targetNode: The node that will receive the travelers.
        :param groupSize: the total size of the travelling population
        """
        if groupSize >= self.Population:
            print("Not enough population in node " + str(self.name) + "!")
            return

        [self.S, self.I, self.R] = self.normalize([self.S, self.I, self.R])

        # Create passenger batch
        travelers = random.multinomial(n=groupSize, pvals=[self.S, self.I, self.R])

        # infect the other node.
        dS = travelers[0]
        dI = travelers[1]
        dR = travelers[2]

        # inform if there was an infection
        infected = targetNode.TravelFrom(dS, dI, dR)

        # remove passenger population from this node.
        # calculate using absolute values
        S_total = (np.multiply(self.S, self.Population) - dS)
        if S_total < 0:
            S_total = 0
        I_total = (np.multiply(self.I, self.Population) - dI)
        if I_total < 0:
            I_total = 0
        R_total = (np.multiply(self.R, self.Population) - dR)
        if R_total < 0:
            R_total = 0
        self.Population -= (dS + dI + dR)

        # update percentages.
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])
        return infected

    def TravelFrom(self, dS, dI, dR):
        """
        Receive the population , seperated into groups of Infected, Susceptible and Resolved(immune)
        people.
        :param dS: Susceptible incoming people.
        :param dI: Infected
        :param dR: Immune
        """

        # Add absolute values
        S_total = (np.multiply(self.S, self.Population) + dS)

        I_total = (np.multiply(self.I, self.Population) + dI)

        R_total = (np.multiply(self.R, self.Population) + dR)
        self.Population += (dS + dI + dR)

        # recalculate percentages
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])

        return self.gotInfected

    def TestInfect(self, dI=100):
        """
        "Inject" the population with dI infected individuals. This is used to initiate
        the epidemic.

        :param dI:
        """
        S_total = (np.multiply(self.S, self.Population)) + 0
        I_total = (np.multiply(self.I, self.Population)) + dI
        R_total = (np.multiply(self.R, self.Population)) + 0
        self.Population += dI

        # recalculate percentages
        [self.S, self.I, self.R] = self.normalize([S_total, I_total, R_total])

    def getName(self):
        return self.name

    def getHistory(self):
        return self.history

    # def getTruePopulations(self):
    #     return [round(np.multiply(self.S , self.Population)), round(np.multiply(self.I , self.Population)),
    #             round(np.multiply(self.R , self.Population))]

    def updatePopulation(self):
        self.PopulationHistory.append(self.Population)

    def normalize(self, IntList):

        normalizedSIR = np.array(IntList)
        size = len(normalizedSIR)
        normalizedSIR = np.divide(normalizedSIR, normalizedSIR.sum())
        normalizedSIR = np.maximum(normalizedSIR, np.zeros(size))
        return normalizedSIR

    def assertCorrectPopulations(self):
        print(self.I + self.S + self.R)
        assert self.I + self.S + self.R == 1, "Population quotients don't sum up to 1"




def addNormalNoise(history,sigma=1):
    S=[i[0] for i in history]
    I=[i[1] for i in history]
    R=[i[2] for i in history]
    [Sn,In,Rn]=[[],[],[]]

    for i in range(0,len(S)):
        Sn.append( np.add(S[i],np.random.normal(loc=0, scale=sigma*S[i])))
        In.append( np.add(I[i],np.random.normal(loc=0, scale=sigma*I[i])))
        Rn.append( np.add(R[i],np.random.normal(loc=0, scale=sigma*R[i])))
    return [Sn,In,Rn]


def addMeasurementNoise(history,l=1):
    S = [i[0] for i in history]
    I = [i[1] for i in history]
    R = [i[2] for i in history]
    Population=[sum(i) for i in history]
    [Sn, In, Rn] = [[], [], []]
    for i in range(0, len(S)):
        Inoise= np.subtract(I[i], np.random.exponential(scale=1/l * I[i]))
        #Rnoise=np.subtract(R[i], np.random.exponential(scale=1/l * R[i]))
        Rnoise=R[i]
        In.append(max(Inoise,0))
        Rn.append(max(Rnoise,0))
        Sn.append(max(np.subtract(Population[i],np.add(Rnoise,Inoise)),0))

    return [Sn, In, Rn]

def ExponentialNoiseTest(truebeta=0.2,max_lambda=200,min_lambda=1,points=80,MAX_REPEATS=50):
    testNode = PopulationNode(3875000, name="Athens", beta=truebeta)

    testNode.TestInfect(10)
    testNode.advanceByDays(1500)

    print("True beta: ", testNode.b)
    history = testNode.getHistory()

    # calculate betas with noise
    betas = []
    lambdas = np.linspace(min_lambda,max_lambda,points)
    for l in lambdas:
        iter_betas = []
        #for each sigma
        Ss=[]
        Is=[]
        Rs=[]
        for repeat in range(0,MAX_REPEATS):
            noiseHistory = addMeasurementNoise(history, l=l)

            S = noiseHistory[0][:]
            I = noiseHistory[1][:]
            R = noiseHistory[2][:]

            #take rolling averages

            M=60
            S = uniform_filter1d(S, size=M)
            I = uniform_filter1d(I, size=M)
            R = uniform_filter1d(R, size=M)

            Ss.append(S)
            Is.append(I)
            Rs.append(R)


            noiseHistory = [[S[i], I[i], R[i]] for i in range(0, len(S))]
            iter_betas.append(betaGammaFromEquations(noiseHistory)[0])
            if repeat%10 ==0 :
                print(l,repeat)


        Ss=np.mean(Ss,axis=0)
        Is=np.mean(Is,axis=0)
        Rs=np.mean(Rs,axis=0)


        iter_betas=np.sort(iter_betas)[int(MAX_REPEATS/3):2*int(MAX_REPEATS/3)]
        print("beta: ",np.median(iter_betas))
        betas.append(np.median(iter_betas))
        #plt.plot(range(0,len(iter_betas)),iter_betas)
        #plt.show()
        print("done lambda",l)

    error =np.abs( np.divide(np.subtract(betas, truebeta), truebeta))
    return [error, lambdas]


def RunNoiseTests():

    R0 = 1
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(sigmas, error)
    axs[0, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[0, 0].grid(axis='y', which='major')


    print("done ",R0)
    R0 = 1.2
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[0, 1].plot(sigmas, error)
    axs[0, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[0, 1].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 1.5
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[1, 0].plot(sigmas, error)
    axs[1, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[1, 0].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 2
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[1, 1].plot(sigmas, error)
    axs[1, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[1, 1].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 4
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[2, 0].plot(sigmas, error)
    axs[2, 0].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[2, 0].grid(axis='y', which='major')
    print("done ", R0)
    R0 = 8
    [error, sigmas] = ExponentialNoiseTest(truebeta=np.multiply(R0, 0.05))
    axs[2, 1].plot(sigmas, error)
    axs[2, 1].set_title('beta =' + str(R0 * 0.05) + ' , R0 = ' + str(R0))
    axs[2, 1].grid(axis='y', which='major')
    print("done ", R0)
    plt.suptitle("Estimation error vs standard deviation (30 days rolling average) ", fontsize=14)

    plt.show()


def betaGammaSingleNode(History):
    NodePopulation = sum(History[0])

    beta = 0
    gamma = 0

    betas = []
    gammas = []
    xToError={}
    theoretical_error={}
    for t in range(len(History) - 1):
        # MEDIAN INSTEAD
        if History[t + 1][1] - History[t][1] == 0 or History[t + 1][0] - History[t][0] == 0:
            break
        St = np.divide(History[t + 1][0], NodePopulation)
        St_1 = np.divide(History[t][0], NodePopulation)
        It = np.divide(History[t + 1][1], NodePopulation)
        It_1 = np.divide(History[t][1], NodePopulation)

        d_beta = - np.divide((St - St_1), (np.multiply(St_1, It_1)))
        d_gamma = 1 - np.divide(It, It_1) + np.multiply(d_beta, St_1)

        xToError[t]=(d_beta-0.2)/0.2
        beta = np.add(beta, d_beta)
        gamma = np.add(gamma, d_gamma)

        # REMOVE
        betas.append(d_beta)
        gammas.append(d_gamma)
        # REMOVE


    beta = np.median(betas)
    gamma = np.median(gammas)

    return [beta, gamma]


def SingleNodeExamples():
    Gamma = 0.3
    fig, axs = plt.subplots(2, 2)
    betas = []
    gammas = []

    for i in range(0, 4):
        R0 = i + 1
        Beta = Gamma * R0
        testNode = PopulationNode(3.5e06, name="Athens", beta=Beta, gamma=Gamma)
        testNode.TestInfect(10)
        testNode.advanceByDays(150)
        history = testNode.getHistory()
        x_axis = range(0, len(history))

        axs[int(i / 2), i % 2].plot(x_axis, [x[0] for x in history], x_axis, [x[1] for x in history], x_axis,
                                    [x[2] for x in history])
        axs[int(i / 2), i % 2].set_title("R0=" + str(R0),fontsize = 18.0)
        axs[int(i / 2), i % 2].set_xlabel('Day',fontsize = 18.0) if i>1 else None
        axs[int(i / 2), i % 2].set_ylabel('Population',fontsize = 18.0) if i%2!=1 else None
        axs[int(i / 2), i % 2].legend(['Susceptibles', 'Infectious', 'Removed'])

        parameters = betaGammaSingleNode(history)
        betas.append(parameters[0])
        gammas.append(parameters[1])


    plt.suptitle("Epidemic progress vs R0")
    plt.show()
    return betas, gammas

def SingleNodeExample():

    beta=0.4
    gamma=0.2
    days=500
    testNode = PopulationNode(1e06, name="Athens", beta=beta, gamma=gamma)
    testNode.TestInfect(10)
    testNode.advanceByDays(days)
    history = testNode.getHistory()

    plt.title("Epidemic progress for 500 days")
    x=range(days)
    plt.plot(x, [x[0] for x in history], x, [x[1] for x in history], x,
         [x[2] for x in history])

    plt.show()
    return history

# #history=SingleNodeExample()
# S=[i[0] for i in history]
# I=[i[1] for i in history]
# R=[i[2] for i in history]
#
# print(S)
# print(I)
# print(R)
