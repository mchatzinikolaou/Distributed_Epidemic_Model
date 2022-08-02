import Graph as gt
from Graph import PopulationNet as pn
from random import random
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt

def generateGraph(betas, gammas, p, num_nodes=20):
    InitialPopulations = 1e04 * np.ones(num_nodes)
    Populations = InitialPopulations

    network = pn()

    for i in range(0, num_nodes):
        network.addNode(Populations[i], str(i), betas[i], gammas[i])
        for j in range(0, i):

            if (random() <= p):
                network.addEdge(str(i), str(j), 1)
            if (random() <= p):
                network.addEdge(str(j), str(i), 1)
    return network


def betaGamas1(num_nodes):
    parameters = gt.GenerateParameters(num_nodes)
    truebetas = [i[0] for i in parameters]
    truegammas = [i[1] for i in parameters]
    return [truebetas, truegammas]


def betaGamas2(b_min, b_max, g_min, g_max, num_nodes):
    parameters = gt.GenerateRandomParameters(b_min, b_max, g_min, g_max, num_nodes)
    truebetas = [i[0] for i in parameters]
    truegammas = [i[1] for i in parameters]
    return [truebetas, truegammas]


def getHistories(network, DAYS, N):
    NodeHistories = []
    for nodename in network.getNodeNames():
        NodeHistories.append(network.getNodeByName(nodename).getHistory())
    History = network.getTotalHistory()

    Ss = []
    Is = []
    Rs = []
    Populations = []
    for k in range(0, N):
        nodeHistory = NodeHistories[k]
        nodePopulationHistory = network.getNodeByName(str(k)).PopulationHistory
        newS = []
        newI = []
        newR = []
        for day in range(0, DAYS - 1):
            newS.append(np.divide(nodeHistory[day][0], nodePopulationHistory[day]))
            newI.append(np.divide(nodeHistory[day][1], nodePopulationHistory[day]))
            newR.append(np.divide(nodeHistory[day][2], nodePopulationHistory[day]))
        Ss.append(newS)
        Is.append(newI)
        Rs.append(newR)
        Populations.append(nodePopulationHistory)
    return [History, Ss, Is, Rs, Populations]


# def getDeltasAndMatrix(Ss,Is, Populations, DAYS, N):
#     DeltaS = []
#     MeasurementMatrix = []
#     for i in range(1, DAYS - 1):
#         DeltaS.append(S[i] - S[i - 1])
#         innerList = []
#         for k in range(0, N):
#             innerList.append((-np.multiply(Ss[k][i - 1], Populations[k][i - 1]) * Is[k][i - 1]))
#         MeasurementMatrix.append(innerList)
#
#     return MeasurementMatrix, DeltaS

def nonDim(Measurements,TotalPopulation):

    return list(map(lambda x: x/TotalPopulation, Measurements))




# def getDeltasAndMatrix(Ss,Is, Populations, DAYS, N):
#     TotalPopulation=sum(i[0] for i in Populations)
#
#     for i in range(len(Ss)):
#         for j in range(len(Ss[i])):
#             Ss[i][j]=Ss[i][j]/TotalPopulation
#             Is[i][j] = Is[i][j] / TotalPopulation
#
#     S=[]
#     for i in range(len(Ss[0])):
#         S.append(np.sum(j[i] for j in Ss))
#     DeltaS = []
#     MeasurementMatrix = []
#     for i in range(1, DAYS - 1):
#
#         DeltaS.append(S[i] - S[i - 1])
#         innerList = []
#         for k in range(0, N):
#             innerList.append(-np.multiply(np.multiply(Ss[k][i - 1] , Is[k][i - 1]),TotalPopulation))
#         MeasurementMatrix.append(innerList)
#
#     return MeasurementMatrix, DeltaS

def getDeltasAndMatrix(Ss,Is, Populations, DAYS, N):
    TotalPopulation=sum(i[0] for i in Populations)

    S=[]
    for i in range(len(Ss[0])):
        S.append(np.sum(j[i] for j in Ss))
    DeltaS = []
    MeasurementMatrix = []
    for i in range(1, DAYS - 1):

        DeltaS.append(S[i] - S[i - 1])
        innerList = []
        for k in range(0, N):
            innerList.append(-np.multiply(Ss[k][i - 1] , Is[k][i - 1]))
        MeasurementMatrix.append(innerList)

    return MeasurementMatrix, DeltaS





N = 50

condNumToNnlsError={}
condNumToPinvError={}
NtoNnlsError={}
NtoPinvError={}
repeats = 1
predicted_b = []
truebetas=[]
truegammas=[]
for repeat in range(0, repeats):
    for n in range(4,N,1):
        DAYS = 100*N
        truebetas, truegammas = betaGamas1(n)
        network = generateGraph(truebetas, truegammas, p=2/(n-1), num_nodes=n)

        network.testInfect()
        network.departures(DAYS)

        [History, Ss, Is, Rs, Populations] = getHistories(network, DAYS, n)

        S = History[0]
        I = History[1]
        R = History[2]

        [MeasurementMatrix, DeltaS] = getDeltasAndMatrix(Ss, Is, Populations, DAYS, n)

        MeasurementMatrix = np.array(MeasurementMatrix)
        DeltaS = np.array(DeltaS)
        nnls_betas = nnls(MeasurementMatrix, DeltaS, maxiter=10 * len(MeasurementMatrix))[0]
        Pinv_betas=np.matmul(np.linalg.pinv(MeasurementMatrix),DeltaS)
        cond_number=np.linalg.cond(MeasurementMatrix)


        nnls_errors = np.divide(abs(np.subtract(nnls_betas, truebetas)), truebetas)


        condNumToNnlsError[cond_number]=[np.mean(nnls_errors)]

        NtoNnlsError[n]=[np.mean(nnls_errors)]

        predicted_b.append(list(nnls_betas))


# x_vals= [float(i) for i in condNumToNnlsError.keys()]
# y_nnls_vals=[float(i[0]) for i in condNumToNnlsError.values()]
# y_Pinv_vals=[float(i[0]) for i in condNumToPinvError.values()]

x_vals= [float(i) for i in NtoNnlsError.keys()]
y_nnls_vals=[float(i[0]) for i in NtoNnlsError.values()]
# y_Pinv_vals=[float(i[0]) for i in NtoPinvError.values()]


fig, axs = plt.subplots(2)
fig.suptitle(f" Nodes:{4} to {N}, {repeats} repetitions")

#axs[0].scatter(x_vals,y_nnls_vals)
#axs[1].scatter(x_vals,y_Pinv_vals)

axs[0].scatter(x_vals,y_nnls_vals)
# axs[1].scatter(x_vals,y_Pinv_vals)


#plt.xlabel('condition number of the Measurement Matrix')
plt.xlabel('Number of nodes')
plt.ylabel('Estimation error')

axs[0].set(ylabel='Estimation error for nnls')
axs[1].set(ylabel='Estimation error for pseudoinverse')

axs[0].grid()
axs[1].grid()
#
# axs[0].set_xlim([0,1e04])
# axs[1].set_xlim([0,1e04])

plt.show()


mean_b = np.mean(predicted_b, axis=0)

x = np.linspace(0.1, 0.5, 1000)

plt.scatter(truebetas, mean_b)
plt.title("Scatter plot of real beta values and estimated beta values (error=" + str(
    np.mean(np.subtract(truebetas, mean_b))) + ")")
plt.ylabel("Predicted beta(mean)")
plt.xlabel("True beta")
plt.grid()
plt.plot(x, x, '-r')
plt.show()

median_b = np.median(predicted_b, axis=0)

plt.scatter(truebetas, median_b)
plt.title("Scatter plot of real beta values and estimated beta values (error = " + str(
np.mean(np.subtract(truebetas, median_b))) + ")")
plt.ylabel("Predicted beta(median)")
plt.xlabel("True beta")
plt.grid()
plt.plot(x, x, '-r')
plt.show()

print("Mean error  : ", np.mean(np.subtract(truebetas, median_b)))
