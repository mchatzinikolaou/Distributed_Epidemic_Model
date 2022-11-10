import math
from random import random
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls
from termcolor import colored
import Graph as gt
from Graph import PopulationNet as pn
import os
import csv
import plotly.graph_objects as go


def equalSplit(nmin, nmax, parts):
    limits = [int(np.sqrt(i)) for i in np.linspace(np.square(nmin), np.square(nmax + 1), num=parts + 1)]
    chunks = []
    for i in range(len(limits) - 1):
        chunks.append([limits[i], limits[i + 1] - 1])
    return chunks


def showFigure(N_vals, nmin, nmax, means, vals, title):
    fig = go.Figure(data=go.Scatter(
        x=sorted(N_vals)[nmin:nmax],
        y=means[nmin:nmax],
        error_y=dict(
            type='data',
            symmetric=False,
            array=np.subtract([max(i) for i in vals[:]], means),
            arrayminus=np.subtract(means, [min(i) for i in vals[:]]))
        , name="error values"

    ))
    fig.update_xaxes(title_text='number of nodes')
    fig.update_yaxes(title_text='error', tickangle=90)
    fig.add_trace(go.Scatter(
        x=sorted(N_vals)[nmin:nmax],
        y=means[nmin:nmax],
        mode="markers+lines",
        name="mean values"
    ))

    fig.update_layout(title_text=f"{title} relative error vs. number of nodes")
    fig.write_image(f"./images/fig{title}.png")
    fig.show()


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def generateGraph(betas, gammas, mean_deg=1, num_nodes=20,dt=1):
    InitialPopulations = 1e04 * np.ones(num_nodes)
    Populations = InitialPopulations

    network = pn()

    for i in range(0, num_nodes):
        network.addNode(Populations[i], str(i), betas[i], gammas[i],dt)
        for j in range(0, i):

            if random() < mean_deg / (num_nodes - 1):
                network.addEdge(str(i), str(j), 1)
                # if (random() < mean_deg / (num_nodes - 1)):
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


def getDeltasAndMatrix(Ss, Is, Rs, Populations, DAYS, N):
    _ = sum(i[0] for i in Populations)

    S = []
    R = []

    for i in range(len(Ss[0])):
        S.append(sum(j[i] for j in Ss))
        R.append(sum(j[i] for j in Rs))

    DeltaS = []
    DeltaR = []
    MeasurementMatrix = []
    IMatrix = []

    i = 1
    while i < DAYS-1:
        if S[i] == S[i - 1]:
            break
        DeltaS.append(S[i] - S[i - 1])
        DeltaR.append(R[i] - R[i - 1])
        innerList = []
        innerListR = []
        for k in range(0, N):
            innerList.append(-np.multiply(Ss[k][i - 1], Is[k][i - 1]))
            innerListR.append(Is[k][i - 1])
        MeasurementMatrix.append(innerList)
        IMatrix.append(innerListR)
        i += 1

    return MeasurementMatrix, DeltaS, IMatrix, DeltaR


def AddNoise(S, I, R, l=1):
    days = len(S[0])
    nodes = len(S)
    for node in range(nodes):
        for i in range(days):
            Inoisy = np.subtract(I[node][i], np.random.exponential(scale=1 / l * I[node][i]))
            I[node][i] = max(Inoisy, 0)

            S[node][i] = (max(np.subtract(1, np.add(R[node][i], I[node][i])), 0))

    return [S, I, R]


def sub_experiments(file_dir, degree=2, repeats=10, n_min=1, n_max=5):
    beta_file = open(f'./relative_results_with_sign_deg{degree}/beta_{file_dir}', 'a')
    gamma_file = open(f'./relative_results_with_sign_deg{degree}/gamma_{file_dir}', 'a')
    beta_writer = csv.writer(beta_file)
    gamma_writer = csv.writer(gamma_file)
    print(f'Starting process {os.getpid()} on chunk: {n_min}-{n_max}')

    NtoNnlsErrorBeta = {}
    NtoNnlsErrorGamma = {}
    for n in range(n_min, n_max + 1, 1):
        for repeat in range(0, repeats):
            DAYS = 2000
            truebetas, truegammas = betaGamas1(n)

            network = generateGraph(truebetas, truegammas, mean_deg=degree, num_nodes=n)
            network.testInfect()
            network.departures(DAYS)

            [_, Ss, Is, Rs, Populations] = getHistories(network, DAYS, n)
            [MeasurementMatrix, DeltaS, IMatrix, DeltaR] = getDeltasAndMatrix(Ss, Is, Rs, Populations, DAYS, n)
            MeasurementMatrix = np.array(MeasurementMatrix)
            IMatrix = np.array(IMatrix)
            DeltaS = np.array(DeltaS)
            DeltaR = np.array(DeltaR)

            nnls_betas = nnls(MeasurementMatrix, DeltaS, maxiter=100 * len(MeasurementMatrix))[0]
            nnls_gammas = nnls(IMatrix, DeltaR, maxiter=100 * len(IMatrix))[0]

            i = 0
            idx = []
            for node in network.getNodes():
                idx.append(i) if node.gotInfected else None
                i = i + 1

            b_idx = idx
            g_idx = idx

            b_c = len(b_idx)
            g_c = len(g_idx)

            nnls_betas = [nnls_betas[i] for i in b_idx]
            nnls_gammas = [nnls_gammas[i] for i in g_idx]
            truebetas = [truebetas[i] for i in b_idx]
            truegammas = [truegammas[i] for i in g_idx]

            BetaErrors = np.divide((np.subtract(nnls_betas, truebetas)), truebetas)
            GammaErrors = np.divide((np.subtract(nnls_gammas, truegammas)), truegammas)

            if b_c not in NtoNnlsErrorBeta.keys():
                NtoNnlsErrorBeta[b_c] = [np.median(BetaErrors)]
            else:
                NtoNnlsErrorBeta[b_c].append(np.median(BetaErrors))
            if g_c not in NtoNnlsErrorGamma.keys():
                NtoNnlsErrorGamma[g_c] = [np.median(GammaErrors)]
            else:
                NtoNnlsErrorGamma[g_c].append(np.median(GammaErrors))

            print(f'{n} nodes : repeat {repeat + 1}/{repeats} done')
    print(colored('finished!', 'red'), f'Process {os.getpid()}')
    for k in NtoNnlsErrorBeta.keys():
        beta_writer.writerow([k, NtoNnlsErrorBeta[k]])
    for k in NtoNnlsErrorGamma.keys():
        gamma_writer.writerow([k, NtoNnlsErrorGamma[k]])

    beta_file.close()
    gamma_file.close()


def noiseExperiment(file_dir, l=50, repeats=10, n_min=1, n_max=5):
    beta_file = open(f'./noise_val_{l}/beta_{file_dir}', 'a')
    gamma_file = open(f'./noise_val_{l}/gamma_{file_dir}', 'a')
    beta_writer = csv.writer(beta_file)
    gamma_writer = csv.writer(gamma_file)
    print(f'Starting process {os.getpid()} on chunk: {n_min}-{n_max}')

    NtoNnlsErrorBeta = {}
    NtoNnlsErrorGamma = {}
    for n in range(n_min, n_max + 1, 1):
        for repeat in range(0, repeats):
            DAYS = 2000
            truebetas, truegammas = betaGamas1(n)

            network = generateGraph(truebetas, truegammas, mean_deg=2, num_nodes=n)
            network.testInfect()
            network.departures(DAYS)

            [_, Ss, Is, Rs, Populations] = getHistories(network, DAYS, n)

            [Ss, Is, Rs] = AddNoise(Ss, Is, Rs, l)

            [MeasurementMatrix, DeltaS, IMatrix, DeltaR] = getDeltasAndMatrix(Ss, Is, Rs, Populations, DAYS, n)
            MeasurementMatrix = np.array(MeasurementMatrix)
            IMatrix = np.array(IMatrix)
            DeltaS = np.array(DeltaS)
            DeltaR = np.array(DeltaR)

            nnls_betas = nnls(MeasurementMatrix, DeltaS, maxiter=100 * len(MeasurementMatrix))[0]
            nnls_gammas = nnls(IMatrix, DeltaR, maxiter=100 * len(IMatrix))[0]

            i = 0
            idx = []
            for node in network.getNodes():
                idx.append(i) if node.gotInfected else None
                i = i + 1

            b_idx = idx
            g_idx = idx

            b_c = len(b_idx)
            g_c = len(g_idx)

            nnls_betas = [nnls_betas[i] for i in b_idx]
            nnls_gammas = [nnls_gammas[i] for i in g_idx]
            truebetas = [truebetas[i] for i in b_idx]
            truegammas = [truegammas[i] for i in g_idx]

            BetaErrors = np.divide((np.subtract(nnls_betas, truebetas)), truebetas)
            GammaErrors = np.divide((np.subtract(nnls_gammas, truegammas)), truegammas)

            if b_c not in NtoNnlsErrorBeta.keys():
                NtoNnlsErrorBeta[b_c] = [np.median(BetaErrors)]
            else:
                NtoNnlsErrorBeta[b_c].append(np.median(BetaErrors))
            if g_c not in NtoNnlsErrorGamma.keys():
                NtoNnlsErrorGamma[g_c] = [np.median(GammaErrors)]
            else:
                NtoNnlsErrorGamma[g_c].append(np.median(GammaErrors))

            print(f'{n} nodes : repeat {repeat + 1}/{repeats} done')
    print(colored('finished!', 'red'), f'Process {os.getpid()}')
    for k in NtoNnlsErrorBeta.keys():
        beta_writer.writerow([k, NtoNnlsErrorBeta[k]])
    for k in NtoNnlsErrorGamma.keys():
        gamma_writer.writerow([k, NtoNnlsErrorGamma[k]])

    beta_file.close()
    gamma_file.close()


def cases_error(file_dir, degree=2, repeats=10, n_min=1, n_max=5):
    max_file = open(f'./cases_error/max_{file_dir}', 'a')
    total_file = open(f'./cases_error/total_{file_dir}', 'a')
    max_writer = csv.writer(max_file)
    total_writer = csv.writer(total_file)
    print(f'Starting process {os.getpid()} on chunk: {n_min}-{n_max}')

    NtoMax = {}
    NtoTotal = {}
    for n in range(n_min, n_max + 1, 1):
        for repeat in range(0, repeats):
            DAYS = 2000
            truebetas, truegammas = betaGamas1(n)

            network = generateGraph(truebetas, truegammas, mean_deg=degree, num_nodes=n)
            network.testInfect()
            network.departures(DAYS)

            [History, Ss, Is, Rs, Populations] = getHistories(network, DAYS, n)

            [MeasurementMatrix, DeltaS, IMatrix, DeltaR] = getDeltasAndMatrix(Ss, Is, Rs, Populations, DAYS, n)
            MeasurementMatrix = np.array(MeasurementMatrix)
            IMatrix = np.array(IMatrix)
            DeltaS = np.array(DeltaS)
            DeltaR = np.array(DeltaR)

            nnls_betas = nnls(MeasurementMatrix, DeltaS, maxiter=100 * len(MeasurementMatrix))[0]
            nnls_gammas = nnls(IMatrix, DeltaR, maxiter=100 * len(IMatrix))[0]

            i = 0
            idx = []
            for node in network.getNodes():
                idx.append(i) if node.gotInfected else None
                i = i + 1

            b_idx = idx
            g_idx = idx

            b_c = len(b_idx)
            g_c = len(g_idx)
            # Re-run
            i = 0
            for node in network.getNodes():
                node.__init__(total_Population=node.Population, name=node.name, beta=nnls_betas[i],
                              gamma=nnls_gammas[i])
                i += 1

            network.testInfect()
            network.departures(DAYS)
            [_, newI, newR] = network.getTotalHistory()
            [_, oldI, oldR] = [History[0], History[1], History[2]]

            maxNew = max(newI)
            maxOld = max(oldI)

            TotalOld = oldR[-1]
            TotalNew = newR[-1]

            maxError = np.divide((maxNew - maxOld), maxOld)
            totalError = np.divide((TotalNew - TotalOld), TotalOld)

            if b_c not in NtoMax.keys():
                NtoMax[b_c] = [maxError]
            else:
                NtoMax[b_c].append(maxError)
            if g_c not in NtoTotal.keys():
                NtoTotal[g_c] = [totalError]
            else:
                NtoTotal[g_c].append(np.median(totalError))

            print(f'{n} /{n_max} : repeat {repeat + 1}/{repeats}')
    print(colored('finished!', 'red'), f'Process {os.getpid()}')
    for k in NtoMax.keys():
        max_writer.writerow([k, NtoMax[k]])
    for k in NtoTotal.keys():
        total_writer.writerow([k, NtoTotal[k]])

    max_file.close()
    total_file.close()


def beta_gamma_corr(file_dir, degree=2, repeats=10, n_min=1, n_max=5):
    file = open(f'./beta_gamma_corr/data{file_dir}', 'a')
    writer = csv.writer(file)
    print(f'Starting process {os.getpid()} on chunk: {n_min}-{n_max}')
    results = []
    for n in range(n_min, n_max + 1, 1):
        for repeat in range(0, repeats):
            DAYS = 2000
            truebetas, truegammas = betaGamas1(n)

            network = generateGraph(truebetas, truegammas, mean_deg=degree, num_nodes=n)
            network.testInfect()
            network.departures(DAYS)

            [_, Ss, Is, Rs, Populations] = getHistories(network, DAYS, n)

            [MeasurementMatrix, DeltaS, IMatrix, DeltaR] = getDeltasAndMatrix(Ss, Is, Rs, Populations, DAYS, n)
            MeasurementMatrix = np.array(MeasurementMatrix)
            IMatrix = np.array(IMatrix)
            DeltaS = np.array(DeltaS)
            DeltaR = np.array(DeltaR)

            # Get condition number
            _ = np.linalg.cond(MeasurementMatrix)
            # calculate betas
            nnls_betas = nnls(MeasurementMatrix, DeltaS, maxiter=100 * len(MeasurementMatrix))[0]
            nnls_gammas = nnls(IMatrix, DeltaR, maxiter=100 * len(IMatrix))[0]

            i = 0
            idx = []
            for node in network.getNodes():
                idx.append(i) if node.gotInfected else None
                i = i + 1

            b_idx = idx
            g_idx = idx

            _ = len(b_idx)
            _ = len(g_idx)

            nnls_betas = [nnls_betas[i] for i in b_idx]
            nnls_gammas = [nnls_gammas[i] for i in g_idx]
            truebetas = [truebetas[i] for i in b_idx]
            truegammas = [truegammas[i] for i in g_idx]

            BetaErrors = np.divide((np.subtract(nnls_betas, truebetas)), truebetas)
            GammaErrors = np.divide((np.subtract(nnls_gammas, truegammas)), truegammas)

            results.append([i, j] for i, j in zip(BetaErrors, GammaErrors))

            print(f'{n} nodes : repeat {repeat + 1}/{repeats}')

    print(f'Proc. {os.getpid()} saving results')
    writer.writerows(results)
    print(colored('finished!', 'red'), f'Process {os.getpid()}')

    file.close()


def runExperiment(func, Nmin=1, Nmax=50, repeats=15, proc_num=6, deg=1):
    if __name__ == "__main__":

        procs = []
        chunks = []
        limits = [int(np.sqrt(i)) for i in np.linspace(np.square(Nmin), np.square(Nmax + 1), num=proc_num + 1)]
        for i in range(len(limits) - 1):
            chunks.append([limits[i], limits[i + 1] - 1])

        for i in range(proc_num):
            procs.append(mp.Process(target=func, args=(f'results{i}.csv', deg, repeats, chunks[i][0], chunks[i][1])))
            procs[i].start()

        for i in range(proc_num):
            procs[i].join()


def runNoiseExperiment(func, Nmin=1, Nmax=50, repeats=15, proc_num=6, l=50.0):
    procs = []
    chunks = []
    limits = [int(np.sqrt(i)) for i in np.linspace(np.square(Nmin), np.square(Nmax + 1), num=proc_num + 1)]
    for i in range(len(limits) - 1):
        chunks.append([limits[i], limits[i + 1] - 1])

    for i in range(proc_num):
        procs.append(mp.Process(target=func, args=(f'results{i}.csv', l, repeats, chunks[i][0], chunks[i][1])))
        procs[i].start()

    for i in range(proc_num):
        procs[i].join()


def plotResults(nmin=0, nmax=-1, num=6, degree=1):
    NtoNnlsErrorBeta = {}
    NtoNnlsErrorGamma = {}
    for i in range(num):
        with open(f'./relative_results_with_sign_deg{degree}/beta_results{i}.csv', 'r') as betafile, open(
                f'./relative_results_with_sign_deg{degree}/gamma_results{i}.csv', 'r') as gammafile:
            beta_reader = csv.reader(betafile, delimiter=',', quotechar="\"")
            gamma_reader = csv.reader(gammafile, delimiter=',', quotechar="\"")
            for row in beta_reader:
                if row:
                    data = row[1].strip("\"[]\"").split(',')
                    if int(row[0]) not in NtoNnlsErrorBeta.keys():
                        NtoNnlsErrorBeta[int(row[0])] = [float(i) for i in data]
                    else:
                        for i in data:
                            NtoNnlsErrorBeta[int(row[0])].append(float(i))
            for row in gamma_reader:
                if row:
                    data = row[1].strip("\"[]\"").split(',')
                    if int(row[0]) not in NtoNnlsErrorGamma.keys():
                        NtoNnlsErrorGamma[int(row[0])] = [float(i) for i in data]
                    else:
                        for i in data:
                            NtoNnlsErrorGamma[int(row[0])].append(float(i))

    beta_N_vals = [float(i) for i in NtoNnlsErrorBeta.keys()]
    gamma_N_vals = [float(i) for i in NtoNnlsErrorGamma.keys()]
    beta_vals = [i for i in NtoNnlsErrorBeta.values()]
    gamma_vals = [i for i in NtoNnlsErrorGamma.values()]
    beta_means = [np.mean(i) for i in beta_vals]
    gamma_means = [np.mean(i) for i in gamma_vals]

    showFigure(beta_N_vals, nmin, nmax, beta_means, beta_vals, title="beta")
    showFigure(gamma_N_vals, nmin, nmax, gamma_means, gamma_vals, title="gamma")

def plotNoiseResults(nmin=0, nmax=-1, num=6, l=50):
    NtoNnlsErrorBeta = {}
    NtoNnlsErrorGamma = {}
    for i in range(num):
        with open(f'./noise_val_{l}/beta_results{i}.csv', 'r') as betafile, open(
                f'./noise_val_{l}/gamma_results{i}.csv', 'r') as gammafile:
            beta_reader = csv.reader(betafile, delimiter=',', quotechar="\"")
            gamma_reader = csv.reader(gammafile, delimiter=',', quotechar="\"")
            for row in beta_reader:
                if row:
                    data = row[1].strip("\"[]\"").split(',')
                    if int(row[0]) not in NtoNnlsErrorBeta.keys():
                        NtoNnlsErrorBeta[int(row[0])] = [float(i) for i in data]
                    else:
                        for i in data:
                            NtoNnlsErrorBeta[int(row[0])].append(float(i))
            for row in gamma_reader:
                if row:
                    data = row[1].strip("\"[]\"").split(',')
                    if int(row[0]) not in NtoNnlsErrorGamma.keys():
                        NtoNnlsErrorGamma[int(row[0])] = [float(i) for i in data]
                    else:
                        for i in data:
                            NtoNnlsErrorGamma[int(row[0])].append(float(i))

    beta_N_vals = [float(i) for i in NtoNnlsErrorBeta.keys()]
    gamma_N_vals = [float(i) for i in NtoNnlsErrorGamma.keys()]
    beta_vals = [i for i in NtoNnlsErrorBeta.values()]
    gamma_vals = [i for i in NtoNnlsErrorGamma.values()]



    beta_means = [np.mean(i) for i in beta_vals]
    gamma_means = [np.mean(i) for i in gamma_vals]

    showFigure(beta_N_vals, nmin, nmax, beta_means, beta_vals, title="beta")
    showFigure(gamma_N_vals, nmin, nmax, gamma_means, gamma_vals, title="gamma")

    # beta_medians = [np.median(i) for i in beta_vals]
    # gamma_medians = [np.median(i) for i in gamma_vals]
    #
    # showFigure(beta_N_vals, nmin, nmax, beta_medians, beta_vals, title="beta")
    # showFigure(gamma_N_vals, nmin, nmax, gamma_medians, gamma_vals, title="gamma")


def plotResults2(num=6):
    results = []
    for i in range(num):
        with open(f'./beta_gamma_corr/dataresults{i}.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',', quotechar="\"")
            for row in reader:
                if row:
                    for item in row:
                        result = item.strip("[]").split(',')
                        results.append([float(result[0]), float(result[1])])

    for result in results:
        print(result)

    x = [i[0] for i in results]
    y = [i[1] for i in results]

    plt.grid()
    plt.scatter(x, y)
    plt.title('Beta-gamma correlation')
    plt.xlabel('beta')
    plt.ylabel('gamma')
    plt.show()

def plotResultsCases(num=6, nmin=1, nmax=-1):
    MaxError = {}
    TotalError = {}
    for i in range(num):
        with open(f'./cases_error/max_results{i}.csv', 'r') as betafile, open(f'./cases_error/total_results{i}.csv',
                                                                              'r') as gammafile:
            beta_reader = csv.reader(betafile, delimiter=',', quotechar="\"")
            gamma_reader = csv.reader(gammafile, delimiter=',', quotechar="\"")
            for row in beta_reader:
                if row:
                    data = row[1].strip("\"[]\"").split(',')
                    if int(row[0]) not in MaxError.keys():
                        MaxError[int(row[0])] = [float(i) for i in data]
                    else:
                        for i in data:
                            MaxError[int(row[0])].append(float(i))
            for row in gamma_reader:
                if row:
                    data = row[1].strip("\"[]\"").split(',')
                    if int(row[0]) not in TotalError.keys():
                        TotalError[int(row[0])] = [float(i) for i in data]
                    else:
                        for i in data:
                            TotalError[int(row[0])].append(float(i))

    max_N_vals = [float(i) for i in MaxError.keys()]
    total_N_vals = [float(i) for i in TotalError.keys()]
    max_vals = [i for i in MaxError.values()]

    total_vals = [i for i in TotalError.values()]


    max_means = [np.mean(i) for i in max_vals]
    total_means = [np.mean(i) for i in total_vals]

    showFigure(max_N_vals, nmin, nmax, max_means, max_vals, title="Maximum simultaneous cases")
    showFigure(total_N_vals, nmin, nmax, total_means, total_vals, title="Total cases")

    # max_medians = [np.median(i) for i in max_vals]
    # total_medians = [np.median(i) for i in total_vals]
    #
    # showFigure(max_N_vals, nmin, nmax, max_medians, max_vals, title="Maximum simultaneous cases")
    # showFigure(total_N_vals, nmin, nmax, total_medians, total_vals, title="Total cases")

def singleNode(dt=1/2000):

    DAYS = 2000
    n=1
    degree=0
    truebetas, truegammas = betaGamas1(n)
    [truebetas, truegammas] = [[0.4],[0.2]]
    network = generateGraph(truebetas, truegammas, mean_deg=degree, num_nodes=n,dt=dt)
    network.testInfect()
    network.departures(DAYS)

    [_, Ss, Is, Rs, Populations] = getHistories(network, DAYS, n)
    [MeasurementMatrix, DeltaS, IMatrix, DeltaR] = getDeltasAndMatrix(Ss, Is, Rs, Populations, DAYS, n)
    MeasurementMatrix = np.array(MeasurementMatrix)
    IMatrix = np.array(IMatrix)
    DeltaS = np.array(DeltaS)
    DeltaR = np.array(DeltaR)

    nnls_betas = nnls(MeasurementMatrix, DeltaS, maxiter=100 * len(MeasurementMatrix))[0]
    nnls_gammas = nnls(IMatrix, DeltaR, maxiter=100 * len(IMatrix))[0]

    i = 0
    idx = []
    for node in network.getNodes():
        idx.append(i) if node.gotInfected else None
        i = i + 1

    b_idx = idx
    g_idx = idx

    b_c = len(b_idx)
    g_c = len(g_idx)

    nnls_betas = [nnls_betas[i] for i in b_idx]
    nnls_gammas = [nnls_gammas[i] for i in g_idx]
    truebetas = [truebetas[i] for i in b_idx]
    truegammas = [truegammas[i] for i in g_idx]

    BetaErrors = np.divide((np.subtract(nnls_betas, truebetas)), truebetas)
    GammaErrors = np.divide((np.subtract(nnls_gammas, truegammas)), truegammas)

    print(f"Beta error for dt=1/{1//dt}: {BetaErrors}")
    print(f"Gamma error dt=1/{1//dt}: {GammaErrors}")
    return [nnls_betas,nnls_gammas]

if __name__ == "__main__":
    for i in range(2001,0,-100):
        results=singleNode(dt=1/i)
        print(f'calculated {results[0]}{results[1]}')


else:
    l=math.inf
    numproc = mp.cpu_count()
    #runNoiseExperiment(func=noiseExperiment, Nmin=20, Nmax=60, repeats=5, proc_num=numproc, l=l)
    plotNoiseResults(num=numproc,l=l,nmax=60)
