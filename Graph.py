import networkx as nx
import matplotlib.pyplot as plt
import Node
import numpy as np
import random


class PopulationNet(nx.DiGraph):
    """
    The network that models the communication between the nodes.
    """

    def __init__(self):
        super().__init__()
        self.infectionEvents = []

    def addNode(self, Population, Name, beta=0.5, gamma=0.15,dt=1):
        """
        Adds a new node to the network.
        :param Population: The population of the node
        :param Name: The name of the node.
        """
        if beta is None and gamma is None:
            NewPopNode = Node.PopulationNode(Population, Name,dt=1)
        else:
            NewPopNode = Node.PopulationNode(Population, Name, beta, gamma,dt=1)

        self.add_node(NewPopNode)
        return NewPopNode

    def addEdge(self, NodeName1, NodeName2, Weight=1):
        """
        Adds a new edge to the graph. The edge initially represents the traffic between the two nodes.
        It could be chosen to represent other things, such as profit/trade flow.
        :param NodeName1: Departure node
        :param NodeName2: Arrival node
        :param Weight: The traffic (this can be expanded)
        :return:
        """
        if NodeName1 == NodeName2:
            print("Can't create self-loops. (", NodeName1, " to ", NodeName2, ")")
            return
        Node1 = self.getNodeByName(NodeName1)
        Node2 = self.getNodeByName(NodeName2)
        self.add_edge(Node1, Node2, weight=Weight)  # vale endexomenws type integer stin synartisi.

    def draw(self):
        """
        Draws the graph
        """

        lablist = {}
        for node in self.getNodes():
            lablist[node] = node.name
        # rename labels for showing purposes
        H = nx.relabel_nodes(self, lablist)
        nx.draw_networkx(H, pos=nx.circular_layout(H))
        plt.show()

    def getNodes(self):
        return self.nodes

    def getNodeNames(self):
        nodelist = []
        for node in self.nodes:
            nodelist.append(node.getName())
        return nodelist

    def getEdges(self):
        return self.edges

    def getNodeDegree(self, name):
        return self.degree[self.getNodeIndexByName(name)]

    def getNodeIndexByName(self, name):
        i = 0
        for node in self.getNodes():
            if node.getName() == name:
                return i
            i = i + 1

    def getNodeByName(self, name):
        """
        Returns the node object using it's name.
        (The object is the entity itself, as opposed to a copy of it. This allows us to
        alter its variables such as the populations etc.)

        :param name: The searched name.
        :return: The node object.
        """
        for node in self.getNodes():
            if node.getName() == name:
                return node

    def testInfect(self,n=100):
        list(self.nodes)[0].TestInfect(n)

    def advance(self, days=1):
        nodes = self.getNodes()
        for node in nodes:
            node.advanceByDays(days)

    def isConnected(self):
        return nx.is_connected(self)

    def FindNeighbours(self, name):
        """
        Returns all neighbours of the named node.
        This can be used to access the traffic between the nodes , alter it ,
        add attributes (more "weight variables") etc.

        :param name: The name of the node in question
        :return: neighbourList = All the neighbours
        """
        neighbourList = []
        for edge in self.getEdges():
            if edge[0].name == name:
                neighbourList.append(edge[1].getName())
        return neighbourList

    def isEmpty(self):
        return self.number_of_nodes() == 0

    def departures(self, days=1):
        for day in range(1, days):

            for node in self.nodes:
                node.updatePopulation()
                node.advanceByDays(1)

            for edge in self.edges.data():
                newInfection = edge[0].TravelTo(edge[1], edge[2]['weight'])
                if newInfection:
                    # print("Node ", edge[0].name, "infected node ", edge[1].name," on day ",day)
                    self.infectionEvents.append([day, edge[0], edge[1]])
            # Print (newly) infected nodes



    def plotNodeHistory(self, nodeName):

        """
        Plot the whole history of the "nodeName" named Node.

        :param nodeName: the name of the node

        """

        history = self.getNodeByName(nodeName).getHistory()

        plt.figure(nodeName)
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

    def getTotalHistory(self):
        """
        Plot the history of the whole system.

        This requires fetching the history of each node and summing their absolute values
        """
        history = []
        for node in self.nodes:
            if len(history) != 0:
                history = np.add(history, node.getHistory())
            else:
                history = node.getHistory()

        S = []
        I = []
        R = []

        for i in range(0, len(history)):
            S.append((history[i])[0])
            I.append((history[i])[1])
            R.append((history[i])[2])

        return [S, I, R]

    def plotTotalHistory(self):
        [S, I, R] = self.getTotalHistory()
        t = range(0, np.size(S, 0))
        name = "nodes_" + str(self.number_of_nodes()) + "_edges_" + str(self.number_of_edges()) + ".png"
        print(name)
        plt.figure()
        plt.plot(t, S, 'r-', t, I, 'g-', t, R, 'b-')
        plt.legend(("Susceptible", "Infected", "Removed"))
        # fig.savefig(name)
        plt.show()


def GenerateNodes(TotalPopulation, num_nodes, node_parameters, PopulationDistributionType="uniform"):
    nodes = []
    if PopulationDistributionType == "uniform":
        for i in range(0, num_nodes):
            nodes.append(
                Node.PopulationNode(TotalPopulation / num_nodes, str(i), node_parameters[i][0], node_parameters[i][1]))
    return nodes


def CreateRandomNetwork(TotalPopulation, p, numberOfNodes=1):
    if p > 1 or p < 0:
        print("Invalid probability value")
        return

    newnet = PopulationNet()
    for i in range(0, numberOfNodes):
        newnet.addNode(int(TotalPopulation / numberOfNodes), str(i))
    for node1 in newnet.getNodeNames():
        for node2 in newnet.getNodeNames():
            if node1 != node2 and random.random() <= p:
                newnet.addEdge(node1, node2, 50)

    return newnet


def CustomNet_manous():
    net = PopulationNet()
    Ss=[90,200,300,400,500]
    i=0
    for S in Ss:
        net.addNode(S,i)
        i+=1
    for node1 in net.getNodeNames():
        for node2 in net.getNodeNames():
            if node1 != node2:
                net.addEdge(node1, node2, 10)

    net.testInfect(10)
    net.departures(10)

    for node in net.getNodes():
        print("Node: ",str(node.name))
        hist=node.getHistory()

        S=[i[0] for i in hist]
        I=[i[1] for i in hist]
        R=[i[2] for i in hist]
        print(f"S: {S}")
        print(f"I: {I}")
        print(f"R: {R}")
        print()

def plotHistory(History):
    S = []
    I = []
    R = []
    for i in History:
        S.append(i[0])
        I.append(i[1])
        R.append(i[2])

    t = range(0, np.size(S, 0))
    plt.figure()
    plt.plot(t, S, 'r-', t, I, 'g-', t, R, 'b-')
    plt.legend(("Susceptible", "Infected", "Removed"))
    # fig.savefig(name)
    plt.show()


def runSimulation(totalNodes=50, totalPopulation=1e08, mean_degree=1.0, days=500):
    if mean_degree < 1:
        print("sub critical region")
    elif mean_degree == 1:
        print("critical region")
    elif mean_degree <= np.log(totalNodes):
        print("supercritical region")
    else:
        print("Connected region")
    if totalNodes > 1:
        net = CreateRandomNetwork(totalPopulation, p=mean_degree * (2 / (totalNodes - 1)), numberOfNodes=totalNodes)
        net.testInfect()
        net.departures(days)
        return net.getTotalHistory(), net.infectionEvents  # if NaN events, can't return


def runAndPlot(nodes=100, TotalPopulation=1e07, p=1.2, days=500, N=5, show_lines=False):
    # Run n simulations
    [History, results] = runSimulation(nodes, TotalPopulation, p, days)
    for i in range(1, N):
        [newHistory, new_results] = runSimulation(nodes, TotalPopulation, p, days)
        History[0] = np.add(History[0], newHistory[0])
        History[1] = np.add(History[1], newHistory[1])
        History[2] = np.add(History[2], newHistory[2])

    S = np.divide(History[0], N)
    I = np.divide(History[1], N)
    R = np.divide(History[2], N)

    t = range(0, np.size(S, 0))

    days = list(np.array(results)[:, 0])

    plt.figure()
    plt.plot(t, S, 'g-', t, I, 'r-', t, R, 'b-')
    plt.title('Progress of the epidemic for ' + str(nodes) + ' nodes (mean degree=' + str(p) + ")")
    plt.ylabel('Population')
    plt.xlabel('Day')
    # Visualization of infection events
    if show_lines:
        i = 0
        while i < len(days):
            prev_day = days[i]
            length = 1
            j = 0
            while i + j < len(days) and days[i + j] == prev_day:
                length = length + 1
                j = j + 1
            plt.vlines(days[i], 0, (length - 1) * TotalPopulation / nodes)
            i = i + j

    plt.legend(("Susceptible", "Infected", "Removed"))
    # fig.savefig(name)
    plt.show()


def GenerateRandomParameters(b_min, b_max, g_min, g_max, num_nodes):
    return np.random.uniform([b_min, g_min], [b_max, g_max], (num_nodes, 2))


def GenerateParameters(num_nodes=100):
    b_range = [0.1, 0.2]
    g_range = [0.01, 0.05]
    return GenerateRandomParameters(b_range[0], b_range[1], g_range[0], g_range[1], num_nodes)


def runRandomNode(days=1000, population=1e07):
    nodeParameters = GenerateParameters(1)
    nodes = GenerateNodes(population, 1, nodeParameters)[0]
    nodes.TestInfect()
    history = nodes.advanceByDays(days)
    return history, nodes.b, nodes.g


# def discreteDerivative(Points):
#     for i in range(len(Points) - 1):
#         Points[i][:] = np.subtract(Points[i + 1][:], Points[i][:])
#     return Points[0:-1][:]


# def betaGammaFromEquations(History):
#     Total_Population = sum(History[0])
#     print(Total_Population)
#     beta = 0
#     gamma = 0
#
#     betas = []
#     gammas = []
#
#     for t in range(len(History) - 1):
#         # MEDIAN INSTEAD
#         if History[t + 1][1] - History[t][1] == 0 or History[t + 1][0] - History[t][0] == 0:
#             break
#         St = np.divide(History[t + 1][0], Total_Population)
#         St_1 = np.divide(History[t][0], Total_Population)
#         It = np.divide(History[t + 1][1], Total_Population)
#         It_1 = np.divide(History[t][1], Total_Population)
#
#         d_beta = - np.divide((St - St_1), (np.multiply(St_1, It_1)))
#
#         d_gamma = 1 - np.divide(It, It_1) + np.multiply(d_beta, St_1)
#         # print(St - St_1,It-It_1)
#         beta = np.add(beta, d_beta)
#         gamma = np.add(gamma, d_gamma)
#
#         # REMOVE
#         betas.append(d_beta)
#         gammas.append(d_gamma)
#         # REMOVE
#
#     beta = np.median(betas)
#     gamma = np.median(gammas)
#
#     # REMOVE
#     plt.plot(range(0, len(betas)), betas, gammas)
#     plt.show()
#     # REMOVE
#     return [beta, gamma]


#[history,events]=runSimulation(totalNodes=50, totalPopulation=1e08, mean_degree=1.0, days=500)

#print(history[0][0]+history[1][0]+history[2][0])
def run1():

    N=3
    days=200
    net = CreateRandomNetwork(10e06, p=1 * (2 / (N - 1)), numberOfNodes=N)
    net.testInfect()
    net.departures(days)



    Ss=[]
    Is=[]
    Rs=[]


    for node in net.getNodes():
        history=node.getHistory()
        Ss.append([j[0] for j in history])
        Is.append([j[1] for j in history])
        Rs.append([j[2] for j in history])


    t=range(days-1)


    fig, axs = plt.subplots(2, 2)
    axs[0][0].plot(t, Ss[0],t,Is[0],t,Rs[0])
    axs[0][0].set_title('First node',fontsize=18.0)
    axs[0][0].grid()


    axs[0][1].plot(t, Ss[1],t,Is[1],t,Rs[1])
    axs[0][1].set_title('Second node',fontsize=18.0)
    axs[0][1].grid()

    axs[1][0].plot(t, Ss[2],t,Is[2],t,Rs[2])
    axs[1][0].set_title('Third node',fontsize=18.0)
    axs[1][0].grid()

    axs[1][1].plot(t, np.sum(Ss,0),t,np.sum(Is,0),t,np.sum(Rs,0))
    axs[1][1].set_title('Total',fontsize=18.0)
    axs[1][1].grid()



    # plt.setp(axs[-1, :], xlabel='day')
    # plt.setp(axs[:, 0], ylabel='population')

    axs[1][0].set_xlabel('Day', fontsize=18.0)
    axs[1][1].set_xlabel('Day', fontsize=18.0)


    axs[0][0].set_ylabel('Population', fontsize=18.0)
    axs[1][0].set_ylabel('Population', fontsize=18.0)


    plt.show()

# runAndPlot(N=1)