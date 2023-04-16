#!/usr/bin/python3
from collections import deque

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

from TSPClasses import *
import heapq


# function used to reduce the matrix by subtracting the minimum value in each row and column
def reduceMatrix(matrix):
    # initialize the reduction cost to 0 and create a copy of the matrix
    reductionCost = 0
    matrixReduction = np.copy(matrix)

    # reduce each row by the minimum value in that row
    for row in matrixReduction:
        # find the minimum value in the row
        minVal = np.min(row)
        # if the minimum value is not infinity, add it to the reduction cost
        if minVal != np.inf:
            reductionCost += minVal
        # subtract the minimum value from each element in the row
        for i in range(len(row)):
            if row[i] != np.inf:
                row[i] -= minVal

    # reduce each column by the minimum value in that column
    for i in range(matrixReduction.shape[1]):
        col = matrixReduction[:, i]
        # find the minimum value in the column
        minVal = np.min(col)
        # if the minimum value is not infinity, add it to the reduction cost
        if minVal != np.inf:
            reductionCost += minVal
        # subtract the minimum value from each element in the column
        for j in range(len(col)):
            if col[j] != np.inf:
                col[j] -= minVal

    # if the reduction cost is infinity, set it to 0
    if reductionCost == np.inf:
        reductionCost = 0

    return matrixReduction, reductionCost


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        # called in the GUI to give the solver a reference to the scenario
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution,
        time spent to find solution, number of permutations tried during search, the
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        # initialize the results dictionary
        results = {}
        # get the list of cities from the scenario
        cities = self._scenario.getCities()
        numCities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        # loop until a valid tour is found or the time allowance is exceeded
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(numCities)
            route = []
            # Now build the route using the random permutation
            for i in range(numCities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True

        end_time = time.time()

        # populate the results dictionary
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def greedy(self, time_allowance=60.0):
        # initialize the results dictionary
        results = {}
        # get the list of cities from the scenario
        cities = self._scenario.getCities()
        # get the number of cities
        distances = moveCitiesToArray(cities)
        # get the number of cities
        n = distances.shape[0]
        # get the starting city randomly
        startCity = random.randint(0, n - 1)
        # initialize the path to the starting city
        path = [startCity]
        # initialize the total distance to 0
        totalDistance = 0

        startTime = time.time()

        # loop until all cities have been visited or the time allowance is exceeded
        while len(path) < n:
            # if the time allowance is exceeded, break out of the loop
            if time.time() - startTime > time_allowance:
                return None

            # initialize the next city to None and the minimum distance to infinity
            nextCity = None
            currentCity = path[-1]
            minDistance = np.inf

            # loop through all the cities
            for city in range(n):
                # if the city has not been visited, check if it is the next city
                if city not in path:
                    cost = distances[currentCity, city]
                    # if the cost is less than the minimum distance, update the minimum distance and the next city
                    if cost < minDistance:
                        minDistance = cost
                        nextCity = city

            # if the next city is None, go to a random unvisited city
            if nextCity is None:
                # get the set of unvisited cities
                unvisitedCities = set(range(n)) - set(path)
                # get a random unvisited city
                startCity = random.choice(list(unvisitedCities))
                # add the random unvisited city to the path
                path.append(startCity)
                continue

            # add the next city to the path
            totalDistance += minDistance
            path.append(nextCity)

        endTime = time.time()

        # add the distance from the last city to the starting city
        totalDistance += distances[path[-1], startCity]
        route = []

        # create the route
        for city in path:
            route.append(cities[city])

        # create the BSSF
        bssf = TSPSolution(route)

        # populate the results dictionary
        results['cost'] = totalDistance
        results['time'] = endTime - startTime
        results['count'] = 1
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints:
        max queue size, total number of states created, and number of pruned states.</returns>
    '''

    def branchAndBound(self, time_allowance=60.0):
        # initialize the results dictionary
        results = {}
        # get the list of cities from the scenario
        cities = self._scenario.getCities()
        # get the number of cities
        numCities = len(cities)
        # get the greedy solution as the initial BSSF
        bssf = self.greedy()['cost']
        if bssf is None:
            bssf = self.defaultRandomTour()['cost']

        # get the matrix of distances and reduce the matrix
        matrix = moveCitiesToArray(cities)
        matrix, reductionValue = reduceMatrix(matrix)

        # initialize the queue
        queue = []
        # create the start node
        start_node = Node(reductionValue, 0, matrix)
        start_node.addToPath(0)
        # add the start node to the queue
        heapq.heappush(queue, start_node)

        # initialize the other variables to keep track of the search
        totalPruned = 0
        numSolution = 0
        nodesCreated = 0
        maxStorage = 0
        bestNodeSoFar = None

        startTime = time.time()

        # loop until the queue is empty or the time allowance is exceeded
        while queue:
            # if the time allowance is exceeded, break out of the loop
            if time.time() - startTime > time_allowance:
                break

            # get the node with the lowest cost
            parentNode = heapq.heappop(queue)
            parentMatrix = parentNode.matrix
            currRow = parentMatrix[parentNode.parent]

            # loop through all the cities
            for rowIndex, distanceToCity in enumerate(currRow):
                # if the distance to the city is infinity, skip the city
                if distanceToCity == np.inf:
                    continue
                # if the length of the queue is greater than the max storage, update the max storage
                if len(queue) > maxStorage:
                    maxStorage = len(queue)

                # create the child matrix
                childMatrix = np.copy(parentNode.matrix)

                # set the row and column of the parent node to infinity within the child matrix
                childMatrix[parentNode.parent, :] = np.inf
                childMatrix[:, rowIndex] = np.inf
                childMatrix[rowIndex, parentNode.parent] = np.inf

                # reduce the matrix
                reducedMatrix, reductionValue = reduceMatrix(childMatrix)
                # update the number of nodes created
                nodesCreated += 1

                # create the child node
                childNode = Node(parentNode.cost + distanceToCity + reductionValue, rowIndex, reducedMatrix)
                childNode.addPathToPath(parentNode.path)
                childNode.addToPath(rowIndex)

                # if the child node is a solution, check if it is the best solution so far
                if len(childNode.path) == numCities:
                    # update the number of solutions found
                    numSolution += 1
                    # if the cost of the child node is less than the BSSF, update the BSSF and the best node so far
                    if childNode.cost <= bssf:
                        bssf = childNode.cost
                        bestNodeSoFar = childNode
                # if the child node is not a solution, check if it is pruned
                elif childNode.cost <= bssf:
                    heapq.heappush(queue, childNode)
                # if the child node is pruned, update the number of pruned nodes
                else:
                    totalPruned += 1

        endTime = time.time()

        # initialize the solution path
        solutionPath = []

        # add the cities to the solution path
        for cityIndex in bestNodeSoFar.path:
            solutionPath.append(cities[cityIndex])

        # create the best solution
        bestSolution = TSPSolution(solutionPath)

        # populate the results dictionary
        results['cost'] = bssf
        results['time'] = endTime - startTime
        results['count'] = numSolution
        results['soln'] = bestSolution
        results['max'] = maxStorage
        results['total'] = nodesCreated
        results['pruned'] = totalPruned

        return results

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found during search, the
        best solution found.  You may use the other three field however you like.
        algorithm</returns>
    '''

    def fancy(self, time_allowance=60.0):
        start_time = time.time()
        cities = self._scenario.getCities()
        adjacency_matrix = moveCitiesToArray(cities)
        results = {}

        s = list(range(len(adjacency_matrix)))
        c = getCost(adjacency_matrix, s)
        num_trials = 1
        T = 30
        alpha = 0.99
        while num_trials <= 1000 and time.time() - start_time < time_allowance:
            n = np.random.randint(0, len(adjacency_matrix))
            while True:
                m = np.random.randint(0, len(adjacency_matrix))
                if n != m:
                    break
            s1 = swap(s, m, n)
            c1 = getCost(adjacency_matrix, s1)
            if c1 < c:
                s, c = s1, c1
            else:
                if np.random.rand() < np.exp(-(c1 - c) / T):
                    s, c = s1, c1
            T = alpha * T
            num_trials += 1

        end_time = time.time()
        results['cost'] = c
        results['time'] = end_time - start_time
        results['count'] = None
        results['soln'] = None
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results


# helper function to move the cities objects into a matrix
def moveCitiesToArray(citiesList):
    # get the number of cities
    numCities = len(citiesList)
    # initialize the matrix of cities with zeros
    citiesMatrix = np.zeros((numCities, numCities))

    # loop through the matrix and set the values to the cost to travel to the city
    for i in range(numCities):
        for j in range(numCities):
            citiesMatrix[i][j] = citiesList[i].costTo(citiesList[j])

    return citiesMatrix


# helper class to represent a node in the branch and bound algorithm
class Node:
    # initialize the node with the cost, parent, and respective matrix
    def __init__(self, cost=0, parent=0, matrix=None):
        self.matrix = matrix
        self.cost = cost
        self.parent = parent
        self.path = []

    # define the less than operator for the node class
    def __lt__(self, other):
        return self.cost < other.cost

    # add the cost to the node
    def addToCost(self, cost):
        self.cost += cost

    # add the index to the path
    def addToPath(self, index):
        self.path.append(index)

    # add the path to the node
    def addPathToPath(self, path):
        for item in path:
            self.path.append(item)


def swap(s, m, n):
    i, j = min(m, n), max(m, n)
    s1 = s.copy()
    while i < j:
        s1[i], s1[j] = s1[j], s1[i]
        i += 1
        j -= 1
    return s1


def getCost(adjacency_matrix, s):
    cost = 0
    for i in range(len(s) - 1):
        cost += adjacency_matrix[s[i]][s[i + 1]]
    cost += adjacency_matrix[s[len(s) - 1]][s[0]]
    return cost
