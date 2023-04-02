#!/usr/bin/python3

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


def reduceMatrix(matrix):
    matrixReduction = np.copy(matrix)
    reductionCost = 0

    for row in matrixReduction:
        minVal = np.min(row)
        if minVal != np.inf:
            reductionCost += minVal
        for i in range(len(row)):
            if row[i] != np.inf:
                row[i] -= minVal

    for i in range(matrixReduction.shape[1]):
        col = matrixReduction[:, i]
        minVal = np.min(col)
        if minVal != np.inf:
            reductionCost += minVal
        for j in range(len(col)):
            if col[j] != np.inf:
                col[j] -= minVal

    if reductionCost == np.inf:
        reductionCost = 0

    return matrixReduction, reductionCost


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
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
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
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
        cities = self._scenario.getCities()

        distances = moveCitiesToArray(cities)

        n = distances.shape[0]
        startCity = random.randint(0, n - 1)
        path = [startCity]
        totalDistance = 0

        start_time = time.time()
        while len(path) < n:
            if time.time() - start_time > time_allowance:
                return None
            currentCity = path[-1]
            nextCity = None
            minDistance = np.inf

            for city in range(n):
                if city not in path:
                    cost = distances[currentCity, city]
                    if cost < minDistance:
                        minDistance = cost
                        nextCity = city

            if nextCity is None:
                unvisitedCities = set(range(n)) - set(path)
                startCity = random.choice(list(unvisitedCities))
                path.append(startCity)
                continue

            path.append(nextCity)
            totalDistance += minDistance

        end_time = time.time()
        totalDistance += distances[path[-1], startCity]

        route = []
        for city in path:
            route.append(cities[city])
        bssf = TSPSolution(route)

        results = {}
        results['cost'] = totalDistance
        results['time'] = end_time - start_time
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
        cities = self._scenario.getCities()
        numCities = len(cities)
        bssf = self.greedy()['cost']
        matrix = moveCitiesToArray(cities)

        matrix, reductionValue = reduceMatrix(matrix)

        queue = []

        start_node = Node(matrix, reductionValue, 0)
        start_node.addToPath(0)
        heapq.heappush(queue, start_node)

        totalPruned = 0
        numSolution = 0
        nodesCreated = 0
        maxStorage = 0
        bestNodeSoFar = None

        start_time = time.time()

        while queue:
            parentNode = heapq.heappop(queue)
            parentMatrix = parentNode.matrix

            row = parentMatrix[parentNode.parent]

            for rowIndex, distanceToCity in enumerate(row):
                if distanceToCity == np.inf:
                    continue

                if len(queue) > maxStorage:
                    maxStorage = len(queue)

                childMatrix = np.copy(parentNode.matrix)

                childMatrix[parentNode.parent, :] = np.inf
                childMatrix[:, rowIndex] = np.inf
                childMatrix[rowIndex, parentNode.parent] = np.inf

                reducedMatrix, reductionValue = reduceMatrix(childMatrix)
                nodesCreated += 1

                childNode = Node(reducedMatrix, parentNode.cost + distanceToCity + reductionValue, rowIndex)
                childNode.addPathToPath(parentNode.path)
                childNode.addToPath(rowIndex)

                if len(childNode.path) == numCities:
                    numSolution += 1
                    if childNode.cost <= bssf:
                        bssf = childNode.cost
                        bestNodeSoFar = childNode
                elif childNode.cost <= bssf:
                    heapq.heappush(queue, childNode)
                else:
                    totalPruned += 1

        endTime = time.time()

        solutionPath = []
        for cityIndex in bestNodeSoFar.path:
            solutionPath.append(cities[cityIndex])

        bestSolution = TSPSolution(solutionPath)
        results = {}
        results['cost'] = bssf
        results['time'] = endTime - start_time
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
        pass


def moveCitiesToArray(cities_list):
    numCities = len(cities_list)
    citiesMatrix = np.zeros((numCities, numCities))

    for row in range(numCities):
        for col in range(numCities):
            citiesMatrix[row][col] = cities_list[row].costTo(cities_list[col])

    return citiesMatrix


class Node:
    def __init__(self, matrix=None, cost=0, parent=0):
        self.matrix = matrix
        self.cost = cost
        self.parent = parent
        self.path = []

    def __lt__(self, other):
        return self.cost < other.cost

    def addToCost(self, cost):
        self.cost += cost

    def addToPath(self, index):
        self.path.append(index)

    def addPathToPath(self, path):
        for item in path:
            self.path.append(item)
