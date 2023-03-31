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

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


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

    def calculateBound(self, currentPath, bssf):
        cities = self._scenario.getCities()
        unvisitedCities = [city for city in cities if city not in currentPath]
        bound = bssf.cost
        lastCity = currentPath[-1]

        for city in unvisitedCities:
            newCost = lastCity.costTo(city)
            if newCost < np.inf:
                potentialPath = currentPath + [city]
                for i in range(len(potentialPath) - 1):
                    potentialPathCost = TSPSolution(potentialPath[:i + 1]).cost
                    if potentialPathCost == np.inf:
                        break
                else:
                    potentialBssf = TSPSolution(potentialPath)
                    bound += newCost + potentialBssf.cost
        return bound

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
        pass

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints:
        max queue size, total number of states created, and number of pruned states.</returns>
    '''

    def branchAndBound(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        n = len(cities)
        bssf = self.defaultRandomTour()['soln']
        start_time = time.time()
        # Initialize the priority queue with the root node
        priority_queue = [(0, Node(0, [i], 0, self.calculateBound([i], bssf))) for i in range(n)]
        max_queue_size = 1
        total_states_created = 1
        num_pruned_states = 0
        num_solutions_found = 0

        while priority_queue:
            # Check if we have run out of time
            if time.time() - start_time > time_allowance:
                break
            # Pop the node with the smallest bound from the priority queue
            _, node = heapq.heappop(priority_queue)
            # Generate children nodes
            for i in range(n):
                if i not in node.visited:
                    child_visited = node.visited + [i]
                    child_cost = node.cost + cities[node.visited[-1]].costTo(cities[i])
                    if child_cost < bssf.cost:
                        child_bound = self.calculateBound(child_visited, bssf)
                        if child_bound < bssf.cost:
                            # Update bssf if we have found a better solution
                            if len(child_visited) == n:
                                bssf = TSPSolution([cities[j] for j in child_visited])
                                num_solutions_found += 1
                            else:
                                # Add the child node to the priority queue
                                heapq.heappush(priority_queue,
                                               (child_bound, Node(i, child_visited, child_cost, child_bound)))
                            total_states_created += 1
                        else:
                            num_pruned_states += 1
                    else:
                        num_pruned_states += 1
            # Update the maximum size of the priority queue
            max_queue_size = max(max_queue_size, len(priority_queue))

        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = num_solutions_found
        results['soln'] = bssf
        results['max'] = max_queue_size
        results['total'] = total_states_created
        results['pruned'] = num_pruned_states
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


class Node:
    def __init__(self, current_city, visited, cost, bound):
        self.current_city = current_city
        self.visited = visited
        self.cost = cost
        self.bound = bound
