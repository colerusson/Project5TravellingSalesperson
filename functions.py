def branchAndBound(self, time_allowance=60.0):
    # Create initial solution
    results = {}
    init_route = self.defaultRandomTour(time_allowance)['soln'].route
    init_cost = TSPSolution(init_route).cost
    queue = [(init_cost, init_route)]
    heapq.heapify(queue)

    count = 0
    foundTour = False
    start_time = time.time()
    # Search for optimal solution
    while queue:
        cost, route = heapq.heappop(queue)
        if cost > self.best_cost:
            # Stop searching this branch since it cannot improve the best solution found so far
            continue
        if len(route) == self.ncities:
            # Found a complete tour
            cost += route[-1].costTo(route[0])
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_route = route + [route[0]]
                foundTour = True
        else:
            # Branch by considering all possible cities to visit next
            last_city = route[-1]
            for city in self._scenario.getCities():
                if city not in route:
                    new_cost = cost + last_city.costTo(city)
                    if new_cost < self.best_cost:
                        new_route = route + [city]
                        new_queue_item = (new_cost, new_route)
                        heapq.heappush(queue, new_queue_item)

        count += 1

    end_time = time.time()
    # Build and return the results dictionary
    results['cost'] = self.best_cost if foundTour else math.inf
    results['time'] = end_time - start_time
    results['count'] = count
    results['soln'] = TSPSolution(self.best_route) if foundTour else None
    results['max'] = None
    results['total'] = None
    results['pruned'] = None
    return results


def branchAndBound(self, time_allowance=60.0):
    start_time = time.time()
    break_time = start_time + time_allowance
    count = 0
    foundTour = False
    self.best_route = None
    self.best_cost = math.inf

    # Initialize priority queue
    queue = []
    start_state = (None, set(range(self.ncities)), [], 0)
    queue.append(start_state)

    # Run the branch and bound algorithm
    while queue and time.time() < break_time:
        node = queue.pop(0)
        node_edges = node[0]
        available_nodes = set(range(self.ncities)) - set(visited_nodes)
        path_so_far = node[2]
        path_cost = node[3]

        # Check if we have completed the path and found a better solution
        if not available_nodes and path_cost < self.best_cost:
            self.best_route = path_so_far
            self.best_cost = path_cost
            foundTour = True

        # If we still have nodes to visit, explore them
        if available_nodes:
            for next_node in available_nodes:
                next_edges = set(filter(lambda e: next_node in e, available_nodes))
                for edge in next_edges:
                    next_available_nodes = available_nodes - {next_node}
                    next_path_so_far = path_so_far + [(edge[0], edge[1])]
                    next_path_cost = path_cost + self.edges[edge[0]][edge[1]]
                    queue.append((edge, next_available_nodes, next_path_so_far, next_path_cost))

        # Sort the queue by lower bound estimate
        queue = sorted(queue, key=lambda x: x[3])

        count += 1

    end_time = time.time()
    # Build and return the results dictionary
    results = {}
    results['cost'] = self.best_cost if foundTour else math.inf
    results['time'] = end_time - start_time
    results['count'] = count
    results['soln'] = TSPSolution(self.best_route) if foundTour else None
    results['max'] = None
    results['total'] = None
    results['pruned'] = None
    return results





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
