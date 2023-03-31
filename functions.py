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