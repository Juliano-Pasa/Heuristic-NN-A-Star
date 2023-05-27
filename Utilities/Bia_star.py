from Utilities.Heuristics import *
import heapq

def generatePath(current, currentReversed, start, goal, expanded, expandedReverse, v_weight, count_open, heuristic,g):
    distance = current.get_distance() + currentReversed.get_distance() + r3_heuristic(current, currentReversed)
    path = []
    count_visited = 0

    #print("salvando o path\n")
    while currentReversed.get_id() != goal.get_id():
        path.append(currentReversed.get_coordinates())
        currentReversed = currentReversed.get_previous()

    path.append(currentReversed.get_coordinates())
    path = path[::-1]

    while current.get_id() != start.get_id():
        path.append(current.get_coordinates())
        current = current.get_previous()

    path.append(current.get_coordinates())

    expanded.extend(expandedReverse)
    #expanded.reverse()
    closed_nodes = list(map(lambda v: g.get_vertex(v).get_coordinates(), expanded))
    #print(expanded)
    return closed_nodes, len(path), count_open, path, distance

#Bidirectional A*
#Bidirectional A*
def biastar(g, start, goal, v_weight, heuristic, heuristic1):
    visited = [] #visitados e abertos
    heapq.heapify(visited)
    visitedReverse = []
    heapq.heapify(visitedReverse)

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)
    goal.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + heuristic(start, goal)
    unvisited_queue = [(hscore, start)]
    heapq.heapify(unvisited_queue)
    unvisited_queue_reverse = [(hscore, goal)]
    heapq.heapify(unvisited_queue_reverse)

    count_open = 2
    count_visited = 0
    i = 0
    i += 1
    
    while unvisited_queue and unvisited_queue_reverse:
        # Normal way

        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited(True)
        count_visited = count_visited + 1
        heapq.heappush(visited, current.get_id())

        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            new_dist = current.get_distance() +current.get_edge_weight(next_id) 

            if next.visitedReverse:
                openedR, count_visitedR, count_openR, visitedR, costR = generatePath(current, next, start, goal, visited, visitedReverse, v_weight, count_open, heuristic,g)
                #print("AAAAAAAAAAAA")
                #print(openedR)
                return openedR, count_visitedR, count_openR, visitedR, costR

            if next.has_parent():
                if next.get_previous().visitedReverse:
                    continue

            if new_dist < next.get_distance():
                next.set_previous(current)
                next.set_distance(new_dist)

                hscore = new_dist + heuristic(next, goal)

                if not next.visited:
                    heapq.heappush(unvisited_queue, (hscore, next))
                    count_open = count_open + 1

        # Reverse way

        uv = heapq.heappop(unvisited_queue_reverse)
        current = uv[1]
        current.set_visited_reverse(True)
        count_visited = count_visited + 1
        heapq.heappush(visitedReverse, current.get_id())
        
        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            new_dist = current.get_distance() + current.get_edge_weight(next_id) 

            if next.visited:
                openedR, count_visitedR, count_openR, visitedR, costR = generatePath(next, current, start, goal, visited, visitedReverse, v_weight, count_open, heuristic1,g)
                #print("AAAAAAAAAAAAAAAAAAAAA2")
                #print(openedR)
                return openedR, count_visitedR, count_openR, visitedR, costR

            if next.has_parent():
                if next.get_previous().visited:
                    continue

            if new_dist < next.get_distance():
                next.set_previous(current)
                next.set_distance(new_dist)

                hscore = new_dist + heuristic1(next, goal)

                if not next.visitedReverse:
                    heapq.heappush(unvisited_queue_reverse, (hscore, next))
                    count_open = count_open + 1

