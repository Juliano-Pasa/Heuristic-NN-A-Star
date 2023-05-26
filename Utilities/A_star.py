from Utilities.Heuristics import *

def safe_astar(g, start, goal, v_weight, heuristic):
    opened = []
    visited = []

    visibility_weight = v_weight

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = visibility_weight * start.get_risk() + start.get_distance() + heuristic(start, goal)

    unvisited_queue = [(hscore, start)]
    heapq.heapify(unvisited_queue)

    count_visited = 0
    count_open = 1

    opened.append(start.get_coordinates())

    while len(unvisited_queue):
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        if current == goal:
            #print("ÇOCORRO DEUS\n\n\n\n\n",visited)
            #break
            distance = current.get_distance() + visibility_weight * current.get_risk()
            path=[]
            path=backtracking(current,start)
            
            #closed_nodes = list(map(lambda v: v.get_coordinates(), visited))
            return visited, len(path), count_open, path, distance 


        current.set_visited(True)
        count_visited = count_visited + 1
        visited.append(current.get_coordinates())

        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            new_dist = current.get_distance() + current.get_edge_weight(next_id)
            new_risk = current.get_risk() + next.get_local_risk()

            if new_dist + visibility_weight * new_risk < next.get_distance() + visibility_weight * next.get_risk():
                next.set_previous(current)
                next.set_distance(new_dist)
                next.set_risk(new_risk)

                hscore = visibility_weight * new_risk + new_dist + heuristic(next, goal)

                if not next.visited:
                    heapq.heappush(unvisited_queue, (hscore, next))
                    count_open = count_open + 1
                    opened.append(next.get_coordinates())

def astar_correction_factor(g, start, goal, v_weight, heuristic):
    opened = []
    visited = []

    visibility_weight = v_weight

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + r3_heuristic(start, goal)*heuristic[start.get_id(), 0]

    unvisited_queue = [(hscore, start)]
    heapq.heapify(unvisited_queue)

    count_visited = 0
    count_open = 1

    opened.append(start.get_coordinates())

    while len(unvisited_queue):
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        if current == goal:
            #print("ÇOCORRO DEUS\n\n\n\n\n",visited)
            #break
            distance = current.get_distance()
            path=[]
            path=backtracking(current,start)
            
            #closed_nodes = list(map(lambda v: v.get_coordinates(), visited))
            return visited, len(path), count_open, path, distance

        current.set_visited(True)
        count_visited = count_visited + 1
        visited.append(current.get_coordinates())

        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            new_dist = current.get_distance() + current.get_edge_weight(next_id)
            new_risk = current.get_risk() + next.get_local_risk()

            if new_dist < next.get_distance():
                next.set_previous(current)
                next.set_distance(new_dist)
                next.set_risk(new_risk)
                #print("retorno do fator de correção",heuristic(next,goal))
                hscore = new_dist + r3_heuristic(next, goal)*heuristic[next_id, 0]

                if not next.visited:
                    heapq.heappush(unvisited_queue, (hscore, next))
                    count_open = count_open + 1
                    opened.append(next.get_coordinates())

# A*
def astar(g, start, goal, v_weight, heuristic):
    opened = []
    visited = []
    heuristic_time = 0

    visibility_weight = v_weight

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + heuristic[start.get_id(), 0]

    unvisited_queue = [(hscore, start)]
    heapq.heapify(unvisited_queue)

    count_visited = 0
    count_open = 1

    opened.append(start.get_coordinates())

    while len(unvisited_queue):
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        if current == goal:
            #print("ÇOCORRO DEUS\n\n\n\n\n",visited)
            #break
            distance = current.get_distance()
            path=[]
            path=backtracking(current,start)
            
            #closed_nodes = list(map(lambda v: v.get_coordinates(), visited))
            #print("heuristic time: ", heuristic_time)
            return visited, len(path), count_open, path, distance

        current.set_visited(True)
        count_visited = count_visited + 1
        visited.append(current.get_coordinates())

        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            new_dist = current.get_distance() + current.get_edge_weight(next_id)
            new_risk = current.get_risk() + next.get_local_risk()

            if new_dist < next.get_distance():
                next.set_previous(current)
                next.set_distance(new_dist)
                next.set_risk(new_risk)

                t1 = time()
                hscore = new_dist + heuristic[next_id, 0]
                heuristic_time += time() - t1

                if not next.visited:
                    heapq.heappush(unvisited_queue, (hscore, next))
                    count_open = count_open + 1
                    opened.append(next.get_coordinates())
    '''
    opened = []
    expanded = [] #visitados e abertos

    visibility_weight = v_weight

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + heuristic(start, goal)

    opened = [(start, hscore)]
    

    count_visited = 0
    count_open = 1
    i=0
    i+=1
    #opened.append(start.get_coordinates())
    best = math.inf
    while len(opened):
        best=math.inf
        for i in range(len(opened)):
            x,y = opened[i]
            if y<best:
                best=y
                save=i
                
        uv = opened[save]
        current = uv[0]
        del opened[save]
        expanded.append(current)
        count_open+=1

        if current == goal:
            distance = current.get_distance()
            path = []
            path.append(goal.get_coordinates())
            count_visited=1
            print("salvando o path\n")
            while current.get_id() != start.get_id():
                path.append(current.get_coordinates())
                current = current.get_previous()
                count_visited=+1
            path.append(current.get_coordinates())
            closed_nodes = list(map(lambda v: v.get_coordinates(), expanded))
            return current.get_distance(), len(path), count_open, closed_nodes, path, distance

        current.set_visited(True)
        #visited.append(current.get_previous().get_coordinates())

        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            #mudar de edge weight para a distancia entre o nodo atual e o proximo na 2 parte da soma c
            #new_risk = current.get_risk() + next.get_local_risk()
            if next not in expanded:
                ind = 0
                newborn = True
                for ind in range(len(opened)):
                    c, hs = opened[ind]
                    if(c == next): 
                        newborn = False

                if newborn:
                    next.set_distance(math.inf)
                    next.set_previous(None)
                new_dist = current.get_distance() + r3_heuristic(current,next)
                #print("Eu nodo "+str(current.get_id())+"indo para o nodo "+str(next.get_id())+"custo r3"+str(r3_heuristic(current,next)))
                #print("Eu nodo "+str(current.get_id())+"indo para o nodo "+str(next.get_id())+"custo dnn"+str(heuristic(current,next)))

                if new_dist < next.get_distance():
                    next.set_previous(current)
                    next.set_distance(new_dist)
                #next.set_risk(new_risk)
                    for ind in range(len(opened)):
                            c, hs = opened[ind]
                            if next == c:                            
                                del opened[ind]
                                break
                

                    hscore = new_dist + heuristic(next, goal)

                    opened.append((next, hscore))
                    #count_open = count_open + 1
                    #opened.append(next.get_coordinates())'''


def astarmod(g, start, goal, v_weight, heuristic):
    opened = []
    visited = []

    visibility_weight = v_weight

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + heuristic(start, goal)

    unvisited_queue = [(hscore, start)]
    heapq.heapify(unvisited_queue)

    count_visited = 0
    count_open = 1

    opened.append(start.get_coordinates())

    while len(unvisited_queue):
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        if current == goal:
            #print("ÇOCORRO DEUS\n\n\n\n\n",visited)
            #break
            distance = current.get_distance()
            path=[]
            path=backtracking(current,start)
            
            #closed_nodes = list(map(lambda v: v.get_coordinates(), visited))
            return visited, len(path), count_open, path, distance
        #distance2, count_visited2, count_open2, opened2, visited2, cost2 = astarmod(g, source, dest, b, heuristic)

        current.set_visited(True)
        count_visited = count_visited + 1
        visited.append(current.get_coordinates())

        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            if calcula_angulo(current,next)<ANGULO_MAX:
                new_dist = current.get_distance() + current.get_edge_weight(next_id)
                new_risk = current.get_risk() + next.get_local_risk()

                if new_dist < next.get_distance():
                    next.set_previous(current)
                    next.set_distance(new_dist)
                    next.set_risk(new_risk)

                    hscore = new_dist + heuristic(next, goal)

                    if not next.visited:
                        heapq.heappush(unvisited_queue, (hscore, next))
                        count_open = count_open + 1
                        opened.append(next.get_coordinates())
    '''
    opened = []
    expanded = [] #visitados e abertos

    visibility_weight = v_weight

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + heuristic(start, goal)

    opened = [(start, hscore)]
    

    count_visited = 0
    count_open = 1
    i=0
    i+=1
    #opened.append(start.get_coordinates())
    best = math.inf
    while len(opened):
        best=math.inf
        for i in range(len(opened)):
            x,y = opened[i]
            if y<best:
                best=y
                save=i
                
        uv = opened[save]
        current = uv[0]
        del opened[save]
        expanded.append(current)
        count_open+=1

        if current == goal:
            distance = current.get_distance()
            path = []
            path.append(goal.get_coordinates())
            count_visited=1
            print("salvando o path\n")
            while current.get_id() != start.get_id():
                path.append(current.get_coordinates())
                current = current.get_previous()
                count_visited=+1
            path.append(current.get_coordinates())
            closed_nodes = list(map(lambda v: v.get_coordinates(), expanded))
            return current.get_distance(), len(path), count_open, closed_nodes, path, distance

        current.set_visited(True)
        #visited.append(current.get_previous().get_coordinates())
        
        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            #mudar de edge weight para a distancia entre o nodo atual e o proximo na 2 parte da soma c
            #new_risk = current.get_risk() + next.get_local_risk()
            if next not in expanded and calcula_angulo(current,next)<ANGULO_MAX:
                ind = 0
                newborn = True
                for ind in range(len(opened)):
                    c, hs = opened[ind]
                    if(c == next): 
                        newborn = False

                if newborn:
                    next.set_distance(math.inf)
                    next.set_previous(None)
                new_dist = current.get_distance() + heuristic(current,next)

                if new_dist < next.get_distance():
                    next.set_previous(current)
                    next.set_distance(new_dist)
                #next.set_risk(new_risk)
                    for ind in range(len(opened)):
                            c, hs = opened[ind]
                            if next == c:                            
                                del opened[ind]
                                break
                

                    hscore = new_dist + heuristic(next, goal)

                    opened.append((next, hscore))
                    #count_open = count_open + 1
                    #opened.append(next.get_coordinates())'''
