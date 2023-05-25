

def line_of_sight1(s,s1,g):#original
    if s is None or s1 is None:
        return False, 10
    x0,y0 = s.get_coordinates()
    #print(x0,y0)
    x1,y1 = s1.get_coordinates()
    #print(x1,y1)
    dy=y1 - y0
    dx=x1 - x0
    sy = 1
    sx = 1
    if dy < 0:
        dy = -dy
        sy=-1
    if dx < 0:
        dx = -dx
        sx = -1
    cost = 0
    f=0
    w=0
    
    #calcula o angulo aqui 
    x_s,y_s = s.get_coordinates()
    x_s1,y_s1 = s1.get_coordinates()



    # 
    # equação da reta   y = m*x+n
    #  

    # m = tangente do angulo alfa
    if x_s1-x_s==0:
        m=0
    else:
        m = (y_s1-y_s)/(x_s1-x_s)

    #substituimos um dos pontos na equação da reta para obter o N
    n = m*x_s - y_s

    #if n != m*x_s1-y_s1:
    #    print("Equação da reta está com incosistencias ",n , m*x_s1-y_s1)

    #agora que temos o n podemos calcular a reta em qualquer ponto dando o x e y como input

    # y = m*x+n
    


    
    #definir uma variavel que vai definir se o desnivel é muito grande ou nao
    
    #edge cost é calculado na lineofsight
    # cost padrão é calculado usando pitagoras entre a distancia e a altura
    
    #                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),  # peso da aresta = distância Eclidiana no R3
    #                                     vertex.get_elevation(), vertex2.get_elevation())
    cost=0
    if dx >= dy:
        #cost = cost + (g.get_vertex(get_id_by_coords(x1,y1)).get_elevation()/2)
        while x0 != x1:
            f = f + dy
            #x_i_vertex = x0 + int((sx-1)/2)
            #y_j_vertex = y0 + int((sy-1)/2)
            
            vert_src = g.get_vertex_by_coords(x0,y0)
            vert_tgt = g.get_vertex_by_coords(x0 + int((((sx-1)/2))),y0 + int((sy-1)/2))
            vert_id = vert_tgt.get_id()
            #print("aaaaa vertex1 550",vert)
            #print("aaaaa vertex2 550",vert1)
            #print("vert1 e 2 peso", vert.get_edge_weight(vert1.get_id()))
            '''if m == 0:
                multiplicador=1
            else:
                x_mult = calcula_x(y0,m,n)
                y_mult = calcula_y(x0,m,n)
                multiplicador = r2_distance(x0,x_mult,y0,y_mult)
            flag, valor = calcula_custo_theta(vert_src,vert_tgt,multiplicador)
            if(flag):
                cost = cost + valor'''


            
            #idinicial=get_id_by_coords(x0,y0)
            if f>= dx:
                if (vert_src.get_edge_angle(vert_id))>ANGULO_MAX:#g.get_vertex(get_id_by_coords(x0 + int((sx-1)/2),y0 + int((sy-1)/2))):
                    #print("ENTREI AQUI")
                    return False, math.inf
                y0 = y0 + sy
                f = f - dx

            if f!=0 and (vert_src.get_edge_angle(vert_id))>ANGULO_MAX:
                return False, math.inf
            new_vert1= get_id_by_coords(x0 + int((((sx-1)/2))),y0)
            #new_vert_id1=new_vert1.get_id()
            new_vert2= get_id_by_coords(x0 + int((((sx-1)/2))),y0 - 1)
            #new_vert_id2=new_vert2.get_id()
            if dy==0 and (vert_src.get_edge_angle(new_vert1))>ANGULO_MAX and (vert_src.get_edge_angle(new_vert2))>ANGULO_MAX:
                return False, math.inf
            #cost = cost + (g.get_vertex(get_id_by_coords(x0 + int((sx-1)/2),y0 + int((sy-1)/2))).get_elevation()/2)
            x0 = x0 + sx



            #cost = cost + g.get_vertex_by_coords(x0 + int((((sx-1)/2))),y0 + int((sy-1)/2)).get_edge_weight(idinicial)
    else:
        #cost = cost + (g.get_vertex(get_id_by_coords(x1,y1)).get_elevation()/2)
        while y0 != y1:
            f = f + dx
            idinicial=get_id_by_coords(x0,y0)
            #x0 + int((sx-1)/2) = x0 + int((sx-1)/2)
            #y0 + int((sy-1)/2) = y0 + int((sy-1)/2)
            vert_src = g.get_vertex_by_coords(x0,y0)
            vert_tgt = g.get_vertex_by_coords(x0 + int((((sx-1)/2))),y0 + int((sy-1)/2))
            vert_id = vert_tgt.get_id()
            '''if m == 0:
                multiplicador=1
            else:
                x_mult = calcula_x(y0,m,n)
                y_mult = calcula_y(x0,m,n)
                multiplicador = r2_distance(x0,x_mult,y0,y_mult)
            flag, valor = calcula_custo_theta(vert_src,vert_tgt,multiplicador)
            if(flag):
                cost = cost + valor'''

            #print("aaaaa vertex1 550",vert)
            if f >= dy:
                vert = g.get_vertex_by_coords(x0,y0)
                #print("aaaaa vertex",vert)
                if (vert_src.get_edge_angle(vert_id))>ANGULO_MAX:
                    return False, math.inf
                
                x0 = x0 + sx
                f = f - dy

            if f != 0 and (vert_src.get_edge_angle(vert_id))>ANGULO_MAX:
                return False, math.inf
            
            new_vert1= get_id_by_coords(x0,y0 + int((sy-1)/2))
            #new_vert_id1=new_vert1.get_id()
            new_vert2= get_id_by_coords(x0 - 1,y0 + int((sy-1)/2))
            
            if dx == 0 and (vert_src.get_edge_angle(new_vert1))>ANGULO_MAX and (vert_src.get_edge_angle(new_vert2))>ANGULO_MAX:
                vert = g.get_vertex_by_coords(x0,y0)
                #print("aaaaa vertex",vert)
                return False, math.inf
            y0 = y0 + sy
            #print("AAAAAAAAAAAAAAAAAA ",x0 + int((sx-1)/2),y0 + int((sy-1)/2))
            #print("cords",x0,y0)
            #print("aaaaaaaaaa",g.get_vertex(idinicial))
            #print("aaa",vert.get_edge_weight(28237))
            #cost = cost + vert.get_edge_weight(idinicial)
            #cost = cost + (g.get_vertex(get_id_by_coords(x_i_vertex,y_j_vertex)).get_elevation()/2)
    return True, cost


def theta_custo_diferente(g, start, goal, v_weight, heuristic):
    opened = []
    expanded = []

    visibility_weight = v_weight

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + r3_heuristic(start, goal)

    opened = [(start, hscore)]

    count_visited = 0
    count_open = 1
    
    i=0
    i+=1
    best=math.inf
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
        count_open += 1

        if current == goal:
            distance = current.get_distance()
            path = []
            path.append(goal.get_coordinates())
            count_visited=1
            print("salvando o path\n")
            while current.get_id() != start.get_id():
                path.append(current.get_coordinates())
                current = current.get_previous()
                count_visited+=1
            path.append(current.get_coordinates())
            closed_nodes = list(map(lambda v: v.get_coordinates(), expanded))
            return current.get_distance(), count_visited, count_open, closed_nodes, path, distance
        
        for next_id in current.get_neighbors():
            
            child = g.get_vertex(next_id)
            if child not in expanded:               
                
                ind = 0
                newborn = True
                for ind in range(len(opened)):
                    c, hs = opened[ind]
                    if(c == child): 
                        newborn = False

                if newborn:
                    child.set_distance(math.inf)
                    child.set_previous(None)

                grand_father = current.get_previous()
                flag,cost =line_of_sight1(grand_father, child, g)
                if grand_father is not None and flag:
                    if grand_father.get_distance() + cost < child.get_distance():
                        child.set_distance(grand_father.get_distance() + cost)
                        child.set_previous(grand_father)
                        
                        #Ineficiente, refatorar com alguma built-in function
                        ind = 0
                        for ind in range(len(opened)):
                            c, hs = opened[ind]
                            if child == c:                            
                                del opened[ind]
                                break

                        opened.append((child, child.get_distance() + cost)) # verificar
                else:
                    if current.get_distance() + r3_heuristic(current, child) < child.get_distance():
                        child.set_distance(current.get_distance() + r3_heuristic(current, child))#substituir por edge cost? precisa deixar coerente.
                        child.set_previous(current)

                        #Ineficiente, refatorar com alguma built-in function
                        ind = 0
                        for ind in range(len(opened)):
                            c, hs = opened[ind]
                            if child == c:                            
                                del opened[ind]
                                break
                        

                        opened.append((child, child.get_distance() + r3_heuristic(child, goal))) # verificar
                        
def theta_rapido(g, start, goal, v_weight, heuristic):
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
            
            grand_father = current.get_previous()
            
            
            flag,cost =line_of_sight1(grand_father, next, g)
            if grand_father is not None and flag:
                grand_son_risk = new_risk + grand_father.get_risk() 
                '''r3_heuristic(grand_father,next)'''
                if grand_father.get_distance() + r3_heuristic(grand_father,next) + grand_son_risk * visibility_weight < next.get_distance() + visibility_weight * next.get_risk():
                    next.set_distance(grand_father.get_distance() + r3_heuristic(grand_father,next))
                    next.set_previous(grand_father)
                    next.set_risk(grand_son_risk)
                    
                    hscore = visibility_weight * grand_son_risk + next.get_distance() + r3_heuristic(next,goal)
                    #hscore = next.get_distance() + r3_heuristic(grandfather,next)
                    if not next.visited:
                        heapq.heappush(unvisited_queue, (hscore, next))
                        count_open = count_open + 1
                        opened.append(next.get_coordinates())
            elif new_dist + visibility_weight * new_risk < next.get_distance() + visibility_weight * next.get_risk():
                next.set_previous(current)
                next.set_distance(new_dist)
                next.set_risk(new_risk)

                hscore = visibility_weight * new_risk + next.get_distance() + heuristic(next, goal)

                if not next.visited:
                    heapq.heappush(unvisited_queue, (hscore, next))
                    count_open = count_open + 1
                    opened.append(next.get_coordinates())
    
# A* adaptado com fator de segurança no cálculo do custo
def theta(g, start, goal, v_weight, heuristic):
    opened = []
    expanded = []

    visibility_weight = v_weight

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + r3_heuristic(start, goal)

    opened = [(start, hscore)]

    count_visited = 0
    count_open = 1
    
    i=0
    i+=1
    best=math.inf
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
        count_open += 1

        if current == goal:
            distance = current.get_distance()
            path = []
            path.append(goal.get_coordinates())
            count_visited=1
            print("salvando o path\n")
            while current.get_id() != start.get_id():
                path.append(current.get_coordinates())
                current = current.get_previous()
                count_visited+=1
            path.append(current.get_coordinates())
            closed_nodes = list(map(lambda v: v.get_coordinates(), expanded))
            return closed_nodes, len(path), count_open, path, distance
        
        for next_id in current.get_neighbors():
            
            child = g.get_vertex(next_id)
            if child not in expanded:               
                
                ind = 0
                newborn = True
                for ind in range(len(opened)):
                    c, hs = opened[ind]
                    if(c == child): 
                        newborn = False

                if newborn:
                    child.set_distance(math.inf)
                    child.set_previous(None)

                grand_father = current.get_previous()
                if grand_father is not None and line_of_sight1(grand_father, child, g):
                    if grand_father.get_distance() + r3_heuristic(grand_father, child) < child.get_distance():
                        child.set_distance(grand_father.get_distance() + r3_heuristic(grand_father, child))
                        child.set_previous(grand_father)
                        
                        #Ineficiente, refatorar com alguma built-in function
                        ind = 0
                        for ind in range(len(opened)):
                            c, hs = opened[ind]
                            if child == c:                            
                                del opened[ind]
                                break

                        opened.append((child, child.get_distance() + r3_heuristic(child, goal))) # verificar
                else:
                    if current.get_distance() + r3_heuristic(current, child) < child.get_distance():
                        child.set_distance(current.get_distance() + r3_heuristic(current, child))#substituir por edge cost? precisa deixar coerente.
                        child.set_previous(current)

                        #Ineficiente, refatorar com alguma built-in function
                        ind = 0
                        for ind in range(len(opened)):
                            c, hs = opened[ind]
                            if child == c:                            
                                del opened[ind]
                                break
                        

                        opened.append((child, child.get_distance() + r3_heuristic(child, goal))) # verificar
                
            
            #print("lista",unvisited_queue)


            '''
            g1 = current.get_elevation() + heuristica_padrao(current,c_child)
            line= line_of_sight(current.get_previous(), c_child,g)
            g2 = current.get_previous().get_elevation() + heuristica_padrao(current.get_previous(),c_child)

            if line == True and g2<=g1:
                #print("entrei")
                c_child.set_previous(current.get_previous())
                c_child.set_distance(g2)
                hscore = g2 + heuristic(c_child, goal)
                heapq.heappush(unvisited_queue, (hscore, c_child))
                count_open = count_open + 1
                opened.append(c_child.get_coordinates())
            else:
                c_child.set_distance(g1)
                c_child.set_previous(current)

                hscore = g1 + heuristic(c_child, goal)
                heapq.heappush(unvisited_queue, (hscore, c_child))
                count_open = count_open + 1
                opened.append(c_child.get_coordinates())'''


            '''if line_of_sight(current.get_previous(), child, g):
                c_child.set_previous(current.get_previous())

            else:
                c_child.set_previous(current)
                c_child.set_distance(g_cost)
                c_child.set_risk(new_risk)'''

                


'''def update_vertex(parent, child):
    grand_father = parent.get_previous()
    if line_of_sight1(grand_father, child):
        if grand_father.get_distance() + r3_distance(grand_father, child) < child.get_distance():
            child.set_distance(grand_father.get_distance() + r3_distance(grand_father, child))
            child.set_previous(grand_father)
            if child in opened:
                opened.remove(child) # verificar
            opened.append(child, child.get_distance() + heuristic(child, goal)) # verificar
    else:
        if parent.get_distance() + r3_distance(parent, child) < child.get_distance():
            child.set_distance(parent.get_distance() + r3_distance(parent, child))#substituir por edge cost? precisa deixar coerente.
            child.set_previous(parent)
            if (child, _) in opened:
                opened.remove((child, _)) # verificar
            opened.append(child, child.get_distance() + heuristic(child, goal)) # verificar'''

