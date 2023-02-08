import io
import shutil
import random
import sys
import numpy as np
import math
import csv
import heapq
import os
import glob
from config_variables import TestVars, TestCase 
import matplotlib.image as mpimg
from time import time
from time import process_time
import pickle
#from   itertools import combinations
import datetime

import tensorflow                    as     tf
from   tensorflow                    import keras
from   tensorflow.keras              import losses
from   tensorflow.keras              import metrics
from   tensorflow.keras.callbacks    import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from   tensorflow.keras.initializers import GlorotNormal, GlorotUniform, HeNormal
from   tensorflow.keras.layers       import Dense, Dropout, Input
from   tensorflow.keras.models       import load_model, Sequential
from   tensorflow.keras.optimizers   import Adam

ANGULO_MAX =30


def r2_distance(x1, x2, y1, y2):
    return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


def r3_distance(x1, x2, y1, y2, z1, z2):
    return np.sqrt((r2_distance(x1,x2,y1,y2) ** 2) + ((z1 - z2) ** 2))


class Mde:
    # https://rasterio.readthedocs.io/en/latest/quickstart.html
    import rasterio

    # Parametros:
    # fp = nome do arquivo raster;
    # reduction_factor = grau de redução da dimensão do mapa
    def __init__(self, fp, reduction_factor):
        self.dataset = self.rasterio.open(fp)
        self.band1 = self.dataset.read(1)
        self.pixel_resolution = round(self.dataset.transform[0] * 108000)#usado quando os pixeis estão em graus por metro
        #self.pixel_resolution =self.dataset.transform[0] #usado quando o dataset os m por pixel estão em m
        print("\n\n\n\n\nmetadados:", self.pixel_resolution)

        print("\n\n\n\n\nmetadados:",self.dataset.height)
        self.h_limit = self.dataset.height
        self.w_limit = self.dataset.width

        # Redimensiona o grid
        self.generate_grid(reduction_factor)

        self.cell_size = self.pixel_resolution * reduction_factor
        global CELL_HEIGHT, CELL_WIDTH, GRID_ROWS, GRID_COLS, GRID_HEIGHT, GRID_WIDTH
        CELL_HEIGHT = self.pixel_resolution * reduction_factor
        CELL_WIDTH = self.pixel_resolution * reduction_factor
        GRID_COLS = self.grid.shape[0]
        GRID_ROWS = self.grid.shape[1]
        GRID_WIDTH = CELL_WIDTH * GRID_COLS
        GRID_HEIGHT = CELL_HEIGHT * GRID_ROWS

    # Gera o grid com dimensão reduzida
    # Ex: reduction_factor = 2, mapa 400x400 reduzido para 200x200
    def generate_grid(self, reduction_factor):
        x = int(self.h_limit / reduction_factor)
        y = int(self.w_limit / reduction_factor)
        self.grid = np.zeros(shape=(x, y))
        for j in range(x):
            for i in range(y):
                sub_section = self.band1[i * reduction_factor: (i + 1) * reduction_factor, j * reduction_factor: (j + 1) * reduction_factor]
                self.grid[i, j] = np.sum(sub_section)
                self.grid[i, j] = round(self.grid[i, j] / (len(sub_section) * len(sub_section[0])))

    def get_cell_size(self):
        return self.cell_size

class Vertex:
    def __init__(self, elevation, node_id):
        self.local_risk = 0
        self.elevation = elevation
        self.id = node_id
        self.edges = {}
        self.distance = 99999999    # Somatório da distância percorrida da origem até o vértice
        self.risk = 99999999    # Somatório do grau de visibilidade da origem até o vértice
        self.previous = None
        self.visited = False

    def __str__(self):
        return str(self.id) + ' elevation: ' + str(self.elevation) + ' coord: ' + str(self.get_r2_coordinates()) + ' edges: ' + str([x for x in self.edges.keys()]) + str([x for x in self.edges.values()])

    def get_previous(self):
        return self.previous

    def add_edge(self, node_id, edge_weight):
        self.edges[node_id] = edge_weight
        
    def get_visited(self):
        return self.visited

    def has_parent(self):
        if (self.previous):
            return True
        else:
            return False

    def get_id(self):
        return self.id

    def get_neighbors(self):
        return self.edges.keys()

    def get_x(self):
        return self.get_j() * CELL_WIDTH

    def get_y(self):
        return self.get_i() * CELL_HEIGHT

    def get_i(self):
        return math.floor(self.id / GRID_ROWS)

    def get_j(self):
        return self.id % GRID_COLS

    def get_r2_coordinates(self):
        return self.get_x(), self.get_y()

    def get_r3_coordinates(self):
        return self.get_x(), self.get_y(), self.elevation

    def get_coordinates(self):
        return self.get_i(), self.get_j()

    def get_elevation(self):
        return self.elevation

    def get_edge_weight(self, vertex_id):
        if self.get_id() == vertex_id:
            return math.inf
        if (self.edges[vertex_id] is None):
            return None
        return self.edges[vertex_id]

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self, visit):
        self.visited = visit

    def set_distance(self, distance):
        self.distance = distance

    def get_distance(self):
        return self.distance

    def set_risk(self, risk):
        self.risk = risk

    def get_risk(self):
        return self.risk

    # Reseta os valores do vértice para computar outro caminho utilizando o A*
    def reset(self):
        self.distance = 99999999
        self.risk = 99999999
        self.previous = None
        self.visited = False

    def set_local_risk(self, local_risk):
        self.local_risk = local_risk

    def get_local_risk(self):
        return self.local_risk

    def __lt__(self, other):
        return self.distance + self.risk < other.distance + other.risk

    def __eq__(self, other):
        return self.id == other.get_id()

class Graph:
    def __init__(self, mde):
        self.vertices = {}  # Dicionário de vértices: key = id, value = objeto Vertex
        self.max_edge = 0.0
        self.min_edge = float("inf")
        self.create_vertices(mde)   # Popula o dicionário de vértices com 1 vértice para cada célula do grid do mde
        self.generate_edges(False)  # Parâmetro: True = considera as travessias diagonais; False = considera apenas os vizinhos imediatos

    def __iter__(self):
        return iter(self.vertices.values())

    # Método para resetar o grafo a cada busca
    def reset(self):
        for v in self:
            v.reset()

    def __str__(self):
        for v in self:
            print(v)

    def create_vertices(self, mde):
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                vertex_elevation = mde.grid[i, j]
                vertex_id = i * GRID_COLS + j
                self.add_vertex(vertex_elevation, vertex_id)

    # Atribui o grau de visibilidade ao risco de cada vértice, baseado no parâmetro mapa de visibilidade "viewshed"
    def update_vertices_risk(self, viewshed):
        for v in self:
            i,j = v.get_coordinates()
            v.set_local_risk(viewshed[i,j])

    def get_vertex(self, id):
        if id in self.vertices:
            return self.vertices[id]
        else:
            return None

    def get_vertex_by_coords(self, i, j):
        id = get_id_by_coords(i,j)
        return self.get_vertex(id)

    def get_vertices(self):
        return self.vertices.keys()

    def add_vertex(self, elevation, id):
        self.vertices[id] = Vertex(elevation, id)
    

    def generate_edges(self, diagonal):
        for vertex_id, vertex in self.vertices.items():
            i, j = vertex.get_coordinates()

            j1 = j + 1
            i1 = i
            if j1 < GRID_COLS:
                vertex2_id = i1 * GRID_COLS + j1
                vertex2 = self.get_vertex(vertex2_id)
                if vertex2:
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),  # peso da aresta = distância Eclidiana no R3
                                         vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight)


            j1 = j - 1
            i1 = i
            if j1 >= 0:
                vertex2_id = i1 * GRID_COLS + j1
                vertex2 = self.get_vertex(vertex2_id)
                if vertex2:
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                           vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight)


            j1 = j
            i1 = i + 1
            if i1 < GRID_ROWS:
                vertex2_id = i1 * GRID_COLS + j1
                vertex2 = self.get_vertex(vertex2_id)
                if vertex2:
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                           vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight)

            j1 = j
            i1 = i - 1
            if i1 >= 0:
                vertex2_id = i1 * GRID_COLS + j1
                vertex2 = self.get_vertex(vertex2_id)
                if vertex2:
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                           vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight)

            if diagonal:
                j1 = j + 1
                i1 = i + 1
                if j1 < GRID_COLS and i1 < GRID_ROWS:
                    vertex2_id = i1 * GRID_COLS + j1
                    vertex2 = self.get_vertex(vertex2_id)
                    if vertex2:
                        weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                             vertex.get_elevation(), vertex2.get_elevation())
                        if weight > self.max_edge:
                            self.max_edge = weight
                        if weight < self.min_edge:
                            self.min_edge = weight
                        vertex.add_edge(vertex2_id, weight)

                j1 = j - 1
                i1 = i + 1
                if j1 >= 0 and i1 < GRID_ROWS:
                    vertex2_id = i1 * GRID_COLS + j1
                    vertex2 = self.get_vertex(vertex2_id)
                    if vertex2:
                        weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                             vertex.get_elevation(), vertex2.get_elevation())
                        if weight > self.max_edge:
                            self.max_edge = weight
                        if weight < self.min_edge:
                            self.min_edge = weight
                        vertex.add_edge(vertex2_id, weight)

                j1 = j + 1
                i1 = i - 1
                if i1 >= 0 and j1 < GRID_COLS:
                    vertex2_id = i1 * GRID_COLS + j1
                    vertex2 = self.get_vertex(vertex2_id)
                    if vertex2:
                        weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                             vertex.get_elevation(), vertex2.get_elevation())
                        if weight > self.max_edge:
                            self.max_edge = weight
                        if weight < self.min_edge:
                            self.min_edge = weight
                        vertex.add_edge(vertex2_id, weight)

                j1 = j - 1
                i1 = i - 1
                if i1 >= 0 and j1 >= 0:
                    vertex2_id = i1 * GRID_COLS + j1
                    vertex2 = self.get_vertex(vertex2_id)
                    if vertex2:
                        weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                             vertex.get_elevation(), vertex2.get_elevation())
                        if weight > self.max_edge:
                            self.max_edge = weight
                        if weight < self.min_edge:
                            self.min_edge = weight
                        vertex.add_edge(vertex2_id, weight)

    # Passa os valores de visbilidade [0, 1] para [0, max(edge) - min(edge)]
    def normalize_visibility(self, visibility):
        return visibility * (self.max_edge - self.min_edge)

# Recupera o caminho percorrendo do fim para o inicio
def shortest(v, path):
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return


# Heuristica distancia Euclidiana no R3
def r3_heuristic(start, goal):
    x1, y1 = start.get_r2_coordinates()
    x2, y2 = goal.get_r2_coordinates()
    z1 = start.get_elevation()
    z2 = goal.get_elevation()

    dst = r3_distance(x1, x2, y1, y2, z1, z2)

    return dst

def heuristica_padrao(start,goal):
    x1, y1 = start.get_r2_coordinates()
    x2, y2 = goal.get_r2_coordinates()

    return r2_distance(x1,x2,y1,y2)




def calcula_angulo(vert,vert1):
    if vert is None or vert1 is None:
        return math.inf
    if vert == vert1:
        return math.inf
    #if vertex1.get_id()==vertex2.get_id()
    #aqui é feito o calculo onde é verificado se o caminho está "bloquado", levando em consideração se de um ponto ao outro a elevação é maior que 30%
    altura = abs(vert.get_elevation()-vert1.get_elevation())
    hipotenusa = r3_distance(vert.get_x(),vert1.get_x(),vert.get_y(),vert1.get_y(),vert.get_elevation(),vert1.get_elevation())
    seno = altura/hipotenusa
    #print("hipotenusa: ", hipotenusa)
    #print("lalala seno: ",math.degrees(math.sin(seno)))
    return math.degrees(math.sin(seno))
    
def calcula_hipotenusa(vertex1,vertex2,g):
    #aqui é feito o calculo onde é verificado se o caminho está "bloquado", levando em consideração se de um ponto ao outro a elevação é maior que 30%
    distancia = vertex2.get_edge_weight(vertex1.get_id())
    altura = abs(vertex2.get_elevation()-vertex1.get_elevation())
    hipotenusa = math.sqrt(distancia**2+altura**2)    
    return hipotenusa

def calcula_custo_theta(vert1,vert2,multiplicador):
    #multiplicador é um valor de 0~~1 que é relativo ao tanto que a linha cruzou pelo nodo
    if vert1 is None or vert2 is None:
        return False, math.inf
    return True, r3_heuristic(vert1,vert2)*multiplicador

def calcula_y(x,m,n):
    return m*x-n

def calcula_x(y,m,n):
    return (y+n)/m

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
            #print("aaaaa vertex1 550",vert)
            #print("aaaaa vertex2 550",vert1)
            #print("vert1 e 2 peso", vert.get_edge_weight(vert1.get_id()))
            if m == 0:
                multiplicador=1
            else:
                x_mult = calcula_x(y0,m,n)
                y_mult = calcula_y(x0,m,n)
                multiplicador = r2_distance(x0,x_mult,y0,y_mult)
            flag, valor = calcula_custo_theta(vert_src,vert_tgt,multiplicador)
            if(flag):
                cost = cost + valor


            
            #idinicial=get_id_by_coords(x0,y0)
            if f>= dx:
                if calcula_angulo(vert_src,vert_tgt)>ANGULO_MAX:#g.get_vertex(get_id_by_coords(x0 + int((sx-1)/2),y0 + int((sy-1)/2))):
                    #print("ENTREI AQUI")
                    return False, math.inf
                y0 = y0 + sy
                f = f - dx

            if f!=0 and calcula_angulo(vert_src, vert_tgt)>ANGULO_MAX:
                return False, math.inf
            if dy==0 and calcula_angulo(vert_src, g.get_vertex_by_coords(x0 + int((((sx-1)/2))),y0))>ANGULO_MAX and calcula_angulo(vert_src, g.get_vertex_by_coords(x0 + int((((sx-1)/2))),y0 - 1))>ANGULO_MAX:
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

            if m == 0:
                multiplicador=1
            else:
                x_mult = calcula_x(y0,m,n)
                y_mult = calcula_y(x0,m,n)
                multiplicador = r2_distance(x0,x_mult,y0,y_mult)
            flag, valor = calcula_custo_theta(vert_src,vert_tgt,multiplicador)
            if(flag):
                cost = cost + valor

            #print("aaaaa vertex1 550",vert)
            if f >= dy:
                vert = g.get_vertex_by_coords(x0,y0)
                #print("aaaaa vertex",vert)
                if calcula_angulo(vert_src,vert_tgt)>ANGULO_MAX:
                    return False, math.inf
                
                x0 = x0 + sx
                f = f - dy

            if f != 0 and calcula_angulo(vert_src,vert_tgt)>ANGULO_MAX:
                return False, math.inf
            
            
            if dx == 0 and calcula_angulo(vert_src, g.get_vertex_by_coords(x0,y0 + int((sy-1)/2)))>ANGULO_MAX and calcula_angulo(vert_src, g.get_vertex_by_coords(x0 - 1,y0 + int((sy-1)/2)))>ANGULO_MAX:
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


def CalculateCostNonUniform(child,current,grid):
    g2=math.inf
    linha=False
    
    g1 = current.get_distance() + calcula_hipotenusa(current,child,grid)
    
    #var = current.get_previous()
    
    if(current.has_parent() and not current.get_previous()==child):
        linha, cost = line_of_sight1(current.get_previous(),child,grid)
        
        altura = abs(current.get_previous().get_elevation()-child.get_elevation())
        hipotenusa = math.sqrt(cost**2+altura**2)    
        
        g2 = current.get_previous().get_elevation() + hipotenusa


    if linha and g2 <= g1:
        parent = current.get_previous()
        cost = g2
    else:
        parent = current
        cost = g1
    return (cost , parent)

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
                        child.set_distance(grand_father.get_distance() + r3_heuristic(grand_father, child))
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


# A*
def astar(g, start, goal, v_weight, heuristic):
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
                    #opened.append(next.get_coordinates())

def astarmod(g, start, goal, v_weight, heuristic):
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
                    #opened.append(next.get_coordinates())

def get_visited_coord(graph, visited_vertices):
    path = []
    for vertex_id in visited_vertices[::-1]:
        path.append(graph.get_vertex(vertex_id).get_coordinates())
    return path


# Recupera o id do vértice a partir das coordenadas no grid
def get_id_by_coords(i, j):
    return i * GRID_COLS + j


def save_path_csv(output, path):
    if os.path.exists(output):
        os.remove(output)
    with open(output, 'w') as out:
        csv_out = csv.writer(out)
        for row in path:
            csv_out.writerow(row)


def write_dataset_csv(filename, data_io):
    with open(filename, 'a') as file:
        data_io.seek(0)
        shutil.copyfileobj(data_io, file)


# Gera e salva os mapas de visibilidade em arquivos png
def save_viewsheds(grid, viewpoints, view_radius, viewpoint_height):
    todos = np.zeros((grid.shape[0], grid.shape[1]))
    for viewpoint_i, viewpoint_j in viewpoints:
        viewshed = vs.generate_viewshed(grid, viewpoint_i, viewpoint_j, view_radius, CELL_WIDTH, viewpoint_height)
        todos = todos + viewshed
        output_file = 'VIEWSHED_' + str(viewpoint_i) + '_' + str(viewpoint_j) + '.png'
        vs.save_viewshed_image(viewshed, './VIEWSHEDS/' + output_file)
    vs.save_viewshed_image(todos, './VIEWSHEDS/todos.png')


# Lê o png do mapa de visibilidade
def read_viewshed(file):
    img = mpimg.imread(file)
    viewshed = img[:, :, 0]
    return viewshed


import AUXILIARES.generate_viewshed as vs


# Gera uma lista de tuplas de pontos de amostras para geração do dataset
def generate_sample_points(sampling_percentage):
    sections_n = 5
    sections_m = 5

    # Tamanho das divisões do mapa
    SECTION_ROWS = GRID_ROWS // sections_n
    SECTION_COLS = GRID_COLS // sections_m

    # Conjunto de pontos amostrados, ordenados da esquerda pra direta, de cima pra baixo
    P = []

    # Itera pelas divisões do mapa, da esquerda pra direita, de cima pra baixo
    for section_i in range(sections_n):
        for section_j in range(sections_m):
            section_points = []
            for p_i in range(SECTION_ROWS * section_i, SECTION_ROWS * section_i + SECTION_ROWS):
                for p_j in range(SECTION_COLS * section_j, SECTION_COLS * section_j + SECTION_COLS):
                    section_points.append((p_i, p_j))
            random.shuffle(section_points)
            sampling_size = int(len(section_points) * sampling_percentage)
            if sampling_size < 1:
                sampling_size = 1
            sample_points = section_points[0:sampling_size]
            P.extend(sample_points)
            section_points.clear()
    P.sort(key=lambda tup: (tup[1], tup[0]))
    return P


# Mapa heurístico da DNN com 6 entradas
def heuristic_dict1(g, model, goal):
    todos_vertices = g.get_vertices()

    dataset = []
    t1_start = time()
    for vertice_x in g:
        vertice_y = goal

        (x1, y1, alt1) = vertice_x.get_r3_coordinates() # current
        (x2, y2, alt2) = vertice_y.get_r3_coordinates() # goal

        # Ordena origem e destino da esquerda pra direita, de cima pra baixo (mesma ordem realizada no treinamento da DNN)
        if x2 < x1 or (x2 == x1 and y2 < y1):
            dataset.append([x2, y2, alt2, x1, y1, alt1])
        else:
            dataset.append([x1, y1, alt1, x2, y2, alt2])

    # Monta um dicionário com as predições da DNN
    dataset = np.array(dataset)
    with tf.device('/gpu:0'):
        predicoes = model.predict_on_batch(dataset)

    dict_heuristica = dict(zip(todos_vertices, predicoes))

    t1_stop = time()

    return dict_heuristica, t1_stop - t1_start


# Mapa heurístico da DNN com 9 entradas
def heuristic_dict2(g, model, observer, goal):
    todos_vertices = g.get_vertices()

    dataset = []
    t1_start = time()
    for vertice_x in g:
        vertice_y = goal

        (x1, y1, alt1) = vertice_x.get_r3_coordinates() # current
        (x2, y2, alt2) = vertice_y.get_r3_coordinates() # goal

        # Ordena origem e destino da esquerda pra direita, de cima pra baixo (mesma ordem realizada no treinamento da DNN)
        if x2 < x1 or (x2 == x1 and y2 < y1):
            dataset.append([observer[0], observer[1], observer[2], x2, y2, alt2, x1, y1, alt1])
        else:
            dataset.append([observer[0], observer[1], observer[2], x1, y1, alt1, x2, y2, alt2])

    # Monta um dicionário com as predições da DNN
    dataset = np.array(dataset)
    with tf.device('/gpu:0'):
        predicoes = model.predict_on_batch(dataset)

    dict_heuristica = dict(zip(todos_vertices, predicoes))

    t1_stop = time()

    return dict_heuristica, t1_stop - t1_start

def dict_dnn_heuristic1(start, goal):
    predicao = dnn_heuristic_dict1[start.get_id()][0]
    return predicao


def dict_dnn_heuristic2(start, goal):
    predicao = dnn_heuristic_dict2[start.get_id()][0]
    return predicao


def dnn_predict(start, goal, model, observer):
    (x1, y1, alt1) = start.get_r3_coordinates()  # current
    (x2, y2, alt2) = goal.get_r3_coordinates()  # goal

    # append array 2d
    if x2 < x1 or (x2 == x1 and y2 < y1):
        data = [observer[0], observer[1], observer[2], x2, y2, alt2, x1, y1, alt1]
    else:
        data = [observer[0], observer[1], observer[2], x1, y1, alt1, x2, y2, alt2]
    return model.predict(np.array([data]))


def observer_points(grid, n, m, r=10, spacing=4):  #divide o grid(n x m) em r x r regioes
    nr = (n)/r
    mr = (m)/r
    pontos = []
    for i in range(0,r): #0-9
        for j in range(0,r): #0-9
            regiao = np.array(grid[int(i * nr +spacing): int((i+1) * nr -spacing), int(j * mr +spacing) : int((j+1) * mr -spacing)])
            min = np.argmin(regiao)
            min = (min//regiao.shape[0] +spacing, min % regiao.shape[1]+spacing)
            max = np.argmax(regiao)
            max = (max//regiao.shape[0] +spacing, max % regiao.shape[1]+spacing)
            if ((i + j) % 2) == 0:
                min_coords = (int(i*nr + min[0]), int(j*mr + min[1]))
                pontos.append(min_coords)
            else:
                max_coords = (int(i*nr + max[0]), int(j*mr + max[1]))
                pontos.append(max_coords)
    return pontos

# Recupera o caminho percorrendo do fim para o inicio, contando os nodos visíveis
def count_visible_nodes(v, path, count_visible):
    if v.get_local_risk() > 0:
        count_visible += 1

    if v.previous:
        path.append(v.previous.get_id())
        count_visible = count_visible_nodes(v.previous, path, count_visible)

    return count_visible


def main():
    args = sys.argv
    filename = args[1] # recorte .tif do terreno
    model_name1 = 'model_1_10.hdf5'#'modelo_249_epocas.hdf5' # # modelo 1 de DNN treinada (só para características topográficas)
    #model_name2 = args[3] # modelo 2 de DNN treinada (para características topográficas e posição do observador)

    reduction_factor = 1 # Fator de redução de dimensão do mapa (2 -> mapa 400x400 abstraído em 200x200)

    # Lê o arquivo do MDE e cria o grid do mapa
    mde = Mde(filename, reduction_factor)

    print('Criando o grafo')
    # Cria o grafo a partir do grid do MDE
    g = Graph(mde)

    print('Gerando os viewsheds')
    # Coordenadas de cada observador
    # Gera as mesma coordenadas utilizadas na criação do dataset de treinamento
    viewpoints = observer_points(mde.grid, GRID_ROWS, GRID_COLS, 1)

    # Carrega os modelos das redes neurais treinadas
    #model1 = load_model(model_name1)
    model1 = load_model(model_name1)

    print('Iniciando')

    # Quantidade de caminhos para cada observador (100 X 1000)
    paths_per_map = 1000

    start_time = time()

    data_io_time_cost_r3 = io.StringIO()
    data_io_visited_cost_r3 = io.StringIO()

    data_io_visited = io.StringIO()
    data_io_opened = io.StringIO()
    data_io_visited2 = io.StringIO()
    data_io_opened2 = io.StringIO()
    data_io_visited3 = io.StringIO()
    data_io_opened3 = io.StringIO()
    data_io_visited4 = io.StringIO()
    data_io_opened4 = io.StringIO()

    data_io_time_cost_dnn1 = io.StringIO()
    data_io_visited_cost_dnn1 = io.StringIO()
    data_io_time_cost_dnn2 = io.StringIO()
    data_io_visited_cost_dnn2 = io.StringIO()
    data_io_comp = io.StringIO()
    data_io_comp2 = io.StringIO()
    data_io_comp3 = io.StringIO()
    data_io_comp4 = io.StringIO()
    data_io_all = io.StringIO()

    # cabecalho dos arquivos csv, separador utilizado é o ';'
    data_io_time_cost_r3.write("""y;x\n""")
    data_io_visited_cost_r3.write("""y;x\n""")

    #data_io_visited.write("""y;x\n""")
    #data_io_opened.write("""y;x\n""")

    #data_io_time_cost_dnn1.write("""y;x\n""")
   # data_io_visited_cost_dnn1.write("""y;x\n""")
    #data_io_time_cost_dnn2.write("""y;x\n""")
    #data_io_visited_cost_dnn2.write("""y;x\n""")
    data_io_comp.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
    data_io_comp2.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
    data_io_comp3.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
    data_io_comp4.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
    #data_io_all.write("""ox;oy;oh;x1;y1;h1;x2;y2;h2;c;d;v;nodos_visitados;total_time;time_search;time_h_map\n""")

    if not os.path.exists("./DADOS_RESULTADOS/"):
        os.makedirs("./DADOS_RESULTADOS/")

    #write_dataset_csv('./DADOS_RESULTADOS/time_cost_r3.csv', data_io_time_cost_r3)
   # write_dataset_csv('./DADOS_RESULTADOS/visited_cost_r3.csv', data_io_visited_cost_r3)
    #write_dataset_csv('./DADOS_RESULTADOS/time_cost_dnn1.csv', data_io_time_cost_dnn1)
   # write_dataset_csv('./DADOS_RESULTADOS/visited_cost_dnn1.csv', data_io_visited_cost_dnn1)
   # write_dataset_csv('./DADOS_RESULTADOS/time_cost_dnn2.csv', data_io_time_cost_dnn2)
   # write_dataset_csv('./DADOS_RESULTADOS/visited_cost_dnn2.csv', data_io_visited_cost_dnn2)
    write_dataset_csv('./DADOS_RESULTADOS/A_star.csv', data_io_comp)
    write_dataset_csv('./DADOS_RESULTADOS/A_star_mod.csv', data_io_comp2)
    write_dataset_csv('./DADOS_RESULTADOS/Theta_star.csv', data_io_comp3)
    write_dataset_csv('./DADOS_RESULTADOS/A_star_DNN.csv', data_io_comp4)
   # write_dataset_csv('./DADOS_RESULTADOS/all.csv', data_io_all)
    #write_dataset_csv('./DADOS_RESULTADOS/visited.csv',data_io_visited)
    teste= TestVars.test
    # Realiza o mesmo processo para cada observador
    print(len(viewpoints))
    for vp in viewpoints:
        observer = (int(vp[1] * CELL_WIDTH), int(vp[0] * CELL_HEIGHT), mde.grid[vp[0], vp[1]])  # Coordenadas do observador

        data_io_time_cost_r3 = io.StringIO()
        data_io_visited_cost_r3 = io.StringIO()
        data_io_time_cost_dnn1 = io.StringIO()
        data_io_visited_cost_dnn1 = io.StringIO()
        data_io_time_cost_dnn2 = io.StringIO()
        data_io_visited_cost_dnn2 = io.StringIO()
        data_io_comp = io.StringIO()
        data_io_comp2 = io.StringIO()
        data_io_comp3 = io.StringIO()
        data_io_comp4 = io.StringIO()
        data_io_all = io.StringIO()

        b = 0.5  # Fator de importância da segurança no cálculo do custo
        visibility_map_file = './VIEWSHEDS/VIEWSHED_' + str(vp[0]) + '_' + str(vp[1]) + '.png'

        #viewshed = read_viewshed(visibility_map_file)
        #viewshed = g.normalize_visibility(viewshed) # Normalização dos valores de visibilidade -> do intervalo [0,1] para o intervalo [min(edge), max(edge)]

        # Atribui a cada vértice o nível de visibilidade do viewshed
        
        #g.update_vertices_risk(viewshed)

        # ------------------------------------------------------------ #
        # Cria as combinações de pares de origem-destino
        sampling_rate = 0.125
        sample_coords = generate_sample_points(sampling_rate / 100)

        aux = 0
        combinations = []
        for coords in sample_coords:
            for coords2 in sample_coords[aux+1:]:
                combinations.append([coords, coords2])
            aux += 1

        random.shuffle(combinations)
        combinations = combinations[:paths_per_map]
        # ----------------------------------------------------------- #
        # Itera nos N pares de origem e destino
        for pair in combinations:
            src_coords = (128,192) #pair[0](128,192)
            dest_coords = (58,92) #pair[1](58,92)
            source_id = get_id_by_coords(src_coords[0], src_coords[1]) # Cada ponto da amostra é o ponto de origem da iteração
            source = g.get_vertex(source_id)
            #print("aaaa",source)
            dest_id = get_id_by_coords(dest_coords[0], dest_coords[1])
            dest = g.get_vertex(dest_id)
            global dnn_heuristic_dict1
            global dnn_heuristic_dict2
            
            #carrega a heuristica entre todos os pontos para o ponto alvo posteriormente é usada como consulta
            
            dnn_heuristic_dict1, h_map_time1 = heuristic_dict1(g, model1, dest)
            #dnn_heuristic_dict2, h_map_time2 = heuristic_dict2(g, model1,observer, dest)

            #4 casos:
            #1) A* simples, heurística r3
            heuristic = r3_heuristic
            t1 = time()
            distance1, count_visited1, count_open1, opened1, visited1, cost1 = astar(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
            t1 = time() - t1
            path1 = [dest.get_id()]
            print("custo do a*: ",cost1)
            print("nodos visitados: ",count_visited1)
            print("nodos abertos: ",count_open1)
            count_visible1 = count_visible_nodes(dest, path1, 0)
            path_len1 = len(path1)
            print("tempo de duração: ", t1)
            g.reset()

            print("terminou A*")

            #2)A* adaptado, heuristica r3 e caculo de Angulos
            heuristic = r3_heuristic
            
            t2 = time()
            distance2, count_visited2, count_open2, opened2, visited2, cost2 = astarmod(g, source, dest, b, heuristic)
            t2 = time() - t2
            print("custo do A* topografico: ",cost2)
            print("nodos visitados: ",count_visited2)
            print("nodos abertos: ",count_open2)
            
            path2 = [dest.get_id()]
            count_visible2 = count_visible_nodes(dest, path2, 0)
            path_len2 = len(path2)
            print("tempo de duração: ", t2)
            g.reset()

            #3)Theta* adaptado, heuristica r3 e calculo de angulo
            heuristic = heuristica_padrao
            
            t3 = time()
            distance3, count_visited3, count_open3, opened3, visited3, cost3 = theta_custo_diferente(g, source, dest, b, heuristic)
            t3 = time() - t3
            print("custo do theta: ",cost3)
            print("nodos visitados: ",count_visited3)
            print("nodos abertos: ",count_open3)
            
            path3 = [dest.get_id()]
            count_visible3 = count_visible_nodes(dest, path3, 0)
            path_len3 = len(path3)
            print("tempo de duração: ", t3)
            g.reset()
            
            
            data_io_time_cost_r3.write("""%s;%s\n""" % (t2, cost2))
            data_io_visited_cost_r3.write("""%s;%s\n""" % (count_visited2, cost2))
            
            #4) A* adaptado, heuristica DNN1 (treinado sem visibilidade)
            heuristic = dict_dnn_heuristic1
            t4 = time()
            distance4, count_visited4, count_open4, opened4, visited4, cost4 = astar(g, source, dest, b, heuristic)
            t4 = time() - t4
            path4 = [dest.get_id()]
            #count_visible4 = count_visible_nodes(dest, path4, 0)
            path_len4 = len(path4)
            
            print("custo do a* com dnn: ",cost4)
            print("nodos visitados: ",count_visited4)
            print("nodos abertos: ",count_open4)
            print("tempo de duração: ", t4)
            print("tempo do mapeamente heurístico: ", h_map_time1)
            g.reset()

            data_io_time_cost_dnn1.write("""%s;%s\n""" % (t4, cost4))
            data_io_visited_cost_dnn1.write("""%s;%s\n""" % (count_visited4, cost4))
            '''
            #4) A* adaptado, heuristica DNN2 (treinado com visibilidade)
            heuristic = dict_dnn_heuristic2
            t4 = time()
            distance4, safety4, count_visited4, count_open4, opened4, visited4, cost4 = safe_astar(g, source, dest, b,
                                                                                                   heuristic)
            path4 = [dest.get_id()]
            count_visible4 = count_visible_nodes(dest, path4, 0)
            path_len4 = len(path4)
            t4 = time() - t4 + h_map_time2
            g.reset()'''

            #data_io_time_cost_dnn2.write("""%s;%s\n""" % (t4, cost4))
            #data_io_visited_cost_dnn2.write("""%s;%s\n""" % (count_visited4, cost4))

            data_io_comp.write("""%s;%s;%s;%s\n""" %(cost1,t1,count_visited1,count_open1))
            data_io_comp2.write("""%s;%s;%s;%s\n""" %(cost2,t2,count_visited2,count_open2))
            data_io_comp3.write("""%s;%s;%s;%s\n""" %(cost3,t3,count_visited3,count_open3))
            data_io_comp4.write("""%s;%s;%s;%s\n""" %(cost4,t4+h_map_time1,count_visited4,count_open4))
            

            if teste:
                teste=False
                for i in range(len(opened1)):
                    data_io_opened.write("""%s\n"""%str((opened1[i])))
                for i in range(len(visited1)):
                    data_io_visited.write("""%s\n"""%str((visited1[i])))

                write_dataset_csv('./DADOS_RESULTADOS/visited.csv',data_io_visited)
                write_dataset_csv('./DADOS_RESULTADOS/opened.csv',data_io_opened)
                
                for i in range(len(opened2)):
                    data_io_opened2.write("""%s\n"""%str((opened2[i])))
                for i in range(len(visited2)):
                    data_io_visited2.write("""%s\n"""%str((visited2[i])))

                write_dataset_csv('./DADOS_RESULTADOS/visited2.csv',data_io_visited2)
                write_dataset_csv('./DADOS_RESULTADOS/opened2.csv',data_io_opened2)
                
                for i in range(len(opened3)):
                    data_io_opened3.write("""%s\n"""%str((opened3[i])))
                for i in range(len(visited3)):
                    data_io_visited3.write("""%s\n"""%str((visited3[i])))

                write_dataset_csv('./DADOS_RESULTADOS/visited3.csv',data_io_visited3)
                write_dataset_csv('./DADOS_RESULTADOS/opened3.csv',data_io_opened3)
                
                for i in range(len(opened4)):
                    data_io_opened4.write("""%s\n"""%str((opened4[i])))
                for i in range(len(visited4)):
                    data_io_visited4.write("""%s\n"""%str((visited4[i])))

                write_dataset_csv('./DADOS_RESULTADOS/visited4.csv',data_io_visited4)
                write_dataset_csv('./DADOS_RESULTADOS/opened4.csv',data_io_opened4)
                break


            

            #data_io_all.write("""%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n""" %
            #                  (observer[0], observer[1], observer[2], int(src_coords[1] * CELL_WIDTH), int(src_coords[0] * CELL_HEIGHT), mde.grid[src_coords[0], src_coords[1]],int(dest_coords[1] *CELL_WIDTH), int(dest_coords[0]*CELL_HEIGHT), mde.grid[dest_coords[0], dest_coords[1]], cost4, distance4,safety4,count_visited4,t4, float(t3-h_map_time2), h_map_time2))
        write_dataset_csv('./DADOS_RESULTADOS/A_star.csv', data_io_comp)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_mod.csv', data_io_comp2)
        write_dataset_csv('./DADOS_RESULTADOS/Theta_star.csv', data_io_comp3)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_DNN.csv', data_io_comp4)
        #write_dataset_csv('./DADOS_RESULTADOS/time_cost_r3.csv', data_io_time_cost_r3)
       # write_dataset_csv('./DADOS_RESULTADOS/visited_cost_r3.csv', data_io_visited_cost_r3)
        #write_dataset_csv('./DADOS_RESULTADOS/time_cost_dnn1.csv', data_io_time_cost_dnn1)
        #write_dataset_csv('./DADOS_RESULTADOS/visited_cost_dnn1.csv', data_io_visited_cost_dnn1)
        #write_dataset_csv('./DADOS_RESULTADOS/time_cost_dnn2.csv', data_io_time_cost_dnn2)
        #write_dataset_csv('./DADOS_RESULTADOS/visited_cost_dnn2.csv', data_io_visited_cost_dnn2)
        #write_dataset_csv('./DADOS_RESULTADOS/comp.csv', data_io_comp)
        #write_dataset_csv('./DADOS_RESULTADOS/all.csv', data_io_all)

        print('Tempo: ' + str(time() - start_time) + ' segundos')
        break

if __name__ == '__main__':
    main()
