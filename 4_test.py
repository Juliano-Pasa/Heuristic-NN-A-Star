import io
import shutil
import random
import sys
from matplotlib.ticker import MultipleLocator
import numpy as np
import math
import csv
import heapq
import os
import glob
from config_variables import TestVars, TestCase, GenerateVars
import multiprocessing

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from time import time
from time import process_time
import pickle

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
import tensorflow.compat.v1 as tf1


ANGULO_MAX =30


def r2_distance(x1, x2, y1, y2):
    return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


def r3_distance(x1, x2, y1, y2, z1, z2):
    return np.sqrt((r2_distance(x1,x2,y1,y2) ** 2) + ((z1 - z2) ** 2))

def calcula_angulo(vert,vert1):
    if vert is None or vert1 is None:
        return math.inf
    if vert == vert1:
        return math.inf
    
    
    altura = abs(vert.get_elevation()-vert1.get_elevation())
    hipotenusa = r3_distance(vert.get_x(),vert1.get_x(),vert.get_y(),vert1.get_y(),vert.get_elevation(),vert1.get_elevation())
    seno = altura/hipotenusa
    
    
    return math.degrees(math.asin(seno))


class Mde:
    
    import rasterio

    
    
    
    def __init__(self, fp, reduction_factor):
        self.dataset = self.rasterio.open(fp)
        self.band1 = self.dataset.read(1)
        self.pixel_resolution = round(self.dataset.transform[0] * 108000)
        
        print("\n\n\n\n\nmetadados:", self.pixel_resolution)

        print("\n\n\n\n\nmetadados:",self.dataset.height)
        self.h_limit = self.dataset.height
        self.w_limit = self.dataset.width

        
        self.generate_grid(reduction_factor)

        self.cell_size = self.pixel_resolution * reduction_factor
        global CELL_HEIGHT, CELL_WIDTH, GRID_ROWS, GRID_COLS, GRID_HEIGHT, GRID_WIDTH
        CELL_HEIGHT = self.pixel_resolution * reduction_factor
        CELL_WIDTH = self.pixel_resolution * reduction_factor
        GRID_COLS = self.grid.shape[0]
        GRID_ROWS = self.grid.shape[1]
        GRID_WIDTH = CELL_WIDTH * GRID_COLS
        GRID_HEIGHT = CELL_HEIGHT * GRID_ROWS

    
    
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
        self.angles = {}
        self.distance = 99999999    
        self.risk = 99999999    
        self.previous = None
        self.visited = False
        self.visitedReverse = False
        self.j = self.get_j()
        self.i = self.get_i()
        self.computed = False

    def __str__(self):
        return str(self.id) + ' elevation: ' + str(self.elevation) + ' coord: ' + str(self.get_r2_coordinates()) + ' edges: ' + str([x for x in self.edges.keys()]) + str([x for x in self.edges.values()])

    def get_previous(self):
        return self.previous

    def add_edge(self, node_id, edge_weight, g):
        self.edges[node_id] = edge_weight
        edge = g.get_vertex(node_id)
        self.angles[node_id] = calcula_angulo(self, edge)
        
    def get_visited(self):
        return self.visited
    
    def set_visited_reverse(self, visit):
        self.visitedReverse = visit

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
        return self.j * CELL_WIDTH

    def get_y(self):
        return self.i * CELL_HEIGHT

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
    
    def get_edge_angle(self, vertex_id):
        if self.get_id() == vertex_id:
            return math.inf
        if (self.angles[vertex_id] is None):
            return None
        return self.angles[vertex_id]

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

    def reset(self):
        self.distance = 99999999
        self.risk = 99999999
        self.previous = None
        self.visited = False
        self.visitedReverse = False
        self.computed = False

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
        self.vertices = {}  
        self.max_edge = 0.0
        self.min_edge = float("inf")
        self.create_vertices(mde)   
        self.generate_edges(True)  

    def __iter__(self):
        return iter(self.vertices.values())

    
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
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),  
                                         vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight, self)


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
                    vertex.add_edge(vertex2_id, weight, self)


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
                    vertex.add_edge(vertex2_id, weight, self)

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
                    vertex.add_edge(vertex2_id, weight, self)

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
                        vertex.add_edge(vertex2_id, weight, self)

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
                        vertex.add_edge(vertex2_id, weight, self)

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
                        vertex.add_edge(vertex2_id, weight, self)

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
                        vertex.add_edge(vertex2_id, weight, self)

    
    def normalize_visibility(self, visibility):
        return visibility * (self.max_edge - self.min_edge)


def shortest(v, path):
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return



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

    
def calcula_hipotenusa(vertex1,vertex2,g):
    
    distancia = vertex2.get_edge_weight(vertex1.get_id())
    altura = abs(vertex2.get_elevation()-vertex1.get_elevation())
    hipotenusa = math.sqrt(distancia**2+altura**2)    
    return hipotenusa

def calcula_custo_theta(vert1,vert2,multiplicador):
    
    if vert1 is None or vert2 is None:
        return False, math.inf
    return True, r3_heuristic(vert1,vert2)*multiplicador

def calcula_y(x,m,n):
    return m*x-n

def calcula_x(y,m,n):
    return (y+n)/m

def line_of_sight1(s,s1,g):
    if s is None or s1 is None:
        return False, 10
    x0,y0 = s.get_coordinates()
    
    x1,y1 = s1.get_coordinates()
    
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
    
    
    x_s,y_s = s.get_coordinates()
    x_s1,y_s1 = s1.get_coordinates()



    
    
    

    
    if x_s1-x_s==0:
        m=0
    else:
        m = (y_s1-y_s)/(x_s1-x_s)

    
    n = m*x_s - y_s

    
    

    

    
    


    
    
    
    
    
    
    
    
    cost=0
    if dx >= dy:
        
        while x0 != x1:
            f = f + dy
            
            
            
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


            
            
            if f>= dx:
                if (vert_src.get_edge_angle(vert_id))>ANGULO_MAX:
                    
                    return False, math.inf
                y0 = y0 + sy
                f = f - dx

            if f!=0 and (vert_src.get_edge_angle(vert_id))>ANGULO_MAX:
                return False, math.inf
            new_vert1= get_id_by_coords(x0 + int((((sx-1)/2))),y0)
            
            new_vert2= get_id_by_coords(x0 + int((((sx-1)/2))),y0 - 1)
            
            if dy==0 and (vert_src.get_edge_angle(new_vert1))>ANGULO_MAX and (vert_src.get_edge_angle(new_vert2))>ANGULO_MAX:
                return False, math.inf
            
            x0 = x0 + sx



            
    else:
        
        while y0 != y1:
            f = f + dx
            idinicial=get_id_by_coords(x0,y0)
            
            
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

            
            if f >= dy:
                vert = g.get_vertex_by_coords(x0,y0)
                
                if (vert_src.get_edge_angle(vert_id))>ANGULO_MAX:
                    return False, math.inf
                
                x0 = x0 + sx
                f = f - dy

            if f != 0 and (vert_src.get_edge_angle(vert_id))>ANGULO_MAX:
                return False, math.inf
            
            new_vert1= get_id_by_coords(x0,y0 + int((sy-1)/2))
            
            new_vert2= get_id_by_coords(x0 - 1,y0 + int((sy-1)/2))
            
            if dx == 0 and (vert_src.get_edge_angle(new_vert1))>ANGULO_MAX and (vert_src.get_edge_angle(new_vert2))>ANGULO_MAX:
                vert = g.get_vertex_by_coords(x0,y0)
                
                return False, math.inf
            y0 = y0 + sy
            
            
            
            
            
            
    return True, cost


def CalculateCostNonUniform(child,current,grid):
    g2=math.inf
    linha=False
    
    g1 = current.get_distance() + calcula_hipotenusa(current,child,grid)
    
    
    
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

def backtracking(final,start):
    path=[]
    while final.get_id() != start.get_id():
        path.append(final.get_coordinates())
        
        final = final.get_previous()
    path.append(final.get_coordinates())
    return path

def calculateClusterId(current):
    i = current.get_i()
    j = current.get_j()

    row = i // clusterDivisor
    col = j // clusterDivisor

    return row * 6 + col

def astar(g, start, goal, v_weight, heuristic):
    opened = []
    visited = []
    heuristic_time = 0

    start.set_risk(start.get_local_risk())
    start.set_distance(0)
    
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
            distance = current.get_distance()
            path=[]
            
            return visited, len(path), count_open, path, distance

        current.set_visited(True)
        count_visited = count_visited + 1

        for next_id in current.get_neighbors():
            next = g.get_vertex(next_id)
            new_dist = current.get_distance() + current.get_edge_weight(next_id)
            new_risk = current.get_risk() + next.get_local_risk()

            if new_dist < next.get_distance():
                next.set_previous(current)
                next.set_distance(new_dist)
                next.set_risk(new_risk)

                t1 = time()
                hscore = new_dist + heuristic(next, goal)
                heuristic_time += time() - t1

                if not next.visited:
                    heapq.heappush(unvisited_queue, (hscore, next))
                    visitedClusters[calculateClusterId(next)] = True
                    opened.append(next.get_coordinates())

def get_visited_coord(graph, visited_vertices):
    path = []
    for vertex_id in visited_vertices[::-1]:
        path.append(graph.get_vertex(vertex_id).get_coordinates())
    return path



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

def write_dataset_test_csv(filename, data_io):
    with open(filename, 'w') as file:
        data_io.seek(0)
        shutil.copyfileobj(data_io, file)



def save_viewsheds(grid, viewpoints, view_radius, viewpoint_height):
    todos = np.zeros((grid.shape[0], grid.shape[1]))
    for viewpoint_i, viewpoint_j in viewpoints:
        viewshed = vs.generate_viewshed(grid, viewpoint_i, viewpoint_j, view_radius, CELL_WIDTH, viewpoint_height)
        todos = todos + viewshed
        output_file = 'VIEWSHED_' + str(viewpoint_i) + '_' + str(viewpoint_j) + '.png'
        vs.save_viewshed_image(viewshed, './VIEWSHEDS/' + output_file)
    vs.save_viewshed_image(todos, './VIEWSHEDS/todos.png')



def read_viewshed(file):
    img = mpimg.imread(file)
    viewshed = img[:, :, 0]
    return viewshed


import AUXILIARES.generate_viewshed as vs



def generate_sample_points(sampling_percentage):
    sections_n = 5
    sections_m = 5

    
    SECTION_ROWS = GRID_ROWS // sections_n
    SECTION_COLS = GRID_COLS // sections_m

    
    P = []

    
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

def heuristic_dict1_multiplos_mapas_iterative(g, session, output_tensor, current, goal, map_id):
    dataset = []
    t1_start = time()
    (x2, y2, alt2) = goal.get_r3_coordinates() 

    ids = []
    selected = []
    current_i = current.get_i()
    current_j = current.get_j()

    expanding_factor = 24

    for i in range(-expanding_factor, expanding_factor+1):
        ids.append((current_i + i) * GRID_COLS + current_j)
        for j in range(1, expanding_factor+1):
            ids.append((current_i + i) * GRID_COLS + (current_j + j))
            ids.append((current_i + i) * GRID_COLS + (current_j - j))

    maximum = GRID_COLS * GRID_ROWS

    for id in ids:
        if id < 0 or id > maximum:
            continue

        vertex = g.get_vertex(id)
        if vertex is None:
            continue
        if vertex.computed:
            continue

        vertex.computed = True
        selected.append(id)
        (x1, y1, alt1) = vertex.get_r3_coordinates() 

        
        if x2 < x1 or (x2 == x1 and y2 < y1):
            dataset.append([map_id, x2, y2, alt2, x1, y1, alt1])
        else:
            dataset.append([map_id, x1, y1, alt1, x2, y2, alt2])
        
    
    dataset = np.array(dataset)
    
    predicoes = session.run(output_tensor, {'x:0': dataset})

    dict_heuristica = dict(zip(selected, predicoes))

    t1_stop = time()

    return dict_heuristica, t1_stop - t1_start

def dict_dnn_iterative_abs(start, goal):
    value = 0
    global dnn_heuristic_iterative

    try:
        value = dnn_heuristic_iterative[start.get_id()][0]

    except KeyError:
        aux, _ = heuristic_dict1_multiplos_mapas_iterative(g, session_abs, output_tensor_abs, start, goal, mapId)
        dnn_heuristic_iterative.update(aux)
        value = dnn_heuristic_iterative[start.get_id()][0]

    return value


def count_visible_nodes(v, path, count_visible):
    if v.get_local_risk() > 0:
        count_visible += 1

    if v.previous:
        path.append(v.previous.get_id())
        count_visible = count_visible_nodes(v.previous, path, count_visible)

    return count_visible

def generateClusterHeader(totalClusters):
    aux = ""
    for i in range(totalClusters):
        aux += f",c{i}"
    
    return aux

def writeAllClusters():
    aux = ""
    for _, value in visitedClusters.items():
        if value:
            aux += ",1"
        else:
            aux += ",0"
    
    return aux

def writeSumOfClusters():
    i = 0
    total = 0
    for _, value in visitedClusters.items():
        if value:
            total += pow(2, i)
        i += 1
    
    return total

def writeClusterClustering():
    clusters = {0:0, 1:0, 2:0, 3:0}
    clustersTotalVisits = {0:0, 1:0, 2:0, 3:0}

    for key, value in visitedClusters.items():
        clusterI = (key // 6) // 3
        clusterJ = (key % 6) // 3

        clusterId = clusterI * 2 + clusterJ

        if value:
            clusters[clusterId] += pow(2, 2*clustersTotalVisits[clusterId])
        clustersTotalVisits[clusterId] += 1

    aux = ""
    for _, value in clusters.items():
        aux += f",{value}"
    
    return aux

def ResetDictionary(totalClusters):
    visited = {}
    for i in range(totalClusters):
        visited[i] = False

    return visited

def main():
    global session_abs
    global output_tensor_abs
    global g
    global mapId
    global visitedClusters
    global clusterDivisor

    clusterDivisions = 6
    clusterDivisor = 300 / clusterDivisions
    visitedClusters = ResetDictionary(clusterDivisions**2)

    session_abs = tf1.InteractiveSession()
    frozen_graph="./frozen_models/frozen_graph_abs_cost.pb"

    with tf1.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())

    session_abs.graph.as_default()
    tf1.import_graph_def(graph_def)    
    output_tensor_abs = session_abs.graph.get_tensor_by_name("Identity:0")
    
    print('Iniciando')
    
    if(GenerateVars.use_viewpoints):
        maps = [GenerateVars.vps_map]
    else:
        maps = GenerateVars.maps
        
    # for mp in maps[:1]:
    for iteration in range(100):
        mp = maps[0]
        i=0
        global map_id
        map_id = mp.id_map
        if(GenerateVars.use_viewpoints):
            map_dir = GenerateVars.vps_map_dir
        else:
            map_dir = GenerateVars.maps_dir
        
        mapId = mp.id_map
        map_path = map_dir + mp.filename
        print('Criando o grafo')
        mde = Mde(map_path, mp.reduction_factor)
        print(mp.filename)
        print(mp.id_map)
        g = Graph(mde)
        print('Gerando os viewsheds')

        paths_per_map = 10000

        start_time = time()
        data_io_comp1 = io.StringIO()
        data_io_comp2 = io.StringIO()
        data_io_comp3 = io.StringIO()
        
        data_io_comp1.write("""x0,y0,x1,y1,mapId%s\n""" %(generateClusterHeader(clusterDivisions*clusterDivisions)))
        data_io_comp2.write("""x0,y0,x1,y1,mapId,c\n""")
        data_io_comp3.write("""x0,y0,x1,y1,mapId,c0,c1,c2,c3\n""")

        if not os.path.exists("./DADOS_RESULTADOS/"):
            os.makedirs("./DADOS_RESULTADOS/")

        b = 0.5              

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
        for pair in combinations:
            src_coords = pair[0] 
            dest_coords = pair[1] 
            source_id = get_id_by_coords(src_coords[0], src_coords[1]) 
            source = g.get_vertex(source_id)
            dest_id = get_id_by_coords(dest_coords[0], dest_coords[1])
            dest = g.get_vertex(dest_id)
            
            global dnn_heuristic_iterative
            dnn_heuristic_iterative = {}

            heuristic = dict_dnn_iterative_abs
            t = time()
            opened, count_visited, count_open, visited, cost = astar(g, source, dest, b, heuristic) 
            t = time() - t

            data_io_comp1.write("""%s,%s,%s,%s,%s%s\n""" %(src_coords[0], src_coords[1], dest_coords[0], dest_coords[1], map_id, writeAllClusters()))
            data_io_comp2.write("""%s,%s,%s,%s,%s,%s\n""" %(src_coords[0], src_coords[1], dest_coords[0], dest_coords[1], map_id, writeSumOfClusters()))
            data_io_comp3.write("""%s,%s,%s,%s,%s%s\n""" %(src_coords[0], src_coords[1], dest_coords[0], dest_coords[1], map_id, writeClusterClustering()))
            i+=1

            visitedClusters = ResetDictionary(clusterDivisions**2)

            g.reset()

        write_dataset_csv('./DADOS_RESULTADOS/binaryClustersC'+str(iteration)+'.csv', data_io_comp1)
        write_dataset_csv('./DADOS_RESULTADOS/sumOfClustersC'+str(iteration)+'.csv', data_io_comp2)
        write_dataset_csv('./DADOS_RESULTADOS/clusterClusteringC'+str(iteration)+'.csv', data_io_comp3)

        print('Tempo: ' + str(time() - start_time) + ' segundos')

if __name__ == '__main__':
    main()
