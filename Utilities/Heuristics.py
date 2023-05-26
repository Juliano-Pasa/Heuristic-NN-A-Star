import io
import shutil
import random
import sys
import math
import numpy as np
from time import time

def r2_distance(x1, x2, y1, y2):
    return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

def r3_distance(x1, x2, y1, y2, z1, z2):
    return np.sqrt((r2_distance(x1,x2,y1,y2) ** 2) + ((z1 - z2) ** 2))

# Heuristica distancia Euclidiana no R3
def r3_heuristic(start, goal):
    x1, y1 = start.get_r2_coordinates()
    x2, y2 = goal.get_r2_coordinates()
    z1 = start.get_elevation()
    z2 = goal.get_elevation()

    dst = r3_distance(x1, x2, y1, y2, z1, z2)

    return dst

def heuristica_padrao(start, goal):
    x1, y1 = start.get_r2_coordinates()
    x2, y2 = goal.get_r2_coordinates()

    return r2_distance(x1,x2,y1,y2)

def calcula_hipotenusa(vertex1,vertex2):
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

def heuristic_dict1_multiplos_mapas_frozen_graph(g, goal, map_id, session, output_tensor):
    t1_start = time()
    todos_vertices = g.get_vertices()
    vertice_y = goal
    (x2, y2, alt2) = vertice_y.get_r3_coordinates() # goal

    dataset = []
    (x2, y2, alt2) = goal.get_r3_coordinates() # goal
    for vertice_x in g:
        (x1, y1, alt1) = vertice_x.get_r3_coordinates() # current

        # Ordena origem e destino da esquerda pra direita, de cima pra baixo (mesma ordem realizada no treinamento da DNN)
        if x2 < x1 or (x2 == x1 and y2 < y1):
            dataset.append([map_id, x2, y2, alt2, x1, y1, alt1])
        else:
            dataset.append([map_id, x1, y1, alt1, x2, y2, alt2])

    # Monta um dicionário com as predições da DNN
    dataset = np.array(dataset)
    result = session.run(output_tensor, {'x:0': dataset})
    #with tf.device('/gpu:0'):
    dict_heuristica = dict(zip(todos_vertices, result))
    t1_stop = time()

    return dict_heuristica, t1_stop - t1_start