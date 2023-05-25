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