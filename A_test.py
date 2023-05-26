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

from Utilities.Graph import *
from config_variables import *
from Utilities.Heuristics import *
from Utilities.A_star import astar
from Utilities.Bia_star import biastar

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

def get_id_by_coords(i, j):
    return i * MDEVars.GRID_COLS + j

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
        viewshed = vs.generate_viewshed(grid, viewpoint_i, viewpoint_j, view_radius, MDEVars.CELL_WIDTH, viewpoint_height)
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
    
    SECTION_ROWS = MDEVars.GRID_ROWS // sections_n
    SECTION_COLS = MDEVars.GRID_COLS // sections_m
    
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


def main():
    global session
    global output_tensor

    session_abs = tf1.InteractiveSession()
    frozen_graph="./frozen_models/frozen_graph_abs_cost.pb"

    with tf1.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())

    session_abs.graph.as_default()
    tf1.import_graph_def(graph_def)    
    output_tensor_abs = session_abs.graph.get_tensor_by_name("Identity:0")

    session_cf = tf1.InteractiveSession()
    frozen_graph_cf="./frozen_models/frozen_graph_cf.pb"

    with tf1.gfile.GFile(frozen_graph_cf, "rb") as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())

    session_cf.graph.as_default()
    tf1.import_graph_def(graph_def)    
    output_tensor_cf = session_cf.graph.get_tensor_by_name("Identity:0")

    print('Iniciando')
    
    maps = GenerateVars.maps
        
    for mp in maps[:1]:
        i=0
        global map_id
        map_id = mp.id_map

        map_dir = GenerateVars.maps_dir
        
        map_path = map_dir + mp.filename
        print('Criando o grafo')
        mde = Mde(map_path, mp.reduction_factor)
        print(mp.filename)
        print(mp.id_map)
        g = Graph(mde)
        print('Gerando os viewsheds')

        paths_per_map = 4800

        start_time = time()

        data_io_comp1 = io.StringIO()
        data_io_comp2 = io.StringIO()
        data_io_comp3 = io.StringIO()
        data_io_comp4 = io.StringIO()

        data_io_comp1.write("""custo;tempo;nodos_visitados;nodos_abertos;t_mapa_heuristico\n""")
        data_io_comp2.write("""custo;tempo;nodos_visitados;nodos_abertos;t_mapa_heuristico\n""")
        data_io_comp3.write("""custo;tempo;nodos_visitados;nodos_abertos;t_mapa_heuristico\n""")
        data_io_comp4.write("""custo;tempo;nodos_visitados;nodos_abertos;t_mapa_heuristico\n""")
        
        if not os.path.exists("./DADOS_RESULTADOS/"):
            os.makedirs("./DADOS_RESULTADOS/")

        write_dataset_csv('./DADOS_RESULTADOS/A_star_FROZEN_ABS'+str(mp.id_map)+'.csv', data_io_comp1)
        write_dataset_csv('./DADOS_RESULTADOS/BiA_star_FROZEN_ABS'+str(mp.id_map)+'.csv', data_io_comp2)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_FROZEN_CF'+str(mp.id_map)+'.csv', data_io_comp3)
        write_dataset_csv('./DADOS_RESULTADOS/BiA_star_FROZEN_CF'+str(mp.id_map)+'.csv', data_io_comp4)

        data_io_comp1 = io.StringIO()
        data_io_comp2 = io.StringIO()
        data_io_comp3 = io.StringIO()
        data_io_comp4 = io.StringIO()

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
        for pair in combinations[:50]:
            src_coords = pair[0]
            dest_coords = pair[1]
            source_id = get_id_by_coords(src_coords[0], src_coords[1])
            source = g.get_vertex(source_id)
            dest_id = get_id_by_coords(dest_coords[0], dest_coords[1])
            dest = g.get_vertex(dest_id)
            
            b=0
            heuristicDict, heuristicTime = heuristic_dict1_multiplos_mapas_frozen_graph(g, dest, map_id, session_abs, output_tensor_abs)
            
            t1 = time()
            opened1, count_visited1, count_open1, visited1, cost1 = astar(g, source, dest, b, heuristicDict) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
            t1 = time() - t1 + heuristicTime
            g.reset()

            t2 = time()
            opened2, count_visited2, count_open2, visited2, cost2 = biastar(g, source, dest, b, heuristicDict, heuristicDict) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
            t2 = time() - t2 + heuristicTime
            g.reset()

            heuristicDict, heuristicTime = heuristic_dict1_multiplos_mapas_frozen_graph(g, dest, map_id, session_cf, output_tensor_cf)

            t3 = time()
            opened3, count_visited3, count_open3, visited3, cost3 = astar(g, source, dest, b, heuristicDict) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
            t3 = time() - t3 + heuristicTime         
            g.reset()
            
            t4 = time()
            opened4, count_visited4, count_open4, visited4, cost4 = biastar(g, source, dest, b, heuristicDict, heuristicDict) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
            t4 = time() - t4 + heuristicTime
            g.reset()

            data_io_comp1.write("""%s;%s;%s;%s\n""" %(cost1,t1,count_visited1,count_open1))
            data_io_comp2.write("""%s;%s;%s;%s\n""" %(cost2,t2,count_visited2,count_open2))
            data_io_comp3.write("""%s;%s;%s;%s\n""" %(cost3,t3,count_visited3,count_open3))
            data_io_comp4.write("""%s;%s;%s;%s\n""" %(cost4,t4,count_visited4,count_open4))
            i+=1
            print(i)
            
        write_dataset_csv('./DADOS_RESULTADOS/A_star_FROZEN_ABS'+str(mp.id_map)+'.csv', data_io_comp1)
        write_dataset_csv('./DADOS_RESULTADOS/BiA_star_FROZEN_ABS'+str(mp.id_map)+'.csv', data_io_comp2)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_FROZEN_CF'+str(mp.id_map)+'.csv', data_io_comp3)
        write_dataset_csv('./DADOS_RESULTADOS/BiA_star_FROZEN_CF'+str(mp.id_map)+'.csv', data_io_comp4)

        print('Tempo: ' + str(time() - start_time) + ' segundos')

if __name__ == '__main__':
    main()
