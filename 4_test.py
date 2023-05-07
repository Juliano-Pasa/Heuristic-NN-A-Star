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
import multiprocessing

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
import tensorflow.compat.v1 as tf1

from config_variables import TestVars, TestCase, GenerateVars
import graph_manager
import heuristics
import a_star
import bia_star

ANGULO_MAX =30



# Recupera o caminho percorrendo do fim para o inicio
def shortest(v, path):
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return



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

def backtracking(final,start):
    path=[]
    while final.get_id() != start.get_id():
        path.append(final.get_coordinates())
        #print("aqui",final.get_coordinates())
        final = final.get_previous()
    path.append(final.get_coordinates())
    return path


                    
def ida(g, start, goal, v_weight, heuristic):
    opened = []
    visited = []

    visibility_weight = v_weight
    
    #print(start)
    #print(goal)

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + heuristic(start, goal)
    bound = hscore

    unvisited_queue = [(hscore, start)]
    heapq.heapify(unvisited_queue)

    count_visited = 0
    count_open = 1

    threshold = heuristic(start, goal)
    opened.append(start.get_coordinates())
    found = False
    while len(unvisited_queue)>0:
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        print(len(unvisited_queue))
        flag, node =search(g,current, goal, unvisited_queue,opened,heuristic,bound,visited,count_visited,threshold)
        if flag:
                #print("ÇOCORRO DEUS\n\n\n\n\n",visited)
                #break
                distance = current.get_distance()
                path=[]
                #print(current)
                #print(start)
                #print("VOLTEI KKKKKKKKKKKKKKKKKKK1")
                path=backtracking(current,start)
                #print("VOLTEI KKKKKKKKKKKKKKKKKKK2")
                
                #closed_nodes = list(map(lambda v: v.get_coordinates(), visited))
                
                print("VOLTEI KKKKKKKKKKKKKKKKKKK")
                return visited, len(path), count_open, path, node.get_distance()
        else:
            print("entrei nessa condicao")
            threshold = node.get_distance()
    print("fudeo lek kek")

global N_CHAMADAS
def search(g,current, goal, unvisited_queue, opened, heuristic, bound,visited,count_visited,threshold):
    global N_CHAMADAS
    N_CHAMADAS = N_CHAMADAS + 1
    print(N_CHAMADAS)
    count_open = 0 ##HENRIQUE VER -> Deveria estar aqui?
    
    if len(unvisited_queue)>1:
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
    
    if current == goal:
        #print(current)
        #print(goal)
        return True, current
    for next_id in current.get_neighbors():
        
        
        next = g.get_vertex(next_id)
        if next == goal:
            #print(next)
            #print(goal)
            return True, next
        if current.get_distance()>threshold and current.get_distance()<math.inf:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            print(next)
            return False, next
        current.set_visited(True)
        count_visited = count_visited + 1
        visited.append(current.get_coordinates())
        
        new_dist = current.get_distance() + current.get_edge_weight(next_id)
        new_risk = current.get_risk() + next.get_local_risk()
        if new_dist < next.get_distance():
            next.set_previous(current)
            next.set_distance(new_dist)
            next.set_risk(new_risk)
        
            if not next.visited:
                hscore = new_dist + heuristic(next, goal)
                heapq.heappush(unvisited_queue, (hscore, next))
                count_open = count_open + 1
                opened.append(next.get_coordinates())    
                
                if search(g,next, goal, unvisited_queue, opened, heuristic, bound, visited, count_visited, threshold):##HENRIQUE VER -> Ajustei o visited e o countvisited
                    #print("chegueiaqui")
                    return True, next
    
    return False
    ##HENRIQUE VER -> Python estorou o máximo de recursão.

            

#Backtrack do BiA*
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

def biastar_DNN_CF(g, start, goal, v_weight, heuristicCF, heuristicCF1):
    visited = [] #visitados e abertos
    heapq.heapify(visited)
    visitedReverse = []
    heapq.heapify(visitedReverse)

    # Seta distância inicial para 0 e o risco inicial para o risco do ponto de partida
    start.set_risk(start.get_local_risk())
    start.set_distance(0)
    goal.set_distance(0)

    # Calcula custo = w * risco + distancia + heursítica_r3
    hscore = start.get_distance() + r3_heuristic(start,goal)*heuristicCF(start, goal)
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
            new_dist = current.get_distance() + current.get_edge_weight(next_id) 

            if next.visitedReverse:
                openedR, count_visitedR, count_openR, visitedR, costR = generatePath(current, next, start, goal, visited, visitedReverse, v_weight, count_open, heuristicCF,g)
                return openedR, count_visitedR, count_openR, visitedR, costR

            if next.has_parent():
                if next.get_previous().visitedReverse:
                    continue

            if new_dist < next.get_distance():
                next.set_previous(current)
                next.set_distance(new_dist)

                hscore = new_dist + r3_heuristic(next,goal)*heuristicCF(next, goal)

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
                openedR, count_visitedR, count_openR, visitedR, costR = generatePath(next, current, start, goal, visited, visitedReverse, v_weight, count_open, heuristicCF1,g)
                return openedR, count_visitedR, count_openR, visitedR, costR

            if next.has_parent():
                if next.get_previous().visited:
                    continue

            if new_dist < next.get_distance():
                next.set_previous(current)
                next.set_distance(new_dist)

                hscore = new_dist + r3_heuristic(next,goal)*heuristicCF1(next, goal)

                if not next.visitedReverse:
                    heapq.heappush(unvisited_queue_reverse, (hscore, next))
                    count_open = count_open + 1



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
    #filename = args[1] # recorte .tif do terreno
    '''model_1_10.hdf5''' 
    model_name1 = 'model_32_20230227-164136_checkpoint_19_0.0147.hdf5'#'modelo_249_epocas.hdf5' # # modelo 1 de DNN treinada (só para características topográficas)
    model_name2 = 'model_32_20230220-165452_checkpoint_94_0.2484.hdf5' # modelo 2 de DNN treinada (para características topográficas e posição do observador)

    global session_abs
    global output_tensor_abs

    session_abs = tf1.InteractiveSession()
    frozen_graph="./frozen_models/frozen_graph_abs_cost.pb"

    with tf1.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())

    session_abs.graph.as_default()
    tf1.import_graph_def(graph_def)    
    output_tensor_abs = session_abs.graph.get_tensor_by_name("Identity:0")

    global session_cf
    global output_tensor_cf

    session_cf = tf1.InteractiveSession()
    frozen_graph_cf="./frozen_models/frozen_graph_cf.pb"

    with tf1.gfile.GFile(frozen_graph_cf, "rb") as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())

    session_cf.graph.as_default()
    tf1.import_graph_def(graph_def)    
    output_tensor_cf = session_cf.graph.get_tensor_by_name("Identity:0")
    
    reduction_factor = 1 # Fator de redução de dimensão do mapa (2 -> mapa 400x400 abstraído em 200x200)

    # Lê o arquivo do MDE e cria o grid do mapa

    # Cria o grafo a partir do grid do MDE

    # Coordenadas de cada observador
    # Gera as mesma coordenadas utilizadas na criação do dataset de treinamento

    # Carrega os modelos das redes neurais treinadas
    #model1 = load_model(model_name1)
    model_CF = load_model(model_name1)
    model_ABS = load_model(model_name2)
    global g
    global mapId

    #with tf.device('/gpu:0'):
    
    print('Iniciando')
    
    if(GenerateVars.use_viewpoints):
        maps = [GenerateVars.vps_map]
    else:
        maps = GenerateVars.maps
        
    for mp in maps:
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
        viewpoints = observer_points(mde.grid, GRID_ROWS, GRID_COLS, 1)
        print('Gerando os viewsheds')
    # Quantidade de caminhos para cada observador (100 X 1000)
        paths_per_map = 5000

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
        data_io_visited5 = io.StringIO()
        data_io_opened5 = io.StringIO()

        data_io_time_cost_dnn1 = io.StringIO()
        data_io_visited_cost_dnn1 = io.StringIO()
        data_io_time_cost_dnn2 = io.StringIO()
        data_io_visited_cost_dnn2 = io.StringIO()
        data_io_comp = io.StringIO()
        data_io_comp2 = io.StringIO()
        data_io_comp3 = io.StringIO()
        data_io_comp4 = io.StringIO()
        data_io_comp5 = io.StringIO()
        data_io_comp6 = io.StringIO()
        data_io_comp7 = io.StringIO()
        data_io_comp8 = io.StringIO()
        data_io_comp9 = io.StringIO()
        data_io_comp10 = io.StringIO()
        data_io_comp11 = io.StringIO()
        data_io_comp12 = io.StringIO()
        data_io_comp13 = io.StringIO()
        data_io_comp14 = io.StringIO()
        data_io_comp15 = io.StringIO()
        data_io_comp16 = io.StringIO()
        data_io_all = io.StringIO()

        # cabecalho dos arquivos csv, separador utilizado é o ';'
        #data_io_time_cost_r3.write("""y;x\n""")
        #data_io_visited_cost_r3.write("""y;x\n""")

        #data_io_visited.write("""y;x\n""")
        #data_io_opened.write("""y;x\n""")

        #data_io_time_cost_dnn1.write("""y;x\n""")
    # data_io_visited_cost_dnn1.write("""y;x\n""")
        #data_io_time_cost_dnn2.write("""y;x\n""")
        #data_io_visited_cost_dnn2.write("""y;x\n""")
        # data_io_comp.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp2.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp3.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp4.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp5.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp6.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp7.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp8.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp9.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp10.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp11.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp12.write("""custo;tempo;nodos_visitados;nodos_abertos\n""")
        # data_io_comp13.write("""custo;tempo;nodos_visitados;nodos_abertos;t_mapa_heuristico\n""")
        # data_io_comp14.write("""custo;tempo;nodos_visitados;nodos_abertos;t_mapa_heuristico\n""")
        # data_io_comp15.write("""custo;tempo;nodos_visitados;nodos_abertos;t_mapa_heuristico\n""")
        # data_io_comp16.write("""custo;tempo;nodos_visitados;nodos_abertos;t_mapa_heuristico\n""")
        # #data_io_all.write("""ox;oy;oh;x1;y1;h1;x2;y2;h2;c;d;v;nodos_visitados;total_time;time_search;time_h_map\n""")

        #data_io_all.write("""ox;oy;oh;x1;y1;h1;x2;y2;h2;c;d;v;nodos_visitados;total_time;time_search;time_h_map\n""")

        if not os.path.exists("./DADOS_RESULTADOS/"):
            os.makedirs("./DADOS_RESULTADOS/")

        #write_dataset_csv('./DADOS_RESULTADOS/time_cost_r3.csv', data_io_time_cost_r3)
    # write_dataset_csv('./DADOS_RESULTADOS/visited_cost_r3.csv', data_io_visited_cost_r3)
        #write_dataset_csv('./DADOS_RESULTADOS/time_cost_dnn1.csv', data_io_time_cost_dnn1)
    # write_dataset_csv('./DADOS_RESULTADOS/visited_cost_dnn1.csv', data_io_visited_cost_dnn1)
    # write_dataset_csv('./DADOS_RESULTADOS/time_cost_dnn2.csv', data_io_time_cost_dnn2)
    # write_dataset_csv('./DADOS_RESULTADOS/visited_cost_dnn2.csv', data_io_visited_cost_dnn2)
        write_dataset_csv('./DADOS_RESULTADOS/A_star'+str(mp.id_map)+'.csv', data_io_comp)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_DNN_ABS'+str(mp.id_map)+'.csv', data_io_comp13)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_DNN_CF'+str(mp.id_map)+'.csv', data_io_comp14)
        #write_dataset_csv('./DADOS_RESULTADOS/A_star'+str(mp.id_map)+'.csv', data_io_comp2)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_FROZEN_ABS'+str(mp.id_map)+'.csv', data_io_comp3)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_FROZEN_CF'+str(mp.id_map)+'.csv', data_io_comp4)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_ITERATIVE_ABS'+str(mp.id_map)+'.csv', data_io_comp5)
        write_dataset_csv('./DADOS_RESULTADOS/A_star_ITERATIVE_CF'+str(mp.id_map)+'.csv', data_io_comp6)
        write_dataset_csv('./DADOS_RESULTADOS/BIA_star'+str(mp.id_map)+'.csv', data_io_comp7)
        write_dataset_csv('./DADOS_RESULTADOS/BIA_star_DNN_ABS'+str(mp.id_map)+'.csv', data_io_comp15)
        write_dataset_csv('./DADOS_RESULTADOS/BIA_star_DNN_CF'+str(mp.id_map)+'.csv', data_io_comp16)
        #write_dataset_csv('./DADOS_RESULTADOS/BIA_star'+str(mp.id_map)+'.csv', data_io_comp8)
        write_dataset_csv('./DADOS_RESULTADOS/BIA_star_FROZEN_ABS'+str(mp.id_map)+'.csv', data_io_comp9)
        write_dataset_csv('./DADOS_RESULTADOS/BIA_star_FROZEN_CF'+str(mp.id_map)+'.csv', data_io_comp10)
        write_dataset_csv('./DADOS_RESULTADOS/BIA_star_ITERATIVE_ABS'+str(mp.id_map)+'.csv', data_io_comp11)
        write_dataset_csv('./DADOS_RESULTADOS/BIA_star_ITERATIVE_CF'+str(mp.id_map)+'.csv', data_io_comp12)
    # write_dataset_csv('./DADOS_RESULTADOS/all.csv', data_io_all)
        #write_dataset_csv('./DADOS_RESULTADOS/visited.csv',data_io_visited)
        teste= TestVars.test
        
        # Realiza o mesmo processo para cada observador
        print(len(viewpoints))
        for vpconfig in [1]:#GenerateVars.vpconfigs:
            #observer = (int(vpconfig[1] * CELL_WIDTH), int(vpconfig[0] * CELL_HEIGHT), mde.grid[vpconfig[0], vpconfig[1]])  # Coordenadas do observador

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
            data_io_comp5 = io.StringIO()
            data_io_comp6 = io.StringIO()
            data_io_comp7 = io.StringIO()
            data_io_comp8 = io.StringIO()
            data_io_comp9 = io.StringIO()
            data_io_comp10 = io.StringIO()
            data_io_comp11 = io.StringIO()
            data_io_comp12 = io.StringIO()
            data_io_comp13 = io.StringIO()
            data_io_comp14 = io.StringIO()
            data_io_comp15 = io.StringIO()
            data_io_comp16 = io.StringIO()
            data_io_all = io.StringIO()

            b = 0.5  # Fator de importância da segurança no cálculo do custo
            #visibility_map_file = './VIEWSHEDS/VIEWSHED_' + str(vpconfig[0]) + '_' + str(vpconfig[1]) + '.png'
            #visibility_map_file = './VIEWSHEDS/VIEWSHED_CONFIG_' + str(vpconfig) +'.png'

            #viewshed = read_viewshed(visibility_map_file)
            #viewshed = g.normalize_visibility(viewshed)
            #g.update_vertices_risk(viewshed)
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
                src_coords = pair[0] #pair[0](128,192)
                dest_coords = pair[1] #pair[1](58,92)
                source_id = get_id_by_coords(src_coords[0], src_coords[1]) # Cada ponto da amostra é o ponto de origem da iteração
                source = g.get_vertex(source_id)
                #print("aaaa",source)
                dest_id = get_id_by_coords(dest_coords[0], dest_coords[1])
                dest = g.get_vertex(dest_id)
                
                global dnn_heuristic_dict1
                global dnn_heuristic_dict2
                global dnn_heuristic_dict_CF_D
                global dnn_heuristic_frozen
                global dnn_heuristic_frozen_cf
                global dnn_heuristic_dict_CF_S
                global dnn_heuristic_iterative
                global dnn_heuristic_iterative_cf
                global dnn_heuristic_dict2_ABS_D
                global dnn_heuristic_dict2_ABS_S
                global dnn_heuristic_test
                #print("A distancia em linha reta no r3 é: ",r3_heuristic(source,dest))
                #carrega a heuristica entre todos os pontos para o ponto alvo posteriormente é usada como consulta
                #print("A distancia em linha reta no r3 é: ",r3_heuristic(source,dest))
                #carrega a heuristica entre todos os pontos para o ponto alvo posteriormente é usada como consulta
                '''dnn_heuristic_dict_CF_D, h_map_time1 = heuristic_dict2_observadores(g, model_CF, dest,vpconfig)
                dnn_heuristic_dict_CF_S, h_map_time1 = heuristic_dict2_observadores(g, model_CF, source,vpconfig)
                dnn_heuristic_dict2_ABS_D, h_map_time2 = heuristic_dict2_observadores(g, model_ABS, dest,vpconfig)
                dnn_heuristic_dict2_ABS_S, h_map_time2 = heuristic_dict2_observadores(g, model_ABS, source,vpconfig)'''
                global dnn_heuristic_dict_CF_D
                global dnn_heuristic_dict_CF_S
                global dnn_heuristic_dict2_ABS_D
                global dnn_heuristic_dict2_ABS_S
                global mapa_teste
                #print("A distancia em linha reta no r3 é: ",r3_heuristic(source,dest))
                #carrega a heuristica entre todos os pontos para o ponto alvo posteriormente é usada como consulta
                
                #dnn_heuristic_dict_CF_D, h_map_time1 = heuristic_dict1_multiplos_mapas(g, model_CF, dest,mp.id_map)
                #dnn_heuristic_dict_CF_S, h_map_time1 = heuristic_dict1_multiplos_mapas(g, model_CF, source,mp.id_map)
                #mapa_teste, tempo_aaaa = heuristic_dict1_standard_heuristic(g,dest)
                #dnn_heuristic_dict2_ABS_D, h_map_time2 = heuristic_dict1_multiplos_mapas(g, model_ABS, dest,mp.id_map)
                #dnn_heuristic_dict2_ABS_S, h_map_time2 = heuristic_dict1_multiplos_mapas(g, model_ABS, source,mp.id_map)
                #4 casos:
             #   dnn_heuristic_dict1, h_map_time1 = heuristic_dict1_multiplos_mapas(g, model1, dest)
                #dnn_heuristic_dict2, h_map_time2 = heuristic_dict1_multiplos_mapas(g, model_ABS, dest, mp.id_map)
                # dnn_heuristic_frozen, h_map_frozen = heuristic_dict1_multiplos_mapas_frozen_graph(g, dest, mp.id_map)
                # dnn_heuristic_frozen_cf, h_map_frozen2 = heuristic_dict1_multiplos_mapas_frozen_graph_cf(g, dest, mp.id_map)
                # dnn_heuristic_iterative, h_map_iterative = heuristic_dict1_multiplos_mapas_iterative(g, model_ABS, source, dest, mp.id_map)
                dnn_heuristic_iterative = {}
                dnn_heuristic_iterative_cf = {}

                #print("tempo de duração mapa heuristico: ", h_map_time2)

                #4 casos:
                #1) A* simples, heurística r3
                #1) A* simples, heurística r3
                
                #print("\nA* com heurística")
                '''b=0
                heuristic = dict_dnn_heuristic_abs_d
                t13 = time()
                opened13, count_visited13, count_open13, visited13, cost13 = astar(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t13 = time() - t13'''
                '''path13 = [dest.get_id()]
                print("custo do a* mapa heuristico: ",cost13)
                print("nodos visitados: ",count_visited13)
                print("nodos abertos: ",count_open13)
                count_visible13 = count_visible_nodes(dest, path13, 0)
                path_len13 = len(path13)
                print("tempo de duração: ", t13)'''
                g.reset()
                
                '''b=0
                heuristic = dict_dnn_heuristic_cf_d
                t14 = time()
                opened14, count_visited14, count_open14, visited14, cost14 = astar_correction_factor(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t14 = time() - t14'''
                '''path14 = [dest.get_id()]
                print("custo do a* mapa heuristico CF: ",cost14)
                print("nodos visitados: ",count_visited14)
                print("nodos abertos: ",count_open14)
                count_visible14 = count_visible_nodes(dest, path14, 0)
                path_len14 = len(path14)
                print("tempo de duração: ", t14)'''
                g.reset()

                #print("")

                '''heuristic = consult_frozen_graph_abs
                t3 = time()
                opened3, count_visited3, count_open3, visited3, cost3 = astar(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t3 = time() - t3'''
                '''path3 = [dest.get_id()]
                print("custo do a* frozen graph: ",cost3)
                print("nodos visitados: ",count_visited3)
                print("nodos abertos: ",count_open3)
                count_visible3 = count_visible_nodes(dest, path3, 0)
                path_len3 = len(path3)'''
                # print("tempo de duração: ", t3)
                g.reset()
                
                '''heuristic = consult_frozen_graph_cf
                t4 = time()
                opened4, count_visited4, count_open4, visited4, cost4 = astar_correction_factor(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t4 = time() - t4'''
                '''path4 = [dest.get_id()]
                print("custo do a* frozen graph CF: ",cost4)
                print("nodos visitados: ",count_visited4)
                print("nodos abertos: ",count_open4)
                count_visible4 = count_visible_nodes(dest, path4, 0)
                path_len4 = len(path4)'''
                # print("tempo de duração: ", t4)
                g.reset()

               # print("")

                '''heuristic = r3_heuristic
                t1 = time()
                opened1, count_visited1, count_open1, visited1, cost1 = astar(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t1 = time() - t1
                path1 = [dest.get_id()]
                print("custo do a* r3: ",cost1)
                print("nodos visitados: ",count_visited1)
                print("nodos abertos: ",count_open1)
                count_visible1 = count_visible_nodes(dest, path1, 0)
                path_len1 = len(path1)
                print("tempo de duração: ", t1)
                #print("mapa heuristico ", tempo_aaaa)
                #print("\n")'''
                #g.reset()

                #print("")

                heuristic = dict_dnn_iterative_abs
                t5 = time()
                opened5, count_visited5, count_open5, visited5, cost5 = astar(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t5 = time() - t5
                '''path5 = [dest.get_id()]
                print("custo do a* iterative: ",cost5)
                print("nodos visitados: ",count_visited5)
                print("nodos abertos: ",count_open5)
                count_visible5 = count_visible_nodes(dest, path5, 0)
                path_len5 = len(path5)'''
                # print("tempo de duração: ", t5)
                g.reset()

                #print("")

                heuristic = dict_dnn_iterative_cf
                t6 = time()
                opened6, count_visited6, count_open6, visited6, cost6 = astar_correction_factor(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t6 = time() - t6
                '''path6 = [dest.get_id()]
                print("custo do a* iterative cf: ",cost6)
                print("nodos visitados: ",count_visited6)
                print("nodos abertos: ",count_open6)
                count_visible6 = count_visible_nodes(dest, path6, 0)
                path_len6 = len(path6)'''
                # print("tempo de duração: ", t6)
                g.reset()

                """b=0
                heuristic = dnn_predict_test
                t2 = time()
                opened2, count_visited2, count_open2, visited2, cost2 = astar(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t2 = time() - t2
                path2 = [dest.get_id()]
                print("custo do a* TESTE: ",cost2)
                print("nodos visitados: ",count_visited2)
                print("nodos abertos: ",count_open2)
                count_visible2 = count_visible_nodes(dest, path2, 0)
                path_len2 = len(path2)
                print("tempo de duração: ", t2)
                #print("mapa heuristico ", tempo_aaaa)
                #print("\n")
                g.reset()"""
                
                '''print("\nLISTAAAAAA")
                print(tf.config.list_physical_devices('GPU'))
                print("\nA* DNN Absoluto")'''
                b=0
                heuristic = r3_heuristic
                t1 = time()
                opened1, count_visited1, count_open1, visited1, cost1 = astar(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t1 = time() - t1
                '''path1 = [dest.get_id()]
                print("custo do a*: ",cost1)
                print("nodos visitados: ",count_visited1)
                print("nodos abertos: ",count_open1)
                count_visible1 = count_visible_nodes(dest, path1, 0)
                path_len1 = len(path1)'''
                # print("tempo de duração: ", t1)
                # print("mapa heuristico ", h_map_time2)
                #print("\n")
                g.reset()
                
                #print("\nA* Correction Factor")
                '''b=0
                heuristic = dict_dnn_heuristic_cf_d
                t2 = time()
                opened2, count_visited2, count_open2, visited2, cost2 = astar_correction_factor(g, source, dest, b, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t2 = time() - t2
                path2 = [dest.get_id()]
                print("custo do a*: ",cost2)
                print("nodos visitados: ",count_visited2)
                print("nodos abertos: ",count_open2)
                count_visible2 = count_visible_nodes(dest, path2, 0)
                path_len2 = len(path2)
                print("tempo de duração: ", t2)
                print("mapa heuristico ", h_map_time1)'''
                #print("\n")
                g.reset()

                #print("terminou A*\n")
                
                #2)A* adaptado, heuristica r3 e caminhos seguros
                '''b=0
                heuristic = r3_heuristic
                #heuristicABSS = dict_dnn_heuristic_abs_s
                t5 = time()
                opened5, count_visited5, count_open5, visited5, cost5 = biastar(g, source, dest, b, heuristic, heuristic)
                t5 = time() - t5
                print("custo do biA* topografico: ",cost5)
                print("nodos visitados: ",count_visited5)
                print("nodos abertos: ",count_open5)
                
                path5 = [dest.get_id()]
                count_visible5 = count_visible_nodes(dest, path5, 0)
                path_len5 = len(path5)
                print("tempo de duração: ", t5)
                print("Terminou A* topo\n")
                g.reset()'''
                
                '''b=0
                heuristicABSD = dict_standard_heuristic
                #heuristicABSS = dict_dnn_heuristic_abs_s
                t6 = time()
                opened6, count_visited6, count_open6, visited6, cost6 = biastar(g, source, dest, b, heuristicABSD, heuristicABSD)
                t6 = time() - t6
                print("custo do biA* mapa heuristico: ",cost6)
                print("nodos visitados: ",count_visited6)
                print("nodos abertos: ",count_open6)
                
                path6 = [dest.get_id()]
                count_visible6 = count_visible_nodes(dest, path6, 0)
                path_len6 = len(path6)
                print("tempo de duração: ", t6)
                print("tempo do mapeamente heurístico: ", tempo_aaaa)
                print("Terminou A* topo\n")
                g.reset()'''
                
                
                '''b=0
                heuristic = dict_dnn_heuristic_abs_d
                t15 = time()
                opened15, count_visited15, count_open15, visited15, cost15 = biastar(g, source, dest, b, heuristic,heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t15 = time() - t15'''
                '''path15 = [dest.get_id()]
                print("custo do a* mapa heuristico: ",cost15)
                print("nodos visitados: ",count_visited15)
                print("nodos abertos: ",count_open15)
                count_visible15 = count_visible_nodes(dest, path15, 0)
                path_len15 = len(path15)
                print("tempo de duração: ", t15)'''
                g.reset()
                
                ''''b=0
                heuristic = dict_dnn_heuristic_cf_d
                t16 = time()
                opened16, count_visited16, count_open16, visited16, cost16 = biastar_DNN_CF(g, source, dest, b, heuristic, heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t16 = time() - t16'''
                '''path16 = [dest.get_id()]
                print("custo do a* mapa heuristico CF: ",cost16)
                print("nodos visitados: ",count_visited16)
                print("nodos abertos: ",count_open16)
                count_visible16 = count_visible_nodes(dest, path16, 0)
                path_len16 = len(path16)
                print("tempo de duração: ", t16)'''
                g.reset()
                
                
                
                dnn_heuristic_iterative = {}
                dnn_heuristic_iterative_cf = {}
                
                #print("\nBiA* DNN Absoluto")
                b=0
                heuristicABSD = dict_dnn_iterative_abs
                #heuristicABSS = dict_dnn_heuristic_abs_s
                t11 = time()
                opened11, count_visited11, count_open11, visited11, cost11 = biastar(g, source, dest, b, heuristicABSD, heuristicABSD)
                t11 = time() - t11
                '''print("custo do biA* dnn absoluto iterative: ",cost11)
                print("nodos visitados: ",count_visited11)
                print("nodos abertos: ",count_open11)                
                
                path11 = [dest.get_id()]
                count_visible11 = count_visible_nodes(dest, path11, 0)
                path_len11 = len(path11)'''
                # print("tempo de duração: ", t11)
                # print("tempo do mapeamente heurístico: ", h_map_time2)
                # print("Terminou A* topo\n")
                g.reset()
                
                #print("\nBiA* Correction Factor")
                b=0
                heuristicABSD = dict_dnn_iterative_cf
                t12 = time()
                opened12, count_visited12, count_open12, visited12, cost12 = biastar_DNN_CF(g, source, dest, b, heuristicABSD, heuristicABSD)
                t12 = time() - t12
                '''print("custo do biA* dnn cf iterative: ",cost12)
                print("nodos visitados: ",count_visited12)
                print("nodos abertos: ",count_open12)
                
                path12 = [dest.get_id()]
                count_visible12 = count_visible_nodes(dest, path12, 0)
                path_len12 = len(path12)'''
                # print("tempo de duração: ", t12)
                # print("tempo do mapeamente heurístico: ", h_map_time1)
                # print("Terminou A* topo\n")
                g.reset()
                
                '''heuristic = consult_frozen_graph_abs
                t9 = time()
                opened9, count_visited9, count_open9, visited9, cost9 = biastar(g, source, dest, b, heuristic,heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t9 = time() - t9'''
                '''path9 = [dest.get_id()]
                print("custo do bia* frozen graph: ",cost9)
                print("nodos visitados: ",count_visited9)
                print("nodos abertos: ",count_open9)
                count_visible9 = count_visible_nodes(dest, path9, 0)
                path_len9 = len(path9)'''
                # print("tempo de duração: ", t9)
                g.reset()
                
                '''heuristic = consult_frozen_graph_cf
                t10 = time()
                opened10, count_visited10, count_open10, visited10, cost10 = biastar_DNN_CF(g, source, dest, b, heuristic,heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t10 = time() - t10'''
                '''path10 = [dest.get_id()]
                print("custo do bia* frozen graph CF: ",cost10)
                print("nodos visitados: ",count_visited10)
                print("nodos abertos: ",count_open10)
                count_visible10 = count_visible_nodes(dest, path10, 0)
                path_len10 = len(path10)'''
                # print("tempo de duração: ", t10)
                g.reset()
                
                b=0
                heuristic = r3_heuristic
                t7 = time()
                opened7, count_visited7, count_open7, visited7, cost7 = biastar(g, source, dest, b, heuristic,heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t7 = time() - t7
                '''path7 = [dest.get_id()]
                print("custo do bia* dnn abs: ",cost7)
                print("nodos visitados: ",count_visited7)
                print("nodos abertos: ",count_open7)
                count_visible7 = count_visible_nodes(dest, path7, 0)
                path_len7 = len(path7)'''
                # print("tempo de duração: ", t7)
                # print("mapa heuristico ", h_map_time2)
                #print("\n")
                g.reset()
                
                #print("\nA* Correction Factor")
                '''b=0
                heuristic = dict_dnn_heuristic_cf_d
                t8 = time()
                opened8, count_visited8, count_open8, visited8, cost8 = biastar_DNN_CF(g, source, dest, b, heuristic,heuristic) #fator b não é utilizado no cálculo, mas para fins de análise dos resultados
                t8 = time() - t8'''
                '''path8 = [dest.get_id()]
                print("custo do bia* dnn cf: ",cost8)
                print("nodos visitados: ",count_visited8)
                print("nodos abertos: ",count_open8)
                count_visible8 = count_visible_nodes(dest, path8, 0)
                path_len8 = len(path8)
                print("tempo de duração: ", t8)
                print("mapa heuristico ", h_map_time1)'''
                #print("\n")
                g.reset()
                #data_io_time_cost_r6.write("""%s;%s\n""" % (t2, cost2))
                #data_io_visited_cost_r6.write("""%s;%s\n""" % (count_visited2, cost2))
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
                #exit()
                #data_io_time_cost_dnn2.write("""%s;%s\n""" % (t4, cost4))
                #data_io_visited_cost_dnn2.write("""%s;%s\n""" % (count_visited4, cost4))
                
                #data_io_comp.write("""%s;%s;%s;%s\n""" %(cost1,t1,count_visited1,count_open1))
                #data_io_comp2.write("""%s;%s;%s;%s;%s\n""" %(cost2,t2,count_visited2,count_open2,h_map_time1))
                data_io_comp.write("""%s;%s;%s;%s\n""" %(cost1,t1,count_visited1,count_open1))
                #data_io_comp2.write("""%s;%s;%s;%s;%s\n""" %(cost2,t2+h_map_time2,count_visited2,count_open2,h_map_time2))
                #data_io_comp3.write("""%s;%s;%s;%s\n""" %(cost3,t3,count_visited3,count_open3))
                #data_io_comp4.write("""%s;%s;%s;%s\n""" %(cost4,t4,count_visited4,count_open4))
                data_io_comp5.write("""%s;%s;%s;%s\n""" %(cost5,t5,count_visited5,count_open5))
                data_io_comp6.write("""%s;%s;%s;%s\n""" %(cost6,t6,count_visited6,count_open6))
                #data_io_comp5.write("""%s;%s;%s;%s\n""" %(cost5,t5,count_visited5,count_open5))
                #data_io_comp6.write("""%s;%s;%s;%s;%s\n""" %(cost6,t6,count_visited6,count_open6,h_map_time1))
                data_io_comp7.write("""%s;%s;%s;%s\n""" %(cost7,t7,count_visited7,count_open7))
                #data_io_comp8.write("""%s;%s;%s;%s;%s\n""" %(cost8,t8+h_map_time2,count_visited8,count_open8,h_map_time2))
                #data_io_comp9.write("""%s;%s;%s;%s\n""" %(cost9,t9,count_visited9,count_open9))
                #data_io_comp10.write("""%s;%s;%s;%s\n""" %(cost10,t10,count_visited10,count_open10))
                data_io_comp11.write("""%s;%s;%s;%s\n""" %(cost11,t11,count_visited11,count_open11))
                data_io_comp12.write("""%s;%s;%s;%s\n""" %(cost12,t12,count_visited12,count_open12))
                #data_io_comp13.write("""%s;%s;%s;%s;%s\n""" %(cost13,t13+h_map_time2,count_visited13,count_open13,h_map_time2))
                #data_io_comp14.write("""%s;%s;%s;%s;%s\n""" %(cost14,t14+h_map_time1,count_visited14,count_open14,h_map_time1))
                #data_io_comp15.write("""%s;%s;%s;%s;%s\n""" %(cost15,t15+h_map_time2,count_visited15,count_open15,h_map_time2))
                #data_io_comp16.write("""%s;%s;%s;%s;%s\n""" %(cost16,t16+h_map_time1,count_visited16,count_open16,h_map_time1))
                #data_io_comp2.write("""%s;%s;%s;%s\n""" %(cost2,t2+h_map_time2,count_visited2,count_open2))
                #data_io_comp3.write("""%s;%s;%s;%s\n""" %(cost3,t3,count_visited3,count_open3))
                #data_io_comp4.write("""%s;%s;%s;%s\n""" %(cost4,t4+h_map_time1,count_visited4,count_open4))
                #data_io_comp3.write("""%s;%s;%s;%s\n""" %(cost5,t5,count_visited5,count_open5))            
                i+=1
                print(i)
                # print('Tempo: ' + str(time() - start_time) + ' segundos')
                
                if teste:
                    #teste=False
                    #print("\n\n\n Quero ver ",opened3[0])
                    #opened3=opened3.reverse()
                    #visited3=visited3.reverse()        
                     
                    for i in range(len(opened3)):
                        data_io_opened.write("""%s\n"""%str((opened3[i])))
                    for i in range(len(visited3)):
                        data_io_visited.write("""%s\n"""%str((visited3[i])))
                    
                    
                    write_dataset_test_csv('./DADOS_RESULTADOS/visited.csv',data_io_visited)
                    write_dataset_test_csv('./DADOS_RESULTADOS/opened.csv',data_io_opened)
                    #opened4=opened4.reverse()
                    #visited4=visited4.reverse()        
                    for i in range(len(opened4)):
                        data_io_opened2.write("""%s\n"""%str((opened4[i])))
                    for i in range(len(visited4)):
                        data_io_visited2.write("""%s\n"""%str((visited4[i])))

                    write_dataset_test_csv('./DADOS_RESULTADOS/visited2.csv',data_io_visited2)
                    write_dataset_test_csv('./DADOS_RESULTADOS/opened2.csv',data_io_opened2)
                    #opened7=opened7.reverse()
                    #visited7=visited7.reverse()   
                    for i in range(len(opened7)):
                        data_io_opened3.write("""%s\n"""%str((opened7[i])))
                    for i in range(len(visited7)):
                        data_io_visited3.write("""%s\n"""%str((visited7[i])))

                    write_dataset_test_csv('./DADOS_RESULTADOS/visited3.csv',data_io_visited3)
                    write_dataset_test_csv('./DADOS_RESULTADOS/opened3.csv',data_io_opened3)
                    #opened8=opened8.reverse()
                    #visited8=visited8.reverse()   
                    for i in range(len(opened8)):
                        data_io_opened4.write("""%s\n"""%str((opened8[i])))
                    for i in range(len(visited8)):
                        data_io_visited4.write("""%s\n"""%str((visited8[i])))

                    write_dataset_test_csv('./DADOS_RESULTADOS/visited4.csv',data_io_visited4)
                    write_dataset_test_csv('./DADOS_RESULTADOS/opened4.csv',data_io_opened4)
                    exit()
                
                # break #realizando testes
                
                #data_io_all.write("""%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n""" %
                #                  (observer[0], observer[1], observer[2], int(src_coords[1] * CELL_WIDTH), int(src_coords[0] * CELL_HEIGHT), mde.grid[src_coords[0], src_coords[1]],int(dest_coords[1] *CELL_WIDTH), int(dest_coords[0]*CELL_HEIGHT), mde.grid[dest_coords[0], dest_coords[1]], cost4, distance4,safety4,count_visited4,t4, float(t3-h_map_time2), h_map_time2))
            #write_dataset_csv('./DADOS_RESULTADOS/A_star'+str(mp.id_map)+'.csv', data_io_comp)
            #write_dataset_csv('./DADOS_RESULTADOS/A_star_dnn'+str(mp.id_map)+'.csv', data_io_comp2)
            #write_dataset_csv('./DADOS_RESULTADOS/Theta_star.csv', data_io_comp3)
            #write_dataset_csv('./DADOS_RESULTADOS/A_star_dnn_CF'+str(mp.id_map)+'.csv', data_io_comp4)
            write_dataset_csv('./DADOS_RESULTADOS/A_star'+str(mp.id_map)+'.csv', data_io_comp)
            write_dataset_csv('./DADOS_RESULTADOS/A_star_DNN_ABS'+str(mp.id_map)+'.csv', data_io_comp13)
            write_dataset_csv('./DADOS_RESULTADOS/A_star_DNN_CF'+str(mp.id_map)+'.csv', data_io_comp14)
            #write_dataset_csv('./DADOS_RESULTADOS/A_star'+str(mp.id_map)+'.csv', data_io_comp2)
            write_dataset_csv('./DADOS_RESULTADOS/A_star_FROZEN_ABS'+str(mp.id_map)+'.csv', data_io_comp3)
            write_dataset_csv('./DADOS_RESULTADOS/A_star_FROZEN_CF'+str(mp.id_map)+'.csv', data_io_comp4)
            write_dataset_csv('./DADOS_RESULTADOS/A_star_ITERATIVE_ABS'+str(mp.id_map)+'.csv', data_io_comp5)
            write_dataset_csv('./DADOS_RESULTADOS/A_star_ITERATIVE_CF'+str(mp.id_map)+'.csv', data_io_comp6)
            write_dataset_csv('./DADOS_RESULTADOS/BIA_star'+str(mp.id_map)+'.csv', data_io_comp7)
            write_dataset_csv('./DADOS_RESULTADOS/BIA_star_DNN_ABS'+str(mp.id_map)+'.csv', data_io_comp15)
            write_dataset_csv('./DADOS_RESULTADOS/BIA_star_DNN_CF'+str(mp.id_map)+'.csv', data_io_comp16)
            #write_dataset_csv('./DADOS_RESULTADOS/BIA_star'+str(mp.id_map)+'.csv', data_io_comp8)
            write_dataset_csv('./DADOS_RESULTADOS/BIA_star_FROZEN_ABS'+str(mp.id_map)+'.csv', data_io_comp9)
            write_dataset_csv('./DADOS_RESULTADOS/BIA_star_FROZEN_CF'+str(mp.id_map)+'.csv', data_io_comp10)
            write_dataset_csv('./DADOS_RESULTADOS/BIA_star_ITERATIVE_ABS'+str(mp.id_map)+'.csv', data_io_comp11)
            write_dataset_csv('./DADOS_RESULTADOS/BIA_star_ITERATIVE_CF'+str(mp.id_map)+'.csv', data_io_comp12)
            #write_dataset_csv('./DADOS_RESULTADOS/time_cost_r3.csv', data_io_time_cost_r3)
        # write_dataset_csv('./DADOS_RESULTADOS/visited_cost_r3.csv', data_io_visited_cost_r3)
            #write_dataset_csv('./DADOS_RESULTADOS/time_cost_dnn1.csv', data_io_time_cost_dnn1)
            #write_dataset_csv('./DADOS_RESULTADOS/visited_cost_dnn1.csv', data_io_visited_cost_dnn1)
            #write_dataset_csv('./DADOS_RESULTADOS/time_cost_dnn2.csv', data_io_time_cost_dnn2)
            #write_dataset_csv('./DADOS_RESULTADOS/visited_cost_dnn2.csv', data_io_visited_cost_dnn2)
            #write_dataset_csv('./DADOS_RESULTADOS/comp.csv', data_io_comp)
            #write_dataset_csv('./DADOS_RESULTADOS/all.csv', data_io_all)

            print('Tempo: ' + str(time() - start_time) + ' segundos')
        
        #break #realizando testes tirar depois

def main2():
    maps = GenerateVars.maps
    reduction_factor = 1

    model_name1 = 'model_32_20230227-164136_checkpoint_19_0.0147.hdf5'
    model_name2 = 'model_32_20230220-165452_checkpoint_97_0.2473.hdf5'

    model1 = load_model(model_name1)
    model2 = load_model(model_name2)

    for mp in maps:
        map_dir = GenerateVars.maps_dir
        map_path = map_dir + mp.filename
        mde = Mde(map_path, mp.reduction_factor)

        g = Graph(mde)
        paths_per_map = 1250

        data_io_comp = io.StringIO()
        data_io_comp.write("""cost;euclidean;absCost;cfCost\n""")

        if not os.path.exists("./DADOS_RESULTADOS/"):
            os.makedirs("./DADOS_RESULTADOS/")

        write_dataset_csv('./DADOS_RESULTADOS/HeuristicsCosts'+str(mp.id_map)+'.csv', data_io_comp)

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
            global dnn_heuristic_dict1
            global dnn_heuristic_dict2

            dnn_heuristic_dict1, h_map_time1 = heuristic_dict1_multiplos_mapas(g, model1, dest)
            dnn_heuristic_dict2, h_map_time2 = heuristic_dict1_multiplos_mapas(g, model2, dest)

            b=0
            heuristic = dict_dnn_heuristic2            
            openedAstar, count_visitedAstar, count_openAstar, visitedAstar, costAstar = astar(g, source, dest, b, heuristic)
            g.reset()

            euclidean = r3_heuristic(source, dest)
            cfCost = euclidean * dict_dnn_heuristic1(source, dest)
            absCost = dict_dnn_heuristic2(source, dest)

            data_io_comp.write("""%s;%s;%s;%s\n""" %(costAstar, euclidean, absCost, cfCost))

        write_dataset_csv('./DADOS_RESULTADOS/HeuristicsCosts'+str(mp.id_map)+'.csv', data_io_comp)

def main3():
    prefixes = ["A", "B", "C", "D", "E", "F", "G", "H"]

    name = "recorte300x300"

    for p in prefixes:
        print(p)
        filename = f"{name}{p}.tif"
        outputFileName = f"{name}{p}"

        reduction_factor = 1 # Fator de redução de dimensão do mapa (2 -> mapa 400x400 abstraído em 200x200)

        # Lê o arquivo do MDE e cria o grid do mapa
        mde = Mde(filename, reduction_factor)

        print('Criando o grafo')
        # Cria o grafo a partir do grid do MDE
        g = Graph(mde)

        angles = []

        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                source_id = get_id_by_coords(i, j)
                source = g.get_vertex(source_id)

                if i != GRID_ROWS - 1:
                    n1_id = get_id_by_coords(i+1, j)
                    n1 = g.get_vertex(n1_id)
                    angles.append(calcula_angulo(source, n1))

                if j != GRID_COLS - 1:
                    n2_id = get_id_by_coords(i, j+1)
                    n2 = g.get_vertex(n2_id)
                    angles.append(calcula_angulo(source, n2))

        bins = range(41)

        plt.hist(angles, bins=bins, color="grey")
        plt.xlabel("Angles (degrees)")
        plt.ylabel("Frequency")
        plt.ylim(0, 31500)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.grid(axis='y')
        plt.savefig(outputFileName)
        plt.cla()
        # plt.show()

    return angles


if __name__ == '__main__':
    main()
