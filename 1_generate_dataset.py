import io
import shutil
import random
import sys
import numpy as np
import math
import csv
import os
import glob
from Utilities.Graph import *
from Utilities.Vertex import *
from config_variables import GenerateVars, MapCase, VPCase, MDEVars
import matplotlib.image as mpimg
from time import process_time
from numba import cuda, jit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def r2_distance(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def r3_distance(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)



# Recupera o caminho percorrendo do fim para o inicio
def shortest(v, path):
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return


# Heuristica da distancia Euclidiana
def r3_heuristic(start, goal):
    x1, y1 = start.get_r2_coordinates()
    x2, y2 = goal.get_r2_coordinates()
    z1 = start.get_elevation()
    z2 = goal.get_elevation()

    dst = r3_distance(x1, x2, y1, y2, z1, z2)
    return dst


def get_visited_coord(graph, visited_vertices):
    visited = []
    for vertex_id in visited_vertices[::-1]:
        visited.append(graph.get_vertex(vertex_id).get_coordinates())
    return visited

# Recupera o id do vértice a partir das coordenadas no grid
def get_id_by_coords(i, j):
    return i * MDEVars.GRID_COLS + j


# Escreve os nodos do caminho em um arquivo csv
def save_path_csv(output, path):
    with open(output, 'w') as out:
        csv_out = csv.writer(out)
        for row in path:
            csv_out.writerow(row)


# Escreve os dados do caminho no csv do dataset
def write_dataset_csv(filename, data_io):
    with open(filename, 'a') as file:
        data_io.seek(0)
        shutil.copyfileobj(data_io, file)


# Gera e salva os mapas de visibilidade em arquivos png
def save_viewsheds(grid, viewpoints, view_radius, viewpoint_height):
    #todos = np.zeros((grid.shape[0], grid.shape[1]))
    for viewpoint_i, viewpoint_j in viewpoints:
        viewshed = vs.generate_viewshed(grid, viewpoint_i, viewpoint_j, view_radius, MDEVars.CELL_WIDTH, viewpoint_height)
        #todos = todos + viewshed
        output_file = 'VIEWSHED_' + str(viewpoint_i) + '_' + str(viewpoint_j) + '.png'
        vs.save_viewshed_image(viewshed, './VIEWSHEDS/' + output_file)
    #vs.save_viewshed_image(todos, './VIEWSHEDS/todos.png')


# Lê o png do mapa de visibilidade
def read_viewshed(file):
    img = mpimg.imread(file)
    viewshed = img[:, :, 0]
    return viewshed

# --------------------------------------------- CUDA SAFE SSSP -------------------------------------------------- #
# Funções para o cálculo dos mapas de custo paralelizados em GPU
# Baseados no algoritmo descrito em: Pawan Harish and P. J. Narayanan (2007), Accelerating large graph algorithms on the GPU using CUDA


@cuda.jit
def kernel1_without_S(V, E, W, M, C, U, n, b):
    tid = cuda.grid(1)
    if tid < n:
        if M[tid] == 1:
            M[tid] = 0
            start = V[tid]
            if tid == n-1:
                end = len(E)
            else:
                end = V[tid+1]
            for nid, w in zip(E[start:end], W[start:end]):
                cuda.atomic.min(U, nid, C[tid] + w)


@cuda.jit
def initialize_arrays(source, n, INF, M, C, U, S, b):
    tid = cuda.grid(1)
    if tid < n:
        C[tid] = INF
        U[tid] = INF
        if tid == source:
            M[tid] = 1
            C[tid] = S[tid] * b
            U[tid] = S[tid] * b


@cuda.jit
def initialize_arrays_without_S(source, n, INF, M, C, U, b):
    tid = cuda.grid(1)
    if tid < n:
        C[tid] = INF
        U[tid] = INF
        if tid == source:
            M[tid] = 1



@cuda.reduce
def sum_reduce(a, b):
    return a + b

@cuda.jit
def kernel2(M, C, U, n):
    tid = cuda.grid(1)
    if tid < n:
        if C[tid] > U[tid]:
            C[tid] = U[tid]
            M[tid] = 1
        U[tid] = C[tid]


@cuda.jit
def kernel1(V, E, W, S, M, C, U, n, b):
    tid = cuda.grid(1)
    if tid < n:
        if M[tid] == 1:
            M[tid] = 0
            start = V[tid]
            if tid == n-1:
                end = len(E)
            else:
                end = V[tid+1]
            for nid, w in zip(E[start:end], W[start:end]):
                cuda.atomic.min(U, nid, C[tid] + w + b * S[nid])

def cuda_safe_sssp(V, E, W, S, source, b):
    # V = lista de vertex, E = lista de edges, W = lista de pesos, S = lista de visibilidade do viewshed
    n = V.shape[0]
    INF = 999999
    threadsperblock = 128
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock

    M = np.zeros(n, dtype=np.int32)
    C = np.arange(n, dtype=np.float64)
    U = np.arange(n, dtype=np.float64)
    d_M = cuda.to_device(M)
    d_C = cuda.to_device(C)
    d_U = cuda.to_device(U)
    d_V = cuda.to_device(V)
    d_E = cuda.to_device(E)
    d_W = cuda.to_device(W)
    d_S = cuda.to_device(S)

    initialize_arrays[blockspergrid, threadsperblock](source, n, INF, d_M, d_C, d_U, d_S, b)
    mask = sum_reduce(d_M)
    while mask > 0:
        kernel1[blockspergrid, threadsperblock](d_V, d_E, d_W, d_S, d_M, d_C, d_U, n, b)
        kernel2[blockspergrid, threadsperblock](d_M, d_C, d_U, n)
        mask = sum_reduce(d_M)
    C = d_C.copy_to_host()
    return C

def cuda_safe_sssp_without_S(V, E, W, source, b):
    n = V.shape[0]
    INF = 999999
    threadsperblock = 128
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock

    M = np.zeros(n, dtype=np.int32)
    C = np.arange(n, dtype=np.float64)
    U = np.arange(n, dtype=np.float64)
    d_M = cuda.to_device(M)
    d_C = cuda.to_device(C)
    d_U = cuda.to_device(U)
    d_V = cuda.to_device(V)
    d_E = cuda.to_device(E)
    d_W = cuda.to_device(W)   

    initialize_arrays_without_S[blockspergrid, threadsperblock](source, n, INF, d_M, d_C, d_U, b)
    mask = sum_reduce(d_M)
    while mask > 0:
        kernel1_without_S[blockspergrid, threadsperblock](d_V, d_E, d_W, d_M, d_C, d_U, n, b)
        kernel2[blockspergrid, threadsperblock](d_M, d_C, d_U, n)
        mask = sum_reduce(d_M)
    C = d_C.copy_to_host()
    return C

# Cria listas de adjacências das conexões do grafo
def generate_sssp_arrays(g):
    V = [] #lista de vertex
    E = [] #lista de edges
    W = [] #lista de pesos
    for i in range(MDEVars.GRID_ROWS):
        for j in range(MDEVars.GRID_COLS):
            v_id = get_id_by_coords(i, j)
            v = g.get_vertex(v_id)
            edges_index = len(E)
            V.append(edges_index)
            for u_id in v.get_neighbors():
                E.append(u_id)
                W.append(v.edges[u_id])

    return np.array(V), np.array(E), np.array(W)

# Retorna o mapa de visibilidade como lista, o indice da lista é o id do respectivo vértice
def serialize_viewshed(viewshed):
    serialized_viewshed = []
    for i in range(MDEVars.GRID_ROWS):
        for j in range(MDEVars.GRID_COLS):
            serialized_viewshed.append(viewshed[i, j])
    return np.array(serialized_viewshed)

# ------------------------------------------------------------------------------------------------------------------ #

import AUXILIARES.generate_viewshed as vs


# Gera uma lista de tuplas de pontos de amostras para geração do dataset
def generate_sample_points(sampling_percentage,rows,collumns):

    # Divide o mapa em 4x4 clusters
    sections_n = 4
    sections_m = 4

    # Tamanho das divisões do mapa


    SECTION_ROWS = round(rows/sections_n)
    SECTION_COLS = round(collumns/sections_m)

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
            sample_points = section_points[0:sampling_size]
            P.extend(sample_points)
            section_points.clear()
    P.sort(key=lambda tup: (tup[1], tup[0]))
    return P


# Minimos e maximos locais
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


def generate_dataset():
    print('Gerando os pontos de amostra')    


    #
    #   MUDAR ROWS E COLLUMNS DE ACORDO COM O TAMANHO DO MAPA
    #
    sample_coords = generate_sample_points(GenerateVars.sampling_rate/100,rows=300,collumns=300) # Gera os pontos de amostra
    #
    #   MUDAR ROWS E COLLUMNS DE ACORDO COM O TAMANHO DO MAPA
    #   
    
    count=0
    
    for mp in GenerateVars.maps:
        map_path = GenerateVars.maps_dir + mp.filename
        mde = Mde(map_path, mp.reduction_factor)
        print(mp.filename)
        print(mp.id_map)

        print('Criando o grafo')
        # Cria o grafo a partir do MDE
        g = Graph(mde)
        # Transforma o grafo em 3 listas de vértices, arestas e pesos das arestas

        # Coordenadas de cada observador
        viewpoints = observer_points(mde.grid, MDEVars.GRID_ROWS, MDEVars.GRID_COLS, 1)
        # Raio de visão dos observadores
        view_radius = 40
        # Altura do observador (metros) em relação ao chão
        viewpoint_height = 5

        print('Salvando os viewsheds')
        if not os.path.exists("./VIEWSHEDS/"):
            os.makedirs("./VIEWSHEDS/")
        files = glob.glob('./VIEWSHEDS/*')
        for f in files:
            os.remove(f)
        save_viewsheds(mde.grid, viewpoints, view_radius, viewpoint_height)

        V, E, W = generate_sssp_arrays(g)

        print('Gerando o dataset: ', count)
        count=+1
        
        # Realiza o mesmo processo para cada observador
        start_time = process_time()
        visibility_map_file = './VIEWSHEDS/VIEWSHED_' + str(viewpoints[0][0]) + '_' + str(viewpoints[0][1]) + '.png'

        viewshed = read_viewshed(visibility_map_file)
        viewshed = g.normalize_visibility(viewshed)
        S = serialize_viewshed(viewshed)
        # S = lista de visibilidade do viewshed
        aux = 0
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaa")
        for src_coords in sample_coords:
            data_io = io.StringIO()
            source = get_id_by_coords(src_coords[0], src_coords[1]) # Cada ponto da amostra é o ponto de origem da iteração
            b = 0 # Fator de importância da segurança no cálculo do custo -> 0 para dijkstra padrão
            C = cuda_safe_sssp(V, E, W, S, source, b) # Gera o mapa de custos
            
            # Coleta os custos para cada um dos pontos seguintes da lista de pontos amostrados para evitar caminhos repetidos;
            if(GenerateVars.type_dataset == 1):
                for dest_coords in sample_coords[aux+1:]:
                    dest = get_id_by_coords(dest_coords[0], dest_coords[1])                
                    data_io.write("""%s,%s,%s,%s,%s,%s,%s,%s\n""" % (mp.id_map, 
                    str(int(src_coords[1] * MDEVars.CELL_WIDTH)), 
                    str(int(src_coords[0] * MDEVars.CELL_HEIGHT)),
                    str(mde.grid[src_coords[0], src_coords[1]]), 
                    str(int(dest_coords[1] * MDEVars.CELL_WIDTH)),
                    str(int(dest_coords[0] * MDEVars.CELL_HEIGHT)), 
                    mde.grid[dest_coords[0], dest_coords[1]], 
                    C[dest]))
                    #data_io.write("""%s,%s,%s,%s,%s,%s,%s\n""" % (int(src_coords[1] * MDEVars.CELL_WIDTH), int(src_coords[0] * MDEVars.CELL_HEIGHT),mde.grid[src_coords[0], src_coords[1]], int(dest_coords[1] * MDEVars.CELL_WIDTH),int(dest_coords[0] * MDEVars.CELL_HEIGHT), mde.grid[dest_coords[0], dest_coords[1]], C[dest]))
            elif(GenerateVars.type_dataset == 2):
                for dest_coords in sample_coords[aux+1:]:
                    dest = get_id_by_coords(dest_coords[0], dest_coords[1])                
                    data_io.write("""%s,%s,%s,%s,%s,%s,%s,%s\n""" % (mp.id_map, 
                    str(int(src_coords[1] * MDEVars.CELL_WIDTH)), 
                    str(int(src_coords[0] * MDEVars.CELL_HEIGHT)),
                    str(mde.grid[src_coords[0], src_coords[1]]), 
                    str(int(dest_coords[1] * MDEVars.CELL_WIDTH)),
                    str(int(dest_coords[0] * MDEVars.CELL_HEIGHT)), 
                    mde.grid[dest_coords[0], dest_coords[1]], 
                    C[dest]/r3_heuristic(g.get_vertex(source),g.get_vertex(dest))))
            elif(GenerateVars.type_dataset == 3):
                for dest_coords in sample_coords[aux+1:]:
                    dest = get_id_by_coords(dest_coords[0], dest_coords[1])                
                    data_io.write("""%s,%s,%s,%s,%s,%s,%s,%s\n""" % (mp.id_map, 
                    str(int(src_coords[1] * MDEVars.CELL_WIDTH)), 
                    str(int(src_coords[0] * MDEVars.CELL_HEIGHT)),
                    str(mde.grid[src_coords[0], src_coords[1]]), 
                    str(int(dest_coords[1] * MDEVars.CELL_WIDTH)),
                    str(int(dest_coords[0] * MDEVars.CELL_HEIGHT)), 
                    mde.grid[dest_coords[0], dest_coords[1]], 
                    r3_heuristic(g.get_vertex(source),g.get_vertex(dest))))
                
            aux = aux +1

            write_dataset_csv('dataset_sem_observador_mapa_\\VASCPO'+str(mp.id_map)+'.csv', data_io)
        print('Tempo: ' + str(process_time() - start_time) + ' segundos')
        
    print('Dataset gerado com sucesso!')
   
def main():
    generate_dataset()    

if __name__ == '__main__':
    main()