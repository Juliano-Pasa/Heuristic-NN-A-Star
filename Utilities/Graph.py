import numpy as np

from Utilities.Vertex import Vertex
from Utilities.Heuristics import r3_distance
from config_variables import MDEVars

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
        
        MDEVars.CELL_HEIGHT = self.pixel_resolution * reduction_factor
        MDEVars.CELL_WIDTH = self.pixel_resolution * reduction_factor
        MDEVars.GRID_COLS = self.grid.shape[0]
        MDEVars.GRID_ROWS = self.grid.shape[1]
        MDEVars.GRID_WIDTH = MDEVars.CELL_WIDTH * MDEVars.GRID_COLS
        MDEVars.GRID_HEIGHT = MDEVars.CELL_HEIGHT * MDEVars.GRID_ROWS

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


class Graph:
    def __init__(self, mde):
        self.vertices = {}  # Dicionário de vértices: key = id, value = objeto Vertex
        self.max_edge = 0.0
        self.min_edge = float("inf")
        self.create_vertices(mde)   # Popula o dicionário de vértices com 1 vértice para cada célula do grid do mde
        self.generate_edges(True)  # Parâmetro: True = considera as travessias diagonais; False = considera apenas os vizinhos imediatos

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
        for i in range(MDEVars.GRID_ROWS):
            for j in range(MDEVars.GRID_COLS):
                vertex_elevation = mde.grid[i, j]
                vertex_id = i * MDEVars.GRID_COLS + j
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

    def get_id_by_coords(i, j):
        return i * MDEVars.GRID_COLS + j

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
            if j1 < MDEVars.GRID_COLS:
                vertex2_id = i1 * MDEVars.GRID_COLS + j1
                vertex2 = self.get_vertex(vertex2_id)
                if vertex2:
                    weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),  # peso da aresta = distância Eclidiana no R3
                                         vertex.get_elevation(), vertex2.get_elevation())
                    if weight > self.max_edge:
                        self.max_edge = weight
                    if weight < self.min_edge:
                        self.min_edge = weight
                    vertex.add_edge(vertex2_id, weight, self)


            j1 = j - 1
            i1 = i
            if j1 >= 0:
                vertex2_id = i1 * MDEVars.GRID_COLS + j1
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
            if i1 < MDEVars.GRID_ROWS:
                vertex2_id = i1 * MDEVars.GRID_COLS + j1
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
                vertex2_id = i1 * MDEVars.GRID_COLS + j1
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
                if j1 < MDEVars.GRID_COLS and i1 < MDEVars.GRID_ROWS:
                    vertex2_id = i1 * MDEVars.GRID_COLS + j1
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
                if j1 >= 0 and i1 < MDEVars.GRID_ROWS:
                    vertex2_id = i1 * MDEVars.GRID_COLS + j1
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
                if i1 >= 0 and j1 < MDEVars.GRID_COLS:
                    vertex2_id = i1 * MDEVars.GRID_COLS + j1
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
                    vertex2_id = i1 * MDEVars.GRID_COLS + j1
                    vertex2 = self.get_vertex(vertex2_id)
                    if vertex2:
                        weight = r3_distance(vertex.get_x(), vertex2.get_x(), vertex.get_y(), vertex2.get_y(),
                                             vertex.get_elevation(), vertex2.get_elevation())
                        if weight > self.max_edge:
                            self.max_edge = weight
                        if weight < self.min_edge:
                            self.min_edge = weight
                        vertex.add_edge(vertex2_id, weight, self)
        
    
