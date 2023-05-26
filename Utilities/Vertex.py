import math

from Utilities.Heuristics import r3_distance
from config_variables import MDEVars

class Vertex:
    def __init__(self, elevation, node_id):
        self.local_risk = 0
        self.elevation = elevation
        self.id = node_id
        self.edges = {}
        self.angles = {}
        self.distance = 99999999    # Somatório da distância percorrida da origem até o vértice
        self.risk = 99999999    # Somatório do grau de visibilidade da origem até o vértice
        self.previous = None
        self.visited = False
        self.visitedReverse = False
        self.j = self.get_j()
        self.i = self.get_i()
        self.computed = False
        self.graph = None

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
        return self.j * MDEVars.CELL_WIDTH

    def get_y(self):
        return self.i * MDEVars.CELL_HEIGHT
    def get_i(self):
        return math.floor(self.id / MDEVars.GRID_ROWS)

    def get_j(self):
        return self.id % MDEVars.GRID_COLS

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

    # Reseta os valores do vértice para computar outro caminho utilizando o A*
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
    return math.degrees(math.asin(seno))