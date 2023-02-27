import io
import math
import random

# ----- ----- Teste (Classe auxiliar)----- ----- #
class TestCase:
    def __init__(self, name, file_suffix, algorithm, heuristic, create_heuristic_map = False):
        self.name = name
        self.file_suffix = file_suffix
        self.algorithm = algorithm
        self.heuristic = heuristic
        self.create_heuristic_map = create_heuristic_map
        self.distance = 0 
        self.count_visited = 0
        self.count_open = 0
        self.opened = []
        self.visited = []
        self.cost = 0
        self.path = 0 
        self.count_visible = 0
        self.time = 0
        self.data_io = io.StringIO()
        self.data_io_visited = io.StringIO()
        self.data_io_opened = io.StringIO()
        
# ----- ----- Teste ----- ----- #
class TestVars:
    filename = ".\\recorte400x400.tif"
    dnn_model_name = "modelo_249_epocas.hdf5"
    dnn_model_with_observer_name = "model_100_10.hdf5"
    reduction_factor = 1
    use_dnn = True
    use_observer = True
    test = True

# Casos de Teste
    test_cases = [TestCase("Astar", "astar", "astar", "r3_heuristic"),
        TestCase("Astar (com campo de visão)", "astar_sight", "astarmod", "r3_heuristic"),
        TestCase("Theta", "theta", "theta", "r3_heuristic"),
        TestCase("Astar (com DNN)", "astar_dnn", "astar", "dict_dnn_heuristic", create_heuristic_map=True)]

# ----- ----- Generate dataset ----- ----- #

class MapCase:
    def __init__(self, id_map, filename, pixel_resolution, reduction_factor = 1):
        self.id_map = id_map
        self.filename = filename
        self.pixel_resolution = pixel_resolution #adaptar o código de geração para suportar
        self.reduction_factor = reduction_factor #adaptar o código de geração para suportar

class VPCase:
    def __init__(self, id_vp, x, y):
        self.id_vp = id_vp
        self.point = (x, y)

class GenerateVars:   

    #Configurações gerais 
    use_viewpoints = False  #Não adaptado para suportar múltiplos mapas.
    sampling_rate = 10      # % da Amostragem
    pixel_resolution = 30   #adaptar o código de geração para usar somente o dos mapas
    type_dataset = 2 #Configuração de dataset (custo verdadeiro = 1,taxa de erro = 2)

    #Configuração para geração com viewpoints 
    vps_map_dir = ".\\maps\\vps\\"
    vps_map = MapCase(1, "\\recorte400x400_3.tif", 30)
    
    #Configuração para geração com vpconfigs
    viewpoints = [
        VPCase(1, 47, 180),
        VPCase(2, 121, 149),
        VPCase(4, 191, 215),
        VPCase(8, 232, 169),
        VPCase(16, 241, 56),
        VPCase(32, 350, 109),
        VPCase(64, 338, 205), #234, 399
        VPCase(128, 269, 317),
        VPCase(256, 356, 346),
        VPCase(512, 115, 255),
        VPCase(1024, 73, 349),
        VPCase(2048, 172, 323),
        VPCase(4096, 143, 52),
        VPCase(8192, 50, 50)
    ]
    vpconfigs = [22,28,32,52,64,81,104,112,128,179,194,256,386,400,517,772,1024,1036,1281,1344,1552,2048,2568,2581,2641,3680,4096,4164,4228,4249,4293,4396,4433,4646,4756,5444,5672,6272,7236,8192,8245,8260,8376,8467,8474,9864,11332,11584] #codigo de id
    
        
    #Configuração para geração com múltiplos mapas (sem viewpoints) 
    maps_dir = ".\\maps\\novps\\"
    maps = [
        MapCase(1, "\\recorte300x300A.tif", 30),
        MapCase(2, "\\recorte300x300B.tif", 30),
        MapCase(3, "\\recorte300x300C.tif", 30),
        MapCase(4, "\\recorte300x300D.tif", 30),
        MapCase(5, "\\recorte300x300E.tif", 30),
        MapCase(6, "\\recorte300x300F.tif", 30),
        MapCase(7, "\\recorte300x300G.tif", 30),
        MapCase(8, "\\recorte300x300H.tif", 30)
    ]    
    files_dir = ".\\dataset_sem_observador_mapa_\\"
    
    #Funções
    def random_vpconfigs(count_one, count_three, count_five):
        configs = []
        vps = GenerateVars.viewpoints.copy()
        
        for i in range(0, count_one):
            configs.append(vps.pop(random.randint(0, len(vps)-1)).id_vp)
            
        
        for i in range(0, count_three):
            vps = GenerateVars.viewpoints.copy()
            temp = 0
            for j in range(0, 3):
                temp = temp + vps.pop(random.randint(0, len(vps)-1)).id_vp
            configs.append(temp)
            
        for i in range(0, count_five):
            vps = GenerateVars.viewpoints.copy()
            temp = 0
            for j in range(0, 5):
                temp = temp + vps.pop(random.randint(0, len(vps)-1)).id_vp
            configs.append(temp)
        return configs