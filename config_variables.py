import io

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
    reduction_factor = 2
    use_dnn = True
    use_observer = True

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

class GenerateVars:   

    #Configurações gerais 
    use_viewpoints = False  #Não adaptado para suportar múltiplos mapas.
    sampling_rate = 10      # % da Amostragem
    pixel_resolution = 30   #adaptar o código de geração para usar somente o dos mapas

    #Configuração para geração com viewpoints 
    vps_map_dir = ".\\maps\\vps\\"
    vps_map = MapCase(1, "\\recorte400x400.tif", 30)

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