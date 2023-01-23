import io

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

