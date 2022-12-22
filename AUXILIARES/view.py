from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from pylab import rcParams
#from matplotlib._png import read_png
import numpy as np
#import imageio
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import sys
from matplotlib.colors import LightSource,ListedColormap
from matplotlib import cm
import re
class Mde:
    # https://rasterio.readthedocs.io/en/latest/quickstart.html
    import rasterio

    # Parametros:
    # fp = nome do arquivo raster;
    # reduction_factor = grau de redução da dimensão do mapa
    def __init__(self, fp, reduction_factor):
        self.dataset = self.rasterio.open(fp)
        self.band1 = self.dataset.read(1)
        self.pixel_resolution = self.dataset.transform[0]
        self.h_limit = self.dataset.height
        self.w_limit = self.dataset.width

        # Redimensiona o grid
        self.generate_grid(reduction_factor)

        self.cell_size = self.pixel_resolution * reduction_factor
        global CELL_HEIGHT, CELL_WIDTH, GRID_ROWS, GRID_COLS, GRID_HEIGHT, GRID_WIDTH
        CELL_HEIGHT = self.pixel_resolution * reduction_factor
        CELL_WIDTH = self.pixel_resolution * reduction_factor
        GRID_COLS = self.grid.shape[0]
        GRID_ROWS = self.grid.shape[1]
        GRID_WIDTH = CELL_WIDTH * GRID_COLS
        GRID_HEIGHT = CELL_HEIGHT * GRID_ROWS

    # Gera o grid com dimensão reduzida
    # Ex: reduction_factor = 2, mapa 400x400 reduzido para 200x200
    def generate_grid(self, reduction_factor):
        x = int(self.h_limit / reduction_factor)
        y = int(self.w_limit / reduction_factor)
        self.grid = np.zeros(shape=(x, y))
        for i in range(x):
            for j in range(y):
                sub_section = self.band1[i * reduction_factor: (i + 1) * reduction_factor, j * reduction_factor: (j + 1) * reduction_factor]
                self.grid[i, j] = np.sum(sub_section)
                self.grid[i, j] = round(self.grid[i, j] / (len(sub_section) * len(sub_section[0])))

    def get_cell_size(self):
        return self.cell_size


#Recebe DEM e salva em PNG
def tif_to_png(mde, f_out):
  im = Image.fromarray(mde).convert('RGB')
  im.save(f_out)


def colorbar():
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    import matplotlib.colors

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    xdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ydata = [x * x for x in xdata]
    norm = plt.Normalize(1, 150)
    colorlist = [(0.7,0.7,0.7), (1,0,0)]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)

    cmap = newcmp
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal',
                 label="Visibilidade")


# Modifica a coloração do terreno para mostrar os nodos visíveis pelo observador (quanto menos visível menor é a saturação da tonalidade)
def aplica_viewshed_rgb(rgb, viewshed):
    for i in range(viewshed.shape[0]):
        for j in range(viewshed.shape[1]):
            fundo = rgb[i,j,:3]
            if sum(viewshed[i,j,:]) != 0:
                rgb[i, j, 0] = ((1 - fundo[0])*viewshed[i,j,0]) + fundo[0]
                rgb[i, j, 1] = ((0 - fundo[1])*viewshed[i,j,1]) + fundo[1]
                rgb[i, j, 2] = ((0 - fundo[2]) * viewshed[i, j, 2]) + fundo[2]
    return rgb

# Modifica a coloração do terreno para mostrar os nodos visitados durante a busca do caminho
def aplica_visited_rgb(rgb, visited):
    cor = [126/255, 27/255, 207/255]
    aux = []
    for i, j in visited:
        if (i, j) not in aux:
            rgb[i, j, 0] = rgb[i, j, 0] * cor[0]
            rgb[i, j, 1] = rgb[i, j, 1] * cor[1]
            rgb[i, j, 2] = rgb[i, j, 2] * cor[2]
            aux.append((i, j))
    return rgb

# Desenha a superfície a partir do PNG do terreno
def draw_surface(file, projection, alt_diff):
    plt.rcParams["figure.autolayout"] = False
    img = mpimg.imread(file)

    # ALTURA DAS COORDENADAS <- MDE (COLORAÇÃO DO PNG DO FILE)
    z = img[:, :, :-1] * alt_diff

    visited = open_csv('../DADOS_RESULTADOS/opened2.csv')
    viewshed = mpimg.imread(projection)

    # ------------------------ PLOT SUPERFICIE --------------------------- #
    x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = plt.axes(projection='3d', computed_zorder=False, proj_type='persp')

    # Projeta o terreno com uma fonte de iluminação
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(50 / 256, 0.6, N)
    vals[:, 1] = np.linspace(50 / 256, 0.6, N)
    vals[:, 2] = np.linspace(50 / 256, 0.6, N)
    newcmp = ListedColormap(vals)
    ls = LightSource(azdeg=0, altdeg=65)
    rgb = ls.shade(np.array(img[:,:,0]).reshape((200,200)), cmap=newcmp)

    # rgb = aplica_viewshed_rgb(rgb, viewshed)  # Descomentar para ilustrar a visibilidade do observador
    rgb = aplica_visited_rgb(rgb, visited)      # Descomentar para projetar os nodos abertos

    # Desenha a superfície com as colorações configuradas
    ax.plot_surface(x, y, z[x, y, 0], rstride=1, cstride=1,
                    facecolors=rgb, zorder=4.4, shade=False)
    # -------------------------------------------------------------------- #


    # ----------------------------------------- OBSERVADOR(ES) ---------------------------------------------------- #
    '''viewpoints = [(115, 126)]
    # viewpoints = []
    i = 0
    for vp_y, vp_x in viewpoints:
        i = i+1
        mk = '$' + str(i) + '$'
        mk = '$O$'
        ax.scatter(vp_y, vp_x, z[vp_y, vp_x, 0], marker='o', color='w', zorder=4.55, s=250)
        ax.scatter(vp_y, vp_x, z[vp_y, vp_x, 0], marker=mk, color='r', zorder=4.6, s=150)'''

    # ----------------------------------------------------------------------------------------------------------- #

    # ---------------------------------------- CAMINHO ---------------------------------------------------------- #
    # marca as coordenadas percorridas
    path = open_csv('../DADOS_RESULTADOS/visited2.csv')
    # path = []
    # Pontos do caminho
    for i, cell in enumerate(path):
        if i == 0:
            ax.scatter(cell[0], cell[1], z[cell[0], cell[1], 0], marker='o', c='w', zorder=4.5, s=200)
            ax.scatter(cell[0], cell[1], z[cell[0], cell[1], 0], marker='$G$', color='b', zorder=4.55, s=100)
        elif i == len(path) -1:
            ax.scatter(cell[0], cell[1], z[cell[0], cell[1], 0], marker='o', c='w', zorder=4.5, s=200)
            ax.scatter(cell[0], cell[1], z[cell[0], cell[1], 0], marker='$S$', color='b', zorder=4.55, s=100)
            continue
        else:
            ax.scatter(cell[0], cell[1], z[cell[0], cell[1], 0], marker='o', c='c', zorder=4.49, s=10)
        x1,y1 = path[i]
        x2,y2 = path[i+1]
        plt.plot([x1,x2],[y1,y2],'b')
    # ----------------------------------------------------------------------------------------------------------- #

    #ax.view_init(90, 180)
    ax.view_init(75, 135)
    #ax.view_init(75, 170)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    plt.axis('off')
        
    ax.grid(False)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.xlabel('X', fontsize=16, labelpad=16)
    plt.ylabel('Y', fontsize=16, labelpad=16)

    # Set general font size
    #plt.rcParams['font.size'] = '16'

    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontsize(16)

    #colorbar()

    plt.show()

from csv import reader

def open_csv(f):
  with open(f, 'r') as read_obj:
      list_of_rows = []
      csv_reader = reader(read_obj)
      for l in csv_reader:
        if len(l) == 0:
          continue
        l = [float(l) for l in re.findall(r'-?\d+\.?\d*', str(l))]
        l = list(map(int, l))
        list_of_rows.append(l)
      return list_of_rows

def main():
    args = sys.argv

    filename = args[1]
    viewshed = '../VIEWSHEDS/VIEWSHED_4_11.png'

    #Dimensão em pixels da area do nodo = reduction_factor X reduction_factor
    reduction_factor = int(args[2])
    mde = Mde(filename, reduction_factor)

    max_alt = mde.grid.max()
    min_alt = mde.grid.min()
    diff = max_alt - min_alt

    mde.grid = (mde.grid - min_alt) * (255/diff)

    # SALVA COLORAÇÃO DO TERRENO
    # O VALOR DO PIXEL DENOTA A ELEVAÇÃO (eixo z)
    tif_to_png(np.array(mde.grid), 'terrain.png')

    # Desenha a projeção do terreno
    draw_surface('terrain.png', viewshed, diff)

if __name__ == '__main__':
    main()