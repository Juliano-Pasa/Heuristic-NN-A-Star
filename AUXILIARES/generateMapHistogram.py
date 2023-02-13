import rasterio
from rasterio.plot import *
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    mapSuffixes = ["1", "2", "3", "4"]
    mapSuffixess = ["A"]

    for mapSuffix in mapSuffixes:
        fileName = f"C:/Users/Thiago/Desktop/research/TopoPath_DNN/maps/vps/recorte400x400_{mapSuffix}.tif"
        #fileName = "./maps/vps/recorte400x400_3.tif"
        #title = "Map aaa Histogram"
        title = f"Map {mapSuffix} Histogram"
        #outputFileName = "histMapaaaaa.png"
        outputFileName = f"histMapa{mapSuffix}.png"

        image = rasterio.open(fileName)
        uniqueValues = len(np.unique(image.read(1)))

        print(f"Unique Values: {uniqueValues}")
        show_hist(image, bins=uniqueValues, title=title, label=outputFileName)
