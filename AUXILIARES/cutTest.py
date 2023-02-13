import rasterio
from rasterio.windows import Window

with rasterio.open('../maps/vps/options/4/tif.tif') as src:

    print("xsize: ", src.width, " ysize: ", src.height)

    # The size in pixels of your desired window
    xsize, ysize = 400, 400

    # Generate a random window origin (upper left) that ensures the window 
    # doesn't go outside the image. i.e. origin can only be between 
    # 0 and image width or height less the window width or height
    xmin, xmax = 0, src.width - xsize
    ymin, ymax = 0, src.height - ysize
    # xoff, yoff = random.randint(xmin, xmax), random.randint(ymin, ymax)

    # Create a Window and calculate the transform from the source dataset    
    window = Window(0, 0, xsize, ysize)
    transform = src.window_transform(window)

    # Create a new cropped raster to write to
    profile = src.profile
    profile.update({
        'height': xsize,
        'width': ysize,
        'transform': transform})

    with rasterio.open('recorte400x400_4.tif', 'w', **profile) as dst:
        # Read the data from the window and write it to the output raster
        dst.write(src.read(window=window))