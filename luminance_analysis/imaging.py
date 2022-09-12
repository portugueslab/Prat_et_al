import numpy as np

def compute_resolution(zoom, size_px):
    """ Function to calculate resolution from Labview 2p software data.
    Zoom and size in pixel must refer to the same axis
    :param zoom: zoom parameter from Labview software
    :param size: image size on matching axis
    :return:  Pixel size in microns
    """
    # Calibration data:
    dist_in_um = 10
    dist_in_px = np.array([21.13, 19.62, 8.93])
    zooms = np.array([1.5, 3, 4.5])
    image_max_sizes = np.array([330, 610, 410])
        
    return np.mean((dist_in_um/dist_in_px) * (zoom/zooms) * (image_max_sizes/size_px))