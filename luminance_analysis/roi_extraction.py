import numpy as np
import deepdish as dd
from skimage import io
from pathlib import Path
from skimage.external.tifffile import imsave as tiffimsave
from fimpy.core.split_dataset import H5SplitDataset
from processing.roi_extraction import correlation_map, correlation_flooding, extract_traces
from processing.roi_extraction import get_ROI_coords_areas_traces_3D


settings = dict(
    corr_threshold=0.45,
    max_labels=300,
    max_corr_threshold=0.6,
    max_radius=4,
    corr_threshold_steps=5,
    max_investigate=500,
    min_area=10,
    across_planes=True,
    voxel_size=(1, 0.6, 0.6))

t_final = 500

if __name__ == "__main__":
    master_path = Path(r"J:\_Shared\experiments\E0032_luminance\ECs")
    for fish_path in master_path.glob("*f[0-9]"):
        print("working on {}".format(fish_path.name))
        dataset = H5SplitDataset(str(fish_path / "src"))
        substack = dataset[:, :, :, :]
        mask = io.imread(next(fish_path.glob("*mask*")))
        substack = substack * np.concatenate([mask[np.newaxis, :, :, :]]*substack.shape[0], 0)

        cmap = io.imread(next(fish_path.glob("*corrmap*")))
        rois = correlation_flooding(substack[:t_final, :, :, :], cmap, **settings)
        print("found ROIs: {}".format(rois.max()))

        coords, sizes, traces = get_ROI_coords_areas_traces_3D(substack, rois)
        data_dict = dict(stack=rois, traces=traces, coords=coords, sizes=sizes)

        dd.io.save(str(fish_path / "python_rois.hdf5"), data_dict)
        tiffimsave(str(fish_path / "python_rois_stack.tif"), rois.astype(np.int16))

    #
    # for fish_path in master_path.glob("*f[0-9]"):
    #     print("working on {}".format(fish_path.name))
    #     dataset = load_split_dataset(next(fish_path.glob("*/aligned")))
    #     substack = dataset[:, :, :, :]
    #     rois = np.swapaxes(io.imread(str(next(fish_path.glob("*manual_rois.tif*")))),
    #                        1, 2)
    #
    #     coords, sizes, traces = get_ROI_coords_areas_traces_3D(substack, rois)
    #     data_dict = dict(stack=rois, traces=traces, coords=coords, sizes=sizes)
    #
    #     dd.io.save(str(fish_path / "python_rois.hdf5"), data_dict)
    #     # tiffimsave(str(fish_path / "python_cmap.tif"), cmap.astype(np.uint16))

