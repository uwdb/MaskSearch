# %%
# NOTE: see https://github.com/microsoft/vscode-jupyter/issues/1837 for
# sys.argv = [''] below
import time
from masksearch.masksearch import *
from masksearch import wilds_mp_vars
import sys
sys.argv = ['']
sys.argv = ['']
wilds_mp_vars.init()
dataset_name = "wilds"

# %%
wilds_mp_vars.init_load_index()

# %%
# Filter query benchmark setup

toy_examples = []
for distribution, image_total in zip(
        ["id_val", "ood_val"], [wilds_mp_vars.id_total, wilds_mp_vars.ood_total]):
    for image_idx in range(1, 1 + image_total):
        toy_examples.append(f"{distribution}_{image_idx}")

region = (50, 50, 150, 150)
threshold = 0.6
v = 10000
region_area_threshold = None
grayscale_threshold = int(threshold * 255)

# %%
# MaskSearch (vanilla): filter query

start = time.time()
count, area_images = get_images_based_on_area_filter("wilds", wilds_mp_vars.cam_map, wilds_mp_vars.object_detection_map, wilds_mp_vars.bin_width, wilds_mp_vars.hist_size, wilds_mp_vars.cam_size_y, wilds_mp_vars.cam_size_x, toy_examples, threshold,
                                                     region, v, wilds_mp_vars.in_memory_index_suffix, region_area_threshold=region_area_threshold, ignore_zero_area_region=True, reverse=False, visualize=False, available_coords=wilds_mp_vars.available_coords, compression=None)  # type: ignore
end = time.time()
print("(MaskSearch vanilla) Query time (cold cache):", end - start)

# %%
# MaskSearch (MP): filter query

start = time.time()
count, area_images = mp_get_images_based_on_area_filter(dataset_name, toy_examples, threshold, region, v, processes=4,
                                                        region_area_threshold=region_area_threshold, ignore_zero_area_region=True, reverse=False, visualize=False, compression=None)
end = time.time()
print("(MaskSearch MP) Query time (cold cache):", end - start)

# %%
# Naive: filter query

start = time.time()
area_images = naive_get_images_satisfying_filter(
    wilds_mp_vars.cam_map,
    wilds_mp_vars.object_detection_map,
    wilds_mp_vars.cam_size_y,
    wilds_mp_vars.cam_size_x,
    toy_examples,
    threshold,
    region,
    v,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    compression=None)
end = time.time()
print("(NumPy) Query time (cold cache):", end - start)
