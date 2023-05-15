# %%
# see https://github.com/microsoft/vscode-jupyter/issues/1837 for sys.argv
# = [''] below
import random
import sys
import os
from masksearch import imagenet_mp_vars
from masksearch.masksearch import *
import time

sys.argv = ['']
imagenet_mp_vars.init()
dataset_name = "imagenet"

# %%
imagenet_mp_vars.init_load_index()

# %%
# Filter query benchmark setup (static region)

toy_examples = [str(i) for i in range(len(imagenet_mp_vars.dp))]
region = (50, 50, 200, 200)
threshold = 0.6
v = 5000
region_area_threshold = 10000
grayscale_threshold = int(threshold * 255)

# %%
# MaskSearch (vanilla): filter query
start = time.time()
count, area_images = get_images_based_on_area_filter(dataset_name, imagenet_mp_vars.cam_map, imagenet_mp_vars.object_detection_map, imagenet_mp_vars.bin_width, imagenet_mp_vars.hist_size, imagenet_mp_vars.cam_size_y, imagenet_mp_vars.cam_size_x, toy_examples,
                                                     threshold, region, v, imagenet_mp_vars.in_memory_index_suffix, region_area_threshold=region_area_threshold, ignore_zero_area_region=True, reverse=False, visualize=False, available_coords=imagenet_mp_vars.available_coords, compression=None)
end = time.time()
print("(MaskSearch vanilla) Query time (cold cache):", end - start)

# %%
# MaskSearch (+ MP): filter query

start = time.time()
count, area_images = mp_get_images_based_on_area_filter(dataset_name, toy_examples, threshold, region, v, processes=8,
                                                        region_area_threshold=region_area_threshold, ignore_zero_area_region=True, reverse=False, visualize=False, compression=None)
end = time.time()
print("(MaskSearch + MP) Query time (cold cache):", end - start)

# %%
# Naive: filter query

start = time.time()
area_images = naive_get_images_satisfying_filter(
    imagenet_mp_vars.cam_map,
    imagenet_mp_vars.object_detection_map,
    imagenet_mp_vars.cam_size_y,
    imagenet_mp_vars.cam_size_x,
    toy_examples,
    threshold,
    region,
    v,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    compression=None)
end = time.time()
print("(Naive) Query time (cold cache):", end - start)
