import multiprocessing as mp
import time
import numpy as np
import shelve
import argparse


def process_chunk(start_idx, end_idx):
    mesh = np.meshgrid(
        np.arange(
            cam_size_y + 1),
        np.arange(
            cam_size_x + 1),
        indexing='ij')
    y_mesh = mesh[0].ravel()
    x_mesh = mesh[1].ravel()
    grayscale = np.zeros((cam_size_y + 1, cam_size_x + 1), dtype=np.uint8)
    hist = np.zeros(
        (cam_size_y + 1,
         cam_size_x + 1,
         hist_size),
        dtype=np.int64)

    for image_idx in range(start_idx, end_idx):
        generic_image_id = image_idx
        cam = cam_map[f"{image_idx}"]

        # NOTE: 1-indexed now
        grayscale[1:, 1:] = np.uint8(255 * cam)
        bins_for_grayscale = grayscale.ravel() // bin_width
        hist.fill(0)
        hist[y_mesh, x_mesh, bins_for_grayscale] = 1
        full_prefix = np.cumsum(np.cumsum(hist, axis=0), axis=1)
        hist_prefix[generic_image_id][:] = full_prefix[0:cam_size_y + \
            1:available_coords, 0:cam_size_x + 1:available_coords, :]

        # hist_prefix itself is now reverse cumulative
        hist_prefix[generic_image_id] = np.cumsum(
            hist_prefix[generic_image_id, :, :, ::-1], axis=2)[:, :, ::-1]


if __name__ == '__main__':

    available_coords = 14
    hist_size = 16

    print(
        f"Building index for imagenet with available_coords: {available_coords} and hist_size: {hist_size}")

    start = time.time()
    hist_edges = []
    bin_width = 256 // hist_size
    for i in range(1, hist_size):
        hist_edges.append(bin_width * i)
    hist_edges.append(256)
    print(hist_edges)
    cam_size_x = 224
    cam_size_y = 224

    total_images = 1331167
    cam_map = shelve.open(
        '/data/explain_imagenet/shelves/imagenet_cam_map.shelve')

    filename = f'/data/explain_imagenet/npy/trial_imagenet_cam_hist_prefix_{hist_size}_available_coords_{available_coords}_memmap_suffix.npy'
    hist_prefix = np.memmap(
        filename,
        dtype=np.int64,
        mode='w+',
        shape=(
            total_images,
            (cam_size_y // available_coords) + 1,
            (cam_size_x // available_coords) + 1,
            hist_size))

    num_processes = mp.cpu_count()
    chunk_size = total_images // num_processes
    processes = []

    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        if i == num_processes - 1:
            end_idx = total_images
        p = mp.Process(target=process_chunk, args=(start_idx, end_idx))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    cam_map.close()
    hist_prefix.flush()

    end = time.time()
    print(f"Index building time for imagenet: {end - start}")
