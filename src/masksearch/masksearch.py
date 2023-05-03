import multiprocessing as mp
from masksearch import wilds_mp_vars
from masksearch import imagenet_mp_vars
from multiprocessing import Pool
import heapq
import shelve
from statistics import mean
import sys
import time
from matplotlib import patches, pyplot as plt

import numpy as np
import torch
import torchvision
import cv2

from pytorch_grad_cam.utils.image import show_cam_on_image


def get_generic_image_id_for_wilds(distribution, idx):
    if distribution == "id_val" or distribution == "id":
        return int(idx)
    elif distribution == "ood_val" or distribution == "ood":
        return int(idx) + 7314
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def get_object_region(object_detection_map, cam_size_y, cam_size_x, image_idx):
    """Returns the object region for the given image_id"""
    try:
        detection = object_detection_map[image_idx]
        minx = max(int(detection[0]), 0)
        miny = max(int(detection[1]), 0)
        maxx = min(int(detection[2]), cam_size_x)
        maxy = min(int(detection[3]), cam_size_y)
        return minx, miny, maxx - minx, maxy - miny
    except KeyError:
        return 0, 0, 0, 0


def load_object_region_index_in_memory(dataset_examples, filename):
    object_region_index = {}
    object_detection_shelve = shelve.open(filename)
    count = 0
    for example in dataset_examples:
        try:
            object_region_index[example] = object_detection_shelve[example]
        except BaseException:
            count += 1
    print("Number of examples not found in object detection map:", count)
    object_detection_shelve.close()
    return object_region_index


def compute_area_for_cam(cam, threshold=0.6, subregion=None):
    if subregion is not None:
        x, y, w, h = subregion
        grid = cam[y: y + h, x: x + w]
    else:
        grid = cam
    grid = grid > threshold
    area = np.count_nonzero(grid)
    return area


def imagenet_random_access_images(dp, image_idx_list, batch_size=16):
    """ returns a dict, key: image_idx, value: image"""
    res = dict()
    subset = dp["input"][image_idx_list]
    dataloader = subset.batch(
        batch_size=batch_size,
        num_workers=8,
        shuffle=False)
    indices_idx = -1
    for batch_data in dataloader:
        x = batch_data.data
        for i in range(x.shape[0]):
            indices_idx += 1
            res[image_idx_list[indices_idx]] = x[i]
    return res


def wilds_random_access_images(
        id_val_data,
        ood_val_data,
        image_idx_list,
        batch_size=16):
    """ returns a dict, key: image_idx, value: image"""
    indices = {"id": [], "ood": []}
    for image_idx in image_idx_list:
        idx = int(image_idx.split("_")[-1].strip()) - 1
        distribution = image_idx.split("_")[0].strip()
        indices[distribution].append(idx)
    id_subset = torch.utils.data.Subset(id_val_data, indices["id"])
    ood_subset = torch.utils.data.Subset(ood_val_data, indices["ood"])

    res = dict()
    for distribution, subset in zip(["id", "ood"], [id_subset, ood_subset]):
        dataloader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=False)
        indices_idx = -1
        for x, y_true, metadata in dataloader:
            for i in range(x.shape[0]):
                indices_idx += 1
                res[f"{distribution}_val_{indices[distribution][indices_idx] + 1}"] = x[i]
    return res


def from_input_to_image(image):
    if isinstance(image, torch.Tensor):
        image = torchvision.utils.make_grid(
            image.cpu().data, normalize=True).numpy()
    image = np.transpose(image, (1, 2, 0))
    return image


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def get_area_map(
        cam_map,
        object_detection_map,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        compression):
    area_map = dict()
    if isinstance(region, tuple):
        x, y, w, h = region

    count = 0
    for image_idx in examples:
        if not isinstance(region, tuple):
            (x, y, w, h) = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_idx)
        cam = cam_map[image_idx]
        if compression is not None:
            if compression == "jpeg" or compression == "JPEG" or compression == "png" or compression == "PNG":
                cam = np.frombuffer(cam, dtype=np.uint8)
                cam = cv2.imdecode(cam, cv2.IMREAD_COLOR)
                cam = cam[:, :, 0]
            else:
                raise ValueError("Unknown compression method")
            cam = cam.reshape((cam_size_y, cam_size_x))
            cam = np.float32(cam) / 255.0

        area = compute_area_for_cam(cam, threshold, (x, y, w + 1, h + 1))
        area_map[image_idx] = area
        count += 1

    return area_map


def naive_get_max_metric(
        dataset,
        dp,
        cam_map,
        object_detection_map,
        label_map,
        pred_map,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        k,
        region_area_threshold,
        ignore_zero_area_region,
        compression=None,
        reverse=False,
        visualize=False):
    start = time.time()
    metric_map = get_area_map(
        cam_map,
        object_detection_map,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        compression)
    tot = len(examples)
    metric_list = []
    for i in range(tot):
        image_id = examples[i]
        if isinstance(region, tuple):
            x, y, w, h = region
        else:
            (x, y, w, h) = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_id)

        box_area = w * h
        if (ignore_zero_area_region and box_area <= 0) or (
                region_area_threshold is not None and box_area < region_area_threshold):
            if reverse:
                metric = int(1e15)
            else:
                metric = -1
        else:
            metric = metric_map[image_id] / box_area

        metric_list.append(metric)
    order = argsort(metric_list)

    if not reverse:
        order = order[::-1]

    end = time.time()

    tot = min(tot, k)

    if not visualize:
        return sorted([(metric_list[i], examples[i])
                      for i in order[:tot]], reverse=not reverse)

    if dataset == "imagenet":
        image_map = imagenet_random_access_images(
            dp, [examples[order[j]] for j in range(tot)])
    else:
        image_map = wilds_random_access_images(
            dp[0], dp[1], [examples[order[j]] for j in range(tot)])

    cnt = 0
    plt.figure(figsize=(8, 10))
    for j in range(tot):
        i = order[j]
        metric = metric_list[i]
        assert metric >= 0 and metric < 1e15
        cnt += 1
        image_id = examples[i]
        image = image_map[image_id].reshape(3, 224, 224)
        cam = cam_map[image_id]
        if isinstance(region, tuple):
            x, y, w, h = region
        else:
            (x, y, w, h) = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_id)

        ax = plt.subplot((tot + 4) // 5, 5, cnt)
        plt.xticks([], [])
        plt.yticks([], [])

        image = from_input_to_image(image)
        cam_image = show_cam_on_image(image, cam, use_rgb=True)
        plt.imshow(cam_image)
        plt.title("{}->{}".format(label_map[image_id], pred_map[image_id]))
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=5, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()
    plt.clf()

    return sorted([(metric_list[order[j]], examples[order[j]])
                  for j in range(tot)], reverse=not reverse)


def get_approximate_region_using_available_coords(
        cam_size_y, cam_size_x, reverse, available_coords, x, y, w, h):
    if not reverse:
        # Overapproximate area
        lower_y, upper_y, lower_x, upper_x = get_smallest_region_covering_roi_using_available_coords(
            cam_size_y, cam_size_x, available_coords, x, y, w, h)
    else:
        # Underapproximate area
        lower_y, upper_y, lower_x, upper_x = get_largest_region_covered_by_roi_using_available_coords(
            cam_size_y, cam_size_x, available_coords, x, y, w, h)
    return lower_y, upper_y, lower_x, upper_x


def minimum_bounding_box(bounding_boxes):
    # Find the minimum and maximum x and y coordinates of all bounding boxes
    min_x = min(box[0] for box in bounding_boxes)
    min_y = min(box[1] for box in bounding_boxes)
    max_x = max(box[0] + box[2] for box in bounding_boxes)
    max_y = max(box[1] + box[3] for box in bounding_boxes)
    print(min_x, min_y, max_x, max_y)

    # Calculate the width and height of the minimum bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Return the minimum bounding box as a tuple
    return (min_x, min_y, width, height)


def get_smallest_region_covering_roi_using_available_coords(
        cam_size_y, cam_size_x, available_coords, x, y, w, h):
    lower_y = ((y - 1) // available_coords)
    upper_y = min(((y + h + available_coords - 1) // available_coords)
                  * available_coords, cam_size_y) // available_coords
    lower_x = ((x - 1) // available_coords)
    upper_x = min(((x + w + available_coords - 1) // available_coords)
                  * available_coords, cam_size_x) // available_coords
    return lower_y, upper_y, lower_x, upper_x


def get_largest_region_covered_by_roi_using_available_coords(
        cam_size_y, cam_size_x, available_coords, x, y, w, h):
    lower_y = min(((y - 1 + available_coords - 1) // available_coords)
                  * available_coords, cam_size_y) // available_coords
    upper_y = ((y + h) // available_coords)
    lower_x = min(((x - 1 + available_coords - 1) // available_coords)
                  * available_coords, cam_size_x) // available_coords
    upper_x = ((x + w) // available_coords)
    return lower_y, upper_y, lower_x, upper_x


def update_max_area_images_in_sub_region_in_memory_version(
        dataset,
        heap,
        cam_map,
        object_detection_map,
        bin_width,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        k,
        region_area_threshold,
        ignore_zero_area_region,
        reverse,
        in_memory_index_suffix,
        available_coords,
        compression,
        image_access_order,
        early_stoppable):
    """returns a list of (metric, area, image_idx)"""

    if region != "object":
        x, y, w, h = region
        x += 1
        y += 1
        lower_y, upper_y, lower_x, upper_x = get_approximate_region_using_available_coords(
            cam_size_y, cam_size_x, reverse, available_coords, x, y, w, h)

    grayscale_threshold = int(threshold * 255)
    tot = len(image_access_order)
    if reverse:
        factor = -1
    else:
        factor = 1

    count = 0
    for offset in range(tot):
        i = image_access_order[offset]
        image_idx = examples[i]
        if region == "object":
            x, y, w, h = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_idx)
            x += 1
            y += 1
        if ignore_zero_area_region and (w == 0 or h == 0):
            count += 1
            continue
        box_area = w * h
        if region_area_threshold is not None and box_area < region_area_threshold:
            count += 1
            continue
        if region == "object":
            lower_y, upper_y, lower_x, upper_x = get_approximate_region_using_available_coords(
                cam_size_y, cam_size_x, reverse, available_coords, x, y, w, h)

        if dataset == "imagenet":
            generic_image_id = int(image_idx)
        else:
            generic_image_id = get_generic_image_id_for_wilds(
                image_idx.split("_")[0], image_idx.split("_")[-1])
        hist_prefix_suffix = in_memory_index_suffix[generic_image_id][:]
        hist_suffix_sum = hist_prefix_suffix[upper_y,
                                             upper_x] - hist_prefix_suffix[upper_y,
                                                                           lower_x] - hist_prefix_suffix[lower_y,
                                                                                                         upper_x] + hist_prefix_suffix[lower_y,
                                                                                                                                       lower_x]

        if reverse:
            approximate_area = hist_suffix_sum[(
                grayscale_threshold // bin_width) + 1]
        else:
            approximate_area = hist_suffix_sum[grayscale_threshold // bin_width]

        if len(heap) < k or factor * approximate_area / box_area > heap[0][0]:
            cam = cam_map[image_idx]
            if compression is not None:
                if compression == "jpeg" or compression == "JPEG" or compression == "png" or compression == "PNG":
                    cam = np.frombuffer(cam, dtype=np.uint8)
                    cam = cv2.imdecode(cam, cv2.IMREAD_COLOR)
                    cam = cam[:, :, 0]
                else:
                    raise ValueError("Unknown compression method")
                cam = cam.reshape((cam_size_y, cam_size_x))
                cam = np.float32(cam) / 255.0
            area = compute_area_for_cam(
                cam, threshold=threshold, subregion=(
                    x - 1, y - 1, w + 1, h + 1))
            if len(heap) < k:
                heapq.heappush(
                    heap, (factor * area / box_area, factor * area, image_idx))
            elif factor * area / box_area > heap[0][0]:
                heapq.heappushpop(
                    heap, (factor * area / box_area, factor * area, image_idx))
        else:
            if early_stoppable:
                count = tot - offset
                break
            else:
                count += 1
    return count


def get_max_area_in_subregion_in_memory_version(
        dataset,
        dp,
        label_map,
        pred_map,
        cam_map,
        object_detection_map,
        bin_width,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        in_memory_index_suffix,
        image_access_order,
        early_stoppable,
        k=25,
        region_area_threshold=None,
        ignore_zero_area_region=True,
        reverse=False,
        visualize=False,
        available_coords=None,
        compression=None):

    area_images = []
    count = update_max_area_images_in_sub_region_in_memory_version(
        dataset,
        area_images,
        cam_map,
        object_detection_map,
        bin_width,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        k,
        region_area_threshold,
        ignore_zero_area_region,
        reverse,
        in_memory_index_suffix,
        available_coords,
        compression,
        image_access_order,
        early_stoppable)
    if reverse:
        factor = -1
    else:
        factor = 1
    area_images = [(factor * metric, factor * area, image_idx)
                   for (metric, area, image_idx) in area_images]
    area_images = sorted(area_images, key=lambda x: x[0], reverse=not reverse)

    if not visualize:
        return count, area_images

    cnt = 0
    plt.figure(figsize=(8, 10))
    tot = len(area_images)

    if isinstance(region, tuple):
        x, y, w, h = region

    if dataset == "imagenet":
        image_map = imagenet_random_access_images(
            dp, [image_idx for (metric, area, image_idx) in area_images])
    else:
        image_map = wilds_random_access_images(
            dp[0], dp[1], [
                image_idx for (
                    metric, area, image_idx) in area_images])

    for j in range(tot):
        cnt += 1
        metric, area, image_id = area_images[j]
        if not isinstance(region, tuple):
            x, y, w, h = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_id)

        if w == 0 or h == 0:
            continue

        image = image_map[image_id].reshape(3, 224, 224)
        cam = cam_map[image_id]

        ax = plt.subplot((tot + 4) // 5, 5, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        image = from_input_to_image(image)
        cam_image = show_cam_on_image(image, cam, use_rgb=True)
        plt.imshow(cam_image)
        plt.title("{}->{}".format(label_map[image_id], pred_map[image_id]))
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=5, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

    return count, area_images


# CP(mask, region, pixel value > threshold) > v
def get_images_satisfying_filter(
        dataset,
        cam_map,
        object_detection_map,
        bin_width,
        hist_size,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        v,
        region_area_threshold,
        ignore_zero_area_region,
        in_memory_index_suffix,
        available_coords,
        compression):
    """returns a list of (metric, area, image_idx)"""

    if region != "object":
        x, y, w, h = region
        x += 1
        y += 1
        lower_ys, upper_ys, lower_xs, upper_xs = get_smallest_region_covering_roi_using_available_coords(
            cam_size_y, cam_size_x, available_coords, x, y, w, h)
        lower_yl, upper_yl, lower_xl, upper_xl = get_largest_region_covered_by_roi_using_available_coords(
            cam_size_y, cam_size_x, available_coords, x, y, w, h)

    grayscale_threshold = int(threshold * 255)
    tot = len(examples)

    count = np.array([0, 0, 0], dtype=np.int32)
    res = []

    overapprox_bin = grayscale_threshold // bin_width
    underapprox_bin = (grayscale_threshold // bin_width) + 1
    if underapprox_bin == hist_size:
        trivial_underapprox = True
        theta_underline = 0
    else:
        trivial_underapprox = False

    for offset in range(tot):
        i = offset
        image_idx = examples[i]
        if region == "object":
            x, y, w, h = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_idx)
            x += 1
            y += 1

        if ignore_zero_area_region and (w == 0 or h == 0):
            count[1] += 1
            continue
        box_area = w * h
        if region_area_threshold is not None and box_area < region_area_threshold:
            count[0] += 1
            continue

        if region == "object":
            lower_ys, upper_ys, lower_xs, upper_xs = get_smallest_region_covering_roi_using_available_coords(
                cam_size_y, cam_size_x, available_coords, x, y, w, h)
            lower_yl, upper_yl, lower_xl, upper_xl = get_largest_region_covered_by_roi_using_available_coords(
                cam_size_y, cam_size_x, available_coords, x, y, w, h)

        if dataset == "imagenet":
            generic_image_id = int(image_idx)
        else:
            generic_image_id = get_generic_image_id_for_wilds(
                image_idx.split("_")[0], image_idx.split("_")[-1])
        hist_prefix_suffix = in_memory_index_suffix[generic_image_id][:]

        hist_suffix_sum_smallest_covering_roi = hist_prefix_suffix[upper_ys,
                                                                   upper_xs] - hist_prefix_suffix[upper_ys,
                                                                                                  lower_xs] - hist_prefix_suffix[lower_ys,
                                                                                                                                 upper_xs] + hist_prefix_suffix[lower_ys,
                                                                                                                                                                lower_xs]
        hist_suffix_sum_largest_covered_by_roi = hist_prefix_suffix[upper_yl,
                                                                    upper_xl] - hist_prefix_suffix[upper_yl,
                                                                                                   lower_xl] - hist_prefix_suffix[lower_yl,
                                                                                                                                  upper_xl] + hist_prefix_suffix[lower_yl,
                                                                                                                                                                 lower_xl]

        theta_bar_1 = hist_suffix_sum_smallest_covering_roi[overapprox_bin]
        theta_bar_2 = hist_suffix_sum_largest_covered_by_roi[overapprox_bin] + box_area - (
            upper_yl - lower_yl) * (upper_xl - lower_xl) * available_coords * available_coords
        theta_bar = min(theta_bar_1, theta_bar_2)

        if theta_bar <= v:
            count[1] += 1
        else:
            # Compute theta_underline
            if not trivial_underapprox:
                theta_underline_1 = hist_suffix_sum_smallest_covering_roi[underapprox_bin] - (
                    upper_ys - lower_ys) * (upper_xs - lower_xs) * available_coords * available_coords + box_area
                theta_underline_2 = hist_suffix_sum_largest_covered_by_roi[underapprox_bin]
                theta_underline = max(theta_underline_1, theta_underline_2)

            if theta_underline > v:
                res.append((theta_underline, image_idx))
                count[2] += 1
            else:
                # Only now does MaskSearch compute the actual theta
                cam = cam_map[image_idx]
                if compression is not None:
                    if compression == "jpeg" or compression == "JPEG" or compression == "png" or compression == "PNG":
                        cam = np.frombuffer(cam, dtype=np.uint8)
                        cam = cv2.imdecode(cam, cv2.IMREAD_COLOR)
                        cam = cam[:, :, 0]
                    else:
                        raise ValueError("Unknown compression method")
                    cam = cam.reshape((cam_size_y, cam_size_x))
                    cam = np.float32(cam) / 255.0
                area = compute_area_for_cam(
                    cam, threshold=threshold, subregion=(
                        x - 1, y - 1, w + 1, h + 1))
                if area > v:
                    res.append((area, image_idx))

    return count, res


def get_global_vars(dataset):
    if dataset == "imagenet":
        cam_map = imagenet_mp_vars.cam_map
        object_detection_map = imagenet_mp_vars.object_detection_map
        cam_size_y = imagenet_mp_vars.cam_size_y
        cam_size_x = imagenet_mp_vars.cam_size_x
        available_coords = imagenet_mp_vars.available_coords
        hist_size = imagenet_mp_vars.hist_size
        bin_width = imagenet_mp_vars.bin_width
    else:
        cam_map = wilds_mp_vars.cam_map
        object_detection_map = wilds_mp_vars.object_detection_map
        cam_size_y = wilds_mp_vars.cam_size_y
        cam_size_x = wilds_mp_vars.cam_size_x
        available_coords = wilds_mp_vars.available_coords
        hist_size = wilds_mp_vars.hist_size
        bin_width = wilds_mp_vars.bin_width
    return cam_map, object_detection_map, cam_size_y, cam_size_x, available_coords, hist_size, bin_width


def mp_get_images_helper(args):
    return mp_get_images_satisfying_filter(*args)


def mp_get_images_satisfying_filter(
        dataset,
        examples,
        threshold,
        region,
        v,
        region_area_threshold,
        ignore_zero_area_region,
        compression,
        processes=1):
    if processes > 1:
        chunk_size = int(np.ceil(len(examples) / processes))
        example_chunks = [examples[i:i + chunk_size] if i + chunk_size < len(
            examples) else examples[i:] for i in range(0, len(examples), chunk_size)]

        args = [
            (dataset,
             chunk,
             threshold,
             region,
             v,
             region_area_threshold,
             ignore_zero_area_region,
             compression) for chunk in example_chunks]

        with Pool(processes) as pool:
            results = pool.map(mp_get_images_helper, args)

        count = np.array([0, 0, 0], dtype=np.int32)
        area_images = []
        for c, a in results:
            count += c
            area_images.extend(a)

        return count, area_images

    cam_map, object_detection_map, cam_size_y, cam_size_x, available_coords, hist_size, bin_width = get_global_vars(
        dataset)

    if dataset == "imagenet":
        in_memory_index_suffix = imagenet_mp_vars.in_memory_index_suffix
    else:
        in_memory_index_suffix = wilds_mp_vars.in_memory_index_suffix

    count, area_images = get_images_satisfying_filter(dataset, cam_map, object_detection_map, bin_width, hist_size, cam_size_y, cam_size_x,
                                                      examples, threshold, region, v, region_area_threshold, ignore_zero_area_region, in_memory_index_suffix, available_coords, compression)

    return count, area_images


def mp_get_images_based_on_area_filter(
        dataset,
        examples,
        threshold,
        region,
        v,
        processes,
        region_area_threshold=None,
        ignore_zero_area_region=True,
        reverse=False,
        visualize=False,
        available_coords=None,
        compression=None):

    count, area_images = mp_get_images_satisfying_filter(
        dataset, examples, threshold, region, v, region_area_threshold, ignore_zero_area_region, compression, processes=processes)
    area_images = sorted(area_images, key=lambda x: (x[0], x[1]))

    return count, area_images


def get_images_based_on_area_filter(
        dataset,
        cam_map,
        object_detection_map,
        bin_width,
        hist_size,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        v,
        in_memory_index_suffix,
        region_area_threshold=None,
        ignore_zero_area_region=True,
        reverse=False,
        visualize=False,
        available_coords=None,
        compression=None):

    count, area_images = get_images_satisfying_filter(dataset, cam_map, object_detection_map, bin_width, hist_size, cam_size_y, cam_size_x,
                                                      examples, threshold, region, v, region_area_threshold, ignore_zero_area_region, in_memory_index_suffix, available_coords, compression)
    area_images = sorted(area_images, key=lambda x: (x[0], x[1]))

    if not visualize:
        return count, area_images


def naive_get_images_satisfying_filter(
        cam_map,
        object_detection_map,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        v,
        region_area_threshold,
        ignore_zero_area_region,
        compression=None,
        reverse=False,
        visualize=False):
    start = time.time()
    metric_map = get_area_map(
        cam_map,
        object_detection_map,
        cam_size_y,
        cam_size_x,
        examples,
        threshold,
        region,
        compression)
    tot = len(examples)
    metric_list = []
    for i in range(tot):
        image_id = examples[i]
        if isinstance(region, tuple):
            x, y, w, h = region
        else:
            (x, y, w, h) = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_id)

        box_area = w * h
        if (ignore_zero_area_region and box_area <= 0) or (
                region_area_threshold is not None and box_area < region_area_threshold):
            metric = -1
        else:
            metric = metric_map[image_id]

        metric_list.append(metric)

    end = time.time()

    area_images = []
    for i in range(tot):
        metric = metric_list[i]
        if metric < 0:
            continue
        if metric > v:
            area_images.append((metric, examples[i]))

    area_images = sorted(area_images, key=lambda x: (x[0], x[1]))
    return area_images


def mp_get_images_satisfying_filter_and_build_index_worker(
        image_idx, dataset, region, compression, threshold, area_map, index):

    cam_map, object_detection_map, cam_size_y, cam_size_x, available_coords, hist_size, bin_width = get_global_vars(
        dataset)

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
        dtype=np.int32)

    if isinstance(region, tuple):
        x, y, w, h = region
    else:
        (x, y, w, h) = get_object_region(
            object_detection_map, cam_size_y, cam_size_x, image_idx)

    cam = cam_map[image_idx]
    if compression is not None:
        if compression in ("jpeg", "JPEG", "png", "PNG"):
            cam = np.frombuffer(cam, dtype=np.uint8)
            cam = cv2.imdecode(cam, cv2.IMREAD_COLOR)
            cam = cam[:, :, 0]
        else:
            raise ValueError("Unknown compression method")
        cam = cam.reshape((cam_size_y, cam_size_x))
        cam = np.float32(cam) / 255.0

    area = compute_area_for_cam(cam, threshold, (x, y, w + 1, h + 1))
    area_map[image_idx] = area

    grayscale[1:, 1:] = np.uint8(255 * cam)
    bins_for_grayscale = grayscale.ravel() // bin_width
    hist.fill(0)
    hist[y_mesh, x_mesh, bins_for_grayscale] = 1
    full_prefix = np.cumsum(np.cumsum(hist, axis=0), axis=1)
    hist_prefix = full_prefix[0:cam_size_y +
                              1:available_coords, 0:cam_size_x +
                              1:available_coords, :]
    index[image_idx] = np.cumsum(hist_prefix[:, :, ::-1], axis=2)[:, :, ::-1]


def mp_get_images_satisfying_filter_and_build_index(
        dataset,
        examples,
        threshold,
        region,
        v,
        region_area_threshold,
        ignore_zero_area_region,
        compression=None,
        reverse=False,
        visualize=False,
        num_processes=mp.cpu_count()):

    cam_map, object_detection_map, cam_size_y, cam_size_x, available_coords, hist_size, bin_width = get_global_vars(
        dataset)

    if dataset == "imagenet":
        in_memory_index_suffix = imagenet_mp_vars.in_memory_index_suffix
    else:
        in_memory_index_suffix = wilds_mp_vars.in_memory_index_suffix

    with mp.Manager() as manager:
        area_map = manager.dict()
        index = manager.dict()
        process_args = (
            dataset,
            region,
            compression,
            threshold,
            area_map,
            index)

        with mp.Pool(num_processes) as pool:
            pool.starmap(mp_get_images_satisfying_filter_and_build_index_worker, [
                         (image_idx, *process_args) for image_idx in examples])

        area_map = dict(area_map)
        index = dict(index)
        for image_idx, chi in index.items():
            if dataset == "imagenet":
                generic_image_id = int(image_idx)
            else:
                generic_image_id = get_generic_image_id_for_wilds(
                    image_idx.split("_")[0], image_idx.split("_")[-1])
            in_memory_index_suffix[generic_image_id] = chi

    tot = len(examples)
    metric_list = []
    for i in range(tot):
        image_id = examples[i]
        if isinstance(region, tuple):
            x, y, w, h = region
        else:
            (x, y, w, h) = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_idx)

        box_area = w * h
        if (ignore_zero_area_region and box_area <= 0) or (
                region_area_threshold is not None and box_area < region_area_threshold):
            metric = -1
        else:
            metric = area_map[image_id]
        metric_list.append(metric)

    area_images = []
    for i in range(tot):
        metric = metric_list[i]
        if metric < 0:
            continue
        if metric > v:
            area_images.append((metric, examples[i]))

    area_images = sorted(area_images, key=lambda x: (x[0], x[1]))
    return area_images


def get_max_mean_in_area_across_models_in_memory(
        dataset,
        object_detection_map,
        cam_size_y,
        cam_size_x,
        bin_width,
        examples,
        cam_maps,
        thresholds,
        in_memory_index_suffixes,
        region,
        k,
        region_area_threshold,
        ignore_zero_area_region,
        reverse,
        available_coords,
        compression):
    """returns a list of (area, image_idx)"""

    assert len(cam_maps) == len(thresholds) == len(in_memory_index_suffixes)

    if region != "object":
        x, y, w, h = region
        x += 1
        y += 1
        lower_ys, upper_ys, lower_xs, upper_xs = get_smallest_region_covering_roi_using_available_coords(
            cam_size_y, cam_size_x, available_coords, x, y, w, h)
        lower_yl, upper_yl, lower_xl, upper_xl = get_largest_region_covered_by_roi_using_available_coords(
            cam_size_y, cam_size_x, available_coords, x, y, w, h)

    grayscale_thresholds = [int(threshold * 255) for threshold in thresholds]

    overapprox_bins = [
        grayscale_threshold //
        bin_width for grayscale_threshold in grayscale_thresholds]

    tot = len(examples)
    heap = []
    if reverse:
        raise NotImplementedError
    else:
        factor = 1

    count = 0
    t = 0.

    for img_offset in range(tot):
        image_idx = examples[img_offset]
        if region == "object":
            x, y, w, h = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_idx)
            x += 1
            y += 1
            lower_ys, upper_ys, lower_xs, upper_xs = get_smallest_region_covering_roi_using_available_coords(
                cam_size_y, cam_size_x, available_coords, x, y, w, h)
            lower_yl, upper_yl, lower_xl, upper_xl = get_largest_region_covered_by_roi_using_available_coords(
                cam_size_y, cam_size_x, available_coords, x, y, w, h)
        if ignore_zero_area_region and (w == 0 or h == 0):
            count += 1
            continue
        box_area = w * h
        if region_area_threshold is not None and box_area < region_area_threshold:
            count += 1
            continue

        # Construct hist_suffix_sum for region (x, y, x + w, y + h)
        # st = time.time()
        if dataset == "imagenet":
            generic_image_id = int(image_idx)
        else:
            generic_image_id = get_generic_image_id_for_wilds(
                image_idx.split("_")[0], image_idx.split("_")[-1])

        if reverse:
            raise NotImplementedError
        else:
            # Overapproximate area mean
            overapproximate_area_sum = 0
            overapproximate_area_list = []
            for i in range(len(cam_maps)):
                hist_prefix_suffix = in_memory_index_suffixes[i][generic_image_id][:]
                hist_suffix_sum_smallest_covering_roi = hist_prefix_suffix[upper_ys,
                                                                           upper_xs] - hist_prefix_suffix[upper_ys,
                                                                                                          lower_xs] - hist_prefix_suffix[lower_ys,
                                                                                                                                         upper_xs] + hist_prefix_suffix[lower_ys,
                                                                                                                                                                        lower_xs]
                hist_suffix_sum_largest_covered_by_roi = hist_prefix_suffix[upper_yl,
                                                                            upper_xl] - hist_prefix_suffix[upper_yl,
                                                                                                           lower_xl] - hist_prefix_suffix[lower_yl,
                                                                                                                                          upper_xl] + hist_prefix_suffix[lower_yl,
                                                                                                                                                                         lower_xl]

                theta_bar_1 = hist_suffix_sum_smallest_covering_roi[overapprox_bins[i]]
                theta_bar_2 = hist_suffix_sum_largest_covered_by_roi[overapprox_bins[i]] + box_area - (
                    upper_yl - lower_yl) * (upper_xl - lower_xl) * available_coords * available_coords
                theta_bar = min(theta_bar_1, theta_bar_2)

                overapproximate_area = theta_bar
                overapproximate_area_list.append(overapproximate_area)
                overapproximate_area_sum += overapproximate_area

            approximate_area_mean = overapproximate_area_sum / len(cam_maps)

        if len(heap) < k or factor * \
                approximate_area_mean / box_area > heap[0][0]:
            cams = [cam_map[image_idx] for cam_map in cam_maps]
            if compression is not None:
                if compression == "jpeg" or compression == "JPEG" or compression == "png" or compression == "PNG":
                    for i in range(len(cams)):
                        cams[i] = np.frombuffer(cams[i], dtype=np.uint8)
                        cams[i] = cv2.imdecode(cams[i], cv2.IMREAD_COLOR)
                        cams[i] = cams[i][:, :, 0]
                        cams[i] = cams[i].reshape((cam_size_y, cam_size_x))
                else:
                    raise ValueError("Unknown compression method")

                for i in range(len(cams)):
                    cams[i] = np.float32(cams[i]) / 255.0

            areas = [
                compute_area_for_cam(
                    cam,
                    threshold=threshold,
                    subregion=(
                        x - 1,
                        y - 1,
                        w + 1,
                        h + 1)) for cam,
                threshold in zip(
                    cams,
                    thresholds)]
            area_mean = np.mean(areas)

            if len(heap) < k:
                heapq.heappush(
                    heap,
                    (factor *
                     area_mean /
                     box_area,
                     factor *
                     area_mean,
                     image_idx))
            elif factor * area_mean / box_area > heap[0][0]:
                heapq.heappushpop(
                    heap, (factor * area_mean / box_area, factor * area_mean, image_idx))
        else:
            count += 1

    return count, sorted([(factor * item[0], item[2])
                         for item in heap], reverse=not reverse)


def get_max_udf_in_sub_region_in_memory(
        dataset,
        get_images_based_on_udf,
        object_detection_map,
        cam_size_y,
        cam_size_x,
        bin_width,
        examples,
        cam_maps,
        thresholds,
        index_filenames,
        region,
        k=25,
        region_area_threshold=None,
        ignore_zero_area_region=True,
        reverse=False,
        visualize=False,
        available_coords=None,
        compression=None):

    start = time.time()
    count, area_images = get_images_based_on_udf(dataset, object_detection_map, cam_size_y, cam_size_x, bin_width, examples, cam_maps,
                                                 thresholds, index_filenames, region, k, region_area_threshold, ignore_zero_area_region, reverse, available_coords, compression)
    end = time.time()
    if visualize:
        raise NotImplementedError
    return count, area_images


def get_area_mean_map(
        object_detection_map,
        cam_size_y,
        cam_size_x,
        examples,
        cam_maps,
        thresholds,
        region,
        compression):
    udf_map = dict()
    if isinstance(region, tuple):
        x, y, w, h = region

    count = 0
    for image_idx in examples:
        if not isinstance(region, tuple):
            (x, y, w, h) = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_idx)
        cams = [cam_map[image_idx] for cam_map in cam_maps]
        if compression is not None:
            if compression == "jpeg" or compression == "JPEG" or compression == "png" or compression == "PNG":
                for i in range(len(cams)):
                    cams[i] = np.frombuffer(cams[i], dtype=np.uint8)
                    cams[i] = cv2.imdecode(cams[i], cv2.IMREAD_COLOR)
                    cams[i] = cams[i][:, :, 0]
                    cams[i] = cams[i].reshape((cam_size_y, cam_size_x))
            else:
                raise ValueError("Unknown compression method")

            for i in range(len(cams)):
                cams[i] = np.float32(cams[i]) / 255.0

        areas = [
            compute_area_for_cam(
                cam,
                threshold,
                (x,
                 y,
                 w + 1,
                 h + 1)) for cam,
            threshold in zip(
                cams,
                thresholds)]
        udf_map[image_idx] = mean(areas)
        count += 1

    return udf_map


def naive_get_max_udf(
        get_udf_map,
        object_detection_map,
        cam_size_y,
        cam_size_x,
        examples,
        cam_maps,
        thresholds,
        region,
        k,
        region_area_threshold,
        ignore_zero_area_region,
        compression=None,
        reverse=False,
        visualize=False):
    start = time.time()
    metric_map = get_udf_map(
        object_detection_map,
        cam_size_y,
        cam_size_x,
        examples,
        cam_maps,
        thresholds,
        region,
        compression)
    tot = len(examples)
    metric_list = []
    for i in range(tot):
        image_id = examples[i]
        if isinstance(region, tuple):
            x, y, w, h = region
        else:
            (x, y, w, h) = get_object_region(
                object_detection_map, cam_size_y, cam_size_x, image_id)

        box_area = w * h
        if (ignore_zero_area_region and box_area <= 0) or (
                region_area_threshold is not None and box_area < region_area_threshold):
            if reverse:
                metric = int(1e15)
            else:
                metric = -1
        else:
            metric = metric_map[image_id] / box_area
        metric_list.append(metric)
    order = argsort(metric_list)
    if not reverse:
        order = order[::-1]
    end = time.time()
    tot = min(tot, k)
    if not visualize:
        return sorted([(metric_list[i], examples[i])
                      for i in order[:tot]], reverse=not reverse)
    else:
        raise NotImplementedError


def compute_areas(
        dataset,
        cam_map,
        object_detection_map,
        cam_size_y,
        cam_size_x,
        examples,
        thresholds,
        region,
        filename):
    if isinstance(region, tuple):
        x, y, w, h = region

    with open(filename, "w") as f:
        f.write("image_idx, region, region_area, threshold, area\n")
        for image_idx in examples:
            if not isinstance(region, tuple):
                (x, y, w, h) = get_object_region(
                    object_detection_map, cam_size_y, cam_size_x, image_idx)
            cam = cam_map[image_idx]
            grid = cam[y: y + h + 1, x: x + w + 1]
            box_area = w * h
            for threshold in thresholds:
                area = np.count_nonzero(grid > threshold)
                print(
                    f"image_idx: {image_idx}, region: {region}, box_area: {box_area}, threshold: {round(threshold, 1)}, area: {area}")
                f.write(
                    f"{image_idx}, {region}, {box_area}, {round(threshold, 1)}, {area}\n")
                f.flush()


def compute_bounds(
        dataset,
        object_detection_map,
        bin_width,
        hist_size,
        cam_size_y,
        cam_size_x,
        examples,
        thresholds,
        region,
        in_memory_index_suffix,
        available_coords,
        filename):

    if region != "object":
        x, y, w, h = region
        x += 1
        y += 1
        lower_ys, upper_ys, lower_xs, upper_xs = get_smallest_region_covering_roi_using_available_coords(
            cam_size_y, cam_size_x, available_coords, x, y, w, h)
        lower_yl, upper_yl, lower_xl, upper_xl = get_largest_region_covered_by_roi_using_available_coords(
            cam_size_y, cam_size_x, available_coords, x, y, w, h)

    tot = len(examples)

    with open(filename, "w") as f:
        f.write(
            "image_idx, region, region_area, threshold, theta_bar, theta_underline\n")
        for offset in range(tot):
            i = offset
            image_idx = examples[i]
            if region == "object":
                x, y, w, h = get_object_region(
                    object_detection_map, cam_size_y, cam_size_x, image_idx)
                x += 1
                y += 1
            box_area = w * h
            if region == "object":
                lower_ys, upper_ys, lower_xs, upper_xs = get_smallest_region_covering_roi_using_available_coords(
                    cam_size_y, cam_size_x, available_coords, x, y, w, h)
                lower_yl, upper_yl, lower_xl, upper_xl = get_largest_region_covered_by_roi_using_available_coords(
                    cam_size_y, cam_size_x, available_coords, x, y, w, h)

            if dataset == "imagenet":
                generic_image_id = int(image_idx)
            else:
                generic_image_id = get_generic_image_id_for_wilds(
                    image_idx.split("_")[0], image_idx.split("_")[-1])
            hist_prefix_suffix = in_memory_index_suffix[generic_image_id][:]

            hist_suffix_sum_smallest_covering_roi = hist_prefix_suffix[upper_ys,
                                                                       upper_xs] - hist_prefix_suffix[upper_ys,
                                                                                                      lower_xs] - hist_prefix_suffix[lower_ys,
                                                                                                                                     upper_xs] + hist_prefix_suffix[lower_ys,
                                                                                                                                                                    lower_xs]
            hist_suffix_sum_largest_covered_by_roi = hist_prefix_suffix[upper_yl,
                                                                        upper_xl] - hist_prefix_suffix[upper_yl,
                                                                                                       lower_xl] - hist_prefix_suffix[lower_yl,
                                                                                                                                      upper_xl] + hist_prefix_suffix[lower_yl,
                                                                                                                                                                     lower_xl]

            for threshold in thresholds:
                grayscale_threshold = int(threshold * 255)
                overapprox_bin = grayscale_threshold // bin_width
                underapprox_bin = (grayscale_threshold // bin_width) + 1

                theta_bar_1 = hist_suffix_sum_smallest_covering_roi[overapprox_bin]
                if theta_bar_1 < 0:
                    theta_bar_1 = 100000000
                theta_bar_2 = hist_suffix_sum_largest_covered_by_roi[overapprox_bin] + box_area - (
                    upper_yl - lower_yl) * (upper_xl - lower_xl) * available_coords * available_coords
                if theta_bar_2 < 0:
                    theta_bar_2 = 100000000
                theta_bar = min(theta_bar_1, theta_bar_2)

                if underapprox_bin == hist_size:
                    theta_underline = 0
                else:
                    if - (upper_ys - lower_ys) * (upper_xs - lower_xs) * \
                            available_coords * available_coords + box_area > 0:
                        theta_underline_1 = 0
                    else:
                        theta_underline_1 = hist_suffix_sum_smallest_covering_roi[underapprox_bin] - (
                            upper_ys - lower_ys) * (upper_xs - lower_xs) * available_coords * available_coords + box_area
                    theta_underline_2 = hist_suffix_sum_largest_covered_by_roi[underapprox_bin]
                    theta_underline = max(theta_underline_1, theta_underline_2)

                print(
                    f"image_idx: {image_idx}, region: {region}, box_area: {box_area}, threshold: {round(threshold, 1)}, theta_bar: {theta_bar}, theta_underline: {theta_underline}")
                f.write(
                    f"{image_idx}, {region}, {box_area}, {round(threshold, 1)}, {theta_bar}, {theta_underline}\n")
                f.flush()
