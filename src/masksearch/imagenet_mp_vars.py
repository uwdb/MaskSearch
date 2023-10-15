from masksearch.masksearch import *
import numpy as np


def init():
    import torch
    import torchvision.transforms as transforms
    from pytorch_grad_cam import (
        AblationCAM,
        EigenGradCAM,
        GradCAM,
        GradCAMPlusPlus,
        HiResCAM,
        LayerCAM,
        RandomCAM)
    import shelve
    import time
    import meerkat as mk
    import time
    import numpy as np
    import torch
    from torchvision.models import resnet18, resnet34, resnet50, resnet101, vgg19
    import torchvision.transforms as transforms

    # Make every variable global
    global device
    global dataset
    global dp
    global class_idx_column
    global img_column
    global model
    global transform
    global pred_map
    global label_map
    global dataset_examples
    global cam_size_y
    global cam_size_x
    global total_images
    global hist_size
    global hist_edges
    global bin_width
    global available_coords
    global in_memory_index_suffix
    global cam_map
    global object_detection_map

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = "imagenet"
    dp = mk.datasets.get(dataset, download_mode="reuse")

    class_idx_column = "class_idx"
    img_column = "image"

    print(len(dp))

    import torch
    from torchvision.models import resnet18, resnet34, resnet50, resnet101, vgg19
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    dp["input"] = dp[img_column].to_lambda(transform)

    dataset_examples = []
    for i in range(len(dp)):
        dataset_examples.append(f"{i}")

    cam_size_y = 224
    cam_size_x = 224
    total_images = 1331167
    assert len(dp) == total_images

    object_detection_map = load_object_region_index_in_memory(
        dataset_examples, '/data/explain_imagenet/shelves/object_detection_map.shelve')

    hist_size = 16
    hist_edges = []
    bin_width = 256 // hist_size
    for i in range(1, hist_size):
        hist_edges.append(bin_width * i)
    hist_edges.append(256)
    print(hist_edges)

    available_coords = 28

    cam_map = shelve.open(
        '/data/explain_imagenet/shelves/imagenet_cam_map.shelve')


def init_load_index(filename):
    global in_memory_index_suffix

    if filename is None:
        filename = f'/data/explain_imagenet/npy/monolithic_imagenet_cam_hist_prefix_{hist_size}_in_memory_available_coords_{available_coords}_suffix.npy'

    in_memory_index_suffix = np.load(filename)

    assert in_memory_index_suffix.dtype == np.int64


def init_incremental_indexing():
    global in_memory_index_suffix
    in_memory_index_suffix = np.zeros(
        (total_images,
         (cam_size_y // available_coords) + 1,
            (cam_size_x // available_coords) + 1,
            hist_size),
        dtype=np.int64)
