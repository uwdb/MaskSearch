from masksearch.masksearch import *
import sys
sys.argv = ['']


def init():

    import argparse
    import json
    import pickle
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    from models.initializer import initialize_model
    from pytorch_grad_cam import (
        AblationCAM,
        EigenGradCAM,
        GradCAM,
        GradCAMPlusPlus,
        HiResCAM,
        LayerCAM,
        RandomCAM)
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import wilds
    from wilds import get_dataset
    from wilds.common.data_loaders import get_eval_loader, get_train_loader
    import shelve

    global dataset
    global model
    global device
    global id_val_loader
    global ood_val_loader
    global cam_map
    global pred_map
    global label_map
    global cam_size_y
    global cam_size_x
    global id_total
    global ood_total
    global dataset_examples
    global hist_size
    global hist_edges
    global bin_width
    global available_coords
    global object_detection_map
    global total_images

    dataset = get_dataset(
        dataset="iwildcam",
        download=False,
        root_dir="/home/ubuntu/data",
    )

    """
    {'train': 'Train', 'val': 'Validation (OOD/Trans)',
        'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
        'id_test': 'Test (ID/Cis)'}
    """

    # Get the ID validation set
    id_val_data = dataset.get_subset(
        "id_val",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    id_val_loader = get_eval_loader("standard", id_val_data, batch_size=16)

    ood_val_data = dataset.get_subset(
        "val",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    ood_val_loader = get_eval_loader("standard", ood_val_data, batch_size=16)

    # Load from disk
    cam_map = shelve.open(
        '/data/explain_wilds/shelves/id_ood_val_cam_map.shelve')
    with open('/data/explain_wilds/id_ood_val_pred.pkl', 'rb') as f:
        pred_map = pickle.load(f)
    with open('/data/explain_wilds/id_ood_val_label.pkl', 'rb') as f:
        label_map = pickle.load(f)

    cam_size_y = 448
    cam_size_x = 448

    id_total = 7314
    ood_total = 14961
    dataset_examples = []
    for distribution, image_total in zip(
            ["id_val", "ood_val"], [id_total, ood_total]):
        for image_idx in range(1, 1 + image_total):
            dataset_examples.append(f"{distribution}_{image_idx}")

    total_images = len(dataset_examples)
    hist_size = 16
    hist_edges = []
    bin_width = 256 // hist_size
    for i in range(1, hist_size):
        hist_edges.append(bin_width * i)
    hist_edges.append(256)
    print(hist_edges)

    available_coords = 64

    object_detection_map = load_object_region_index_in_memory(
        dataset_examples,
        '/data/explain_wilds/shelves/id_ood_val_object_detection_map.shelve')


def init_load_index(filename):
    global in_memory_index_suffix
    if filename is None:
        filename = f'/data/explain_wilds/npy/id_ood_val_cam_hist_prefix_{hist_size}_in_memory_available_coords_{available_coords}_suffix.npy'
    in_memory_index_suffix = np.load(filename)

    assert in_memory_index_suffix.dtype == np.int64


def init_incremental_indexing():
    global in_memory_index_suffix
    in_memory_index_suffix = np.zeros(
        (total_images + 1,
         (cam_size_y // available_coords) + 1,
            (cam_size_x // available_coords) + 1,
            hist_size),
        dtype=np.int64)
