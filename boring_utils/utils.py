import os
from pathlib import Path
import inspect
import numpy as np
import matplotlib.pyplot as plt
import torch

from boring_utils.colorprint import *


def set_seed(seed, strict=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # starting with pytorch 1.8, we don't need to set the seed for all devices
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(return_str=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # apple silicon GPU
    else:
        device = torch.device("cpu")
    return device if not return_str else device.type


def check_exists(data_folder, dataset_name):
    return os.path.exists(os.path.join(data_folder, dataset_name))


def mkdir(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=False)


def init_graph(figsize=(10, 10), dpi=100):
    plt.style.use(["dark_background", "bmh"])
    plt.rc("axes", facecolor="k")
    plt.rc("figure", facecolor="k")
    plt.rc("figure", figsize=figsize, dpi=dpi)
    plt.rc("font", size=15)


def plot_data(X, y, figsize=(16, 16), save_fig=False, fig_path="temp.png"):
    """plot data generated from make_data.py"""
    plt.figure(figsize=figsize)
    plt.title("Dataset")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")

    if save_fig:
        plt.savefig(fig_path)

    plt.show()


def plot_loss(
    train_losses,
    val_losses,
    num_epochs,
    figsize=(20, 10),
    save_fig=False,
    fig_path="temp.png",
):
    fig = plt.figure(figsize=figsize)
    plt.plot(np.arange(1, num_epochs + 1), train_losses, label="Train loss")
    plt.plot(np.arange(1, num_epochs + 1), val_losses, label="Validation loss")
    plt.xlabel("Loss")
    plt.ylabel("Epochs")
    plt.legend(loc="upper right")

    if save_fig:
        plt.savefig(fig_path)

    plt.show()


def reset_model_weights(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def get_layers(func):
    '''
    Usage:
      from boring_nn import pe
      pe_layers = get_layers(pe)
      pe_name = 'SinusoidalPositionalEncoding'
      pos_encoding = pe_layers[pe_name](d_model, dropout, max_len)
    '''
    func_layers = {}
    for name, obj in inspect.getmembers(func):
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
            # func_layers[name.lower()] = obj
            func_layers[name] = obj
    return func_layers
