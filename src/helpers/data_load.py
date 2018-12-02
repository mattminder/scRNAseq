import numpy as np


def load_data(data_path):
    """Handles loading of data files as exported by R."""

    data = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=float)
    feature_names = np.genfromtxt(data_path, delimiter=",", max_rows=1, dtype=str)
    cell_names = np.genfromtxt(data_path, delimiter=",", usecols=0, skip_header=1, dtype=str)

    return data, feature_names, cell_names


def load_response(data_path):
    """Handles loading of response file as exported by R."""

    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=int, usecols=1)
    cell_names = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=0)

    return y, cell_names



