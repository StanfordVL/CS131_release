"""Utilities for downloading the face dataset.
"""

import os

import numpy as np
from skimage import io
from skimage import img_as_float


def load_dataset(data_dir, train=True, as_grey=False, shuffle=True):
    """ Load faces dataset

    The face dataset for CS131 assignment.
    The directory containing the dataset has the following structure:

        faces/
            train/
                angelina jolie/
                ...
            test/
                angelina jolie/
                ...

    Args:
        data_dir - Directory containing the face datset.
        train - If True, load training data. Load test data otherwise.
        as_grey - If True, open images as grayscale.
        shuffle - shuffle dataset

    Returns:
        X - array of N images (N, 64, 64, 3)
        y - array of class labels (N,)
        class_names - list of class names (string)
    """
    y = []
    X = []
    class_names = []

    if train:
        data_dir = os.path.join(data_dir, 'train')
    else:
        data_dir = os.path.join(data_dir, 'test')

    for i, cls in enumerate(sorted(os.listdir(data_dir))):
        for img_file in os.listdir(os.path.join(data_dir, cls)):
            img_path = os.path.join(data_dir, cls, img_file)
            img = img_as_float(io.imread(img_path, as_grey=as_grey))
            X.append(img)
            y.append(i)
        class_names.append(cls)

    # Convert list of imgs and labels into array
    X = np.array(X)
    y = np.array(y)

    if shuffle:
        idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        X = X[idxs]
        y = y[idxs]

    return np.array(X), np.array(y), class_names
