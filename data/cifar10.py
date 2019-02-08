"""
This module is used to download, extract, load, and delete the CIFAR10 dataset.
"""
import os
import shutil
import tarfile
import urllib.request
import numpy as np


def download_and_extract(verbose=True):
    """
    Download the CIFAR10 data set and extract it into the folder ./data/CIFAR10

    :param verbose: Toggles verbose printing
    """
    dest = './data/CIFAR10'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = url.split('/')[-1]
    filepath = os.path.join(dest, filename)
    if not os.path.exists(dest):
        if not os.path.exists(filepath):
            os.makedirs(dest)

            def progress(count, block_size, total_size):
                """
                Prints a progress bar for the CIFAR10 download
                """
                perc = float(count * block_size) / float(total_size)
                width = 80 - len(">  Downloading {} {:.1f}%".format(
                    filename, perc * 100.0))
                if verbose:
                    print(
                        "\r>> Downloading {} [{}] {:.1f}%".format(
                            filename,
                            ('=' * int(perc * width)) + ">" + (' ' * int(
                                (1.0 - perc) * width)), perc * 100.0),
                        end='')

            filepath, _ = urllib.request.urlretrieve(url, filepath, progress)
            if verbose:
                print()
            statinfo = os.stat(filepath)
            if verbose:
                print("   Successfully downloaded", filename, statinfo.st_size,
                      'bytes.')
    extract_dir = os.path.join(dest, 'cifar-10-batches-py')
    if not os.path.exists(os.path.join(dest, 'batches.meta')):
        if not os.path.exists(extract_dir):
            if verbose:
                print(">> Extracting {}".format(filepath))
            tarfile.open(filepath, 'r:gz').extractall(dest)
            if verbose:
                print("   Successfully extracted {}".format(filepath))
        files = os.listdir(extract_dir)
        if verbose:
            print(">> Moving files")
        for file in files:
            shutil.move(
                os.path.join(extract_dir, file), os.path.join(dest, file))
        os.removedirs(extract_dir)
        if verbose:
            print("   Successfully moved files")


def delete(verbose=True):
    """
    Deletes all downloaded data

    :param verbose: Toggles verbose printing
    """
    dest = './data/CIFAR10'
    if os.path.exists(dest):
        if verbose:
            print(">> Deleting {}".format(dest))
        shutil.rmtree(dest)
        if verbose:
            print("   Successfully deleted {}".format(dest))


def unpickle(file):
    """
    Opens and reads the binary CIFAR10 batch data

    :param file: Path to file to unpickle
    :returns: Data from pickeled file
    """
    import pickle
    with open(file, 'rb') as binary:
        vals = pickle.load(binary, encoding='bytes')
        return vals
    return None


def load(file, verbose=True):
    """
    Loads a CIFAR10 data file, and parses the binary into two numpy.ndarrays.

    :param file: Specifies the training batch, or some other data batch.
    verbose[in] Toggles verbose printing.
    :param verbose: Toggles verbose printing.
    :returns: A pair of data, the first is the input data, and the second is the matching labels.
    """
    download_and_extract(verbose)
    if isinstance(file, int):
        abs_file = os.path.abspath("./data/CIFAR10/data_batch_{}".format(file))
    else:
        abs_file = os.path.abspath("./data/CIFAR10/{}".format(file))
    data_dict = unpickle(abs_file)
    x_data = np.asarray(data_dict[b'data'].T).astype("uint8")
    y_raw = np.asarray(data_dict[b'labels'])
    y_data = np.zeros((10, 10000))
    for i in range(10000):
        y_data[y_raw[i], i] = 1.0
    # labels = np.asarray(data_dict[b'filenames']).astype("str")
    return x_data, y_data.T
